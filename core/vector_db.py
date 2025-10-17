"""
Vector database adapter (refactored for modularity)
Keeps your existing QdrantAdapter but cleaner
"""
import time
import os
import threading
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
    SparseVector,
    PointStruct
)
from sentence_transformers import SentenceTransformer
import uuid


class QdrantAdapter:
    """
    Qdrant vector database adapter with adaptive retrieval
    (Qwen3 + SPLADE + HyDE)
    """
    
    def __init__(
        self,
        persist_dir: str = "./data/qdrant_db",
        collection_name: str = "conversations",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        llm_engine = None
    ):
        """Initialize Qdrant with adaptive retrieval"""
        
        # Connect to Qdrant
        if url and api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
            print(f"Connected to Qdrant Cloud: {url}")
        else:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = QdrantClient(path=persist_dir)
            print(f"Connected to local Qdrant: {persist_dir}")
        
        self.collection_name = collection_name
        self._lock = threading.RLock()
        self.llm_engine = llm_engine
        
        # Load Qwen3 for dense embeddings
        print("Loading Qwen3-Embedding-0.6B...")
        self.dense_encoder = SentenceTransformer(
            'Qwen/Qwen3-Embedding-0.6B',
            device='cuda'
        )
        self.dense_size = 768
        print("✅ Qwen3 loaded")
        
        # Load SPLADE for semantic sparse retrieval
        print("Loading SPLADE...")
        try:
            from splade.models.transformer_rep import Splade
            self.splade = Splade(
                'naver/splade-cocondenser-ensembledistil',
                device='cuda'
            )
            print("✅ SPLADE loaded")
        except Exception as e:
            print(f"⚠️  SPLADE failed: {e}, using BM25 fallback")
            self.splade = None
        
        # Create collection if not exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection with proper schema"""
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' exists")
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.dense_size,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams()
                    )
                }
            )
            print(f"Created collection '{self.collection_name}'")
        
        # Create conversation_id index
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="conversation_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except:
            pass
    
    def add(
        self,
        conversation_id: str,
        text: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add text to vector database"""
        try:
            # Dense embedding
            dense_vector = self.dense_encoder.encode(text).tolist()
            
            # Sparse vector
            if self.splade:
                splade_output = self.splade(text, return_sparse=True)
                if isinstance(splade_output, dict):
                    sparse_vector = SparseVector(
                        indices=list(splade_output.keys()),
                        values=list(splade_output.values())
                    )
                else:
                    sparse_vector = self._generate_sparse_vector_bm25(text)
            else:
                sparse_vector = self._generate_sparse_vector_bm25(text)
            
            # Prepare payload
            payload = {
                'text': text,
                'conversation_id': conversation_id,
                'timestamp': time.time()
            }
            if metadata:
                payload.update(metadata)
            
            # Insert
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector
                },
                payload=payload
            )
            
            with self._lock:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )
            
            return True
            
        except Exception as e:
            print(f"Error adding to Qdrant: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def query(
        self,
        conversation_id: str,
        query_text: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Adaptive retrieval with Qwen3 + SPLADE + HyDE
        
        Returns list of dicts with 'text', 'similarity', 'metadata'
        """
        try:
            if not query_text or not query_text.strip():
                return []
            
            conv_filter = Filter(
                must=[FieldCondition(
                    key="conversation_id",
                    match=MatchValue(value=conversation_id)
                )]
            )
            
            # Stage 1: Qwen3 dense (fast path)
            dense_query = self.dense_encoder.encode(query_text).tolist()
            
            with self._lock:
                qwen_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query,
                    using="dense",
                    query_filter=conv_filter,
                    limit=top_k * 2
                )
            
            if not qwen_results.points:
                return []
            
            top_similarity = qwen_results.points[0].score
            
            # HIGH confidence: Qwen3 only
            if top_similarity > 0.7:
                print(f"  [Adaptive] HIGH confidence ({top_similarity:.3f}) - Qwen3 only ⚡")
                return self._format_results(qwen_results.points[:top_k])
            
            # MEDIUM: Add SPLADE
            elif top_similarity > 0.5:
                print(f"  [Adaptive] MEDIUM confidence ({top_similarity:.3f}) - Adding SPLADE")
                
                if self.splade:
                    splade_dict = self.splade(query_text, return_sparse=True)
                    sparse_query = SparseVector(
                        indices=list(splade_dict.keys()),
                        values=list(splade_dict.values())
                    )
                else:
                    sparse_query = self._generate_sparse_vector_bm25(query_text)
                
                with self._lock:
                    splade_results = self.client.query_points(
                        collection_name=self.collection_name,
                        query=sparse_query,
                        using="sparse",
                        query_filter=conv_filter,
                        limit=top_k * 2
                    )
                
                fused = self._reciprocal_rank_fusion(
                    qwen_results.points,
                    splade_results.points
                )
                return self._format_results(fused[:top_k])
            
            # LOW: Add HyDE (if LLM available)
            else:
                print(f"  [Adaptive] LOW confidence ({top_similarity:.3f}) - Adding HyDE")
                
                if self.splade:
                    splade_dict = self.splade(query_text, return_sparse=True)
                    sparse_query = SparseVector(
                        indices=list(splade_dict.keys()),
                        values=list(splade_dict.values())
                    )
                else:
                    sparse_query = self._generate_sparse_vector_bm25(query_text)
                
                if self.llm_engine:
                    hyde_answer = await self._generate_hyde_answer(query_text)
                    hyde_dense = self.dense_encoder.encode(hyde_answer).tolist()
                    
                    with self._lock:
                        splade_results = self.client.query_points(
                            collection_name=self.collection_name,
                            query=sparse_query,
                            using="sparse",
                            query_filter=conv_filter,
                            limit=top_k * 2
                        )
                        
                        hyde_results = self.client.query_points(
                            collection_name=self.collection_name,
                            query=hyde_dense,
                            using="dense",
                            query_filter=conv_filter,
                            limit=top_k * 2
                        )
                    
                    fused = self._reciprocal_rank_fusion(
                        qwen_results.points,
                        splade_results.points,
                        hyde_results.points
                    )
                else:
                    with self._lock:
                        splade_results = self.client.query_points(
                            collection_name=self.collection_name,
                            query=sparse_query,
                            using="sparse",
                            query_filter=conv_filter,
                            limit=top_k * 2
                        )
                    fused = self._reciprocal_rank_fusion(
                        qwen_results.points,
                        splade_results.points
                    )
                
                return self._format_results(fused[:top_k])
        
        except Exception as e:
            print(f"Query error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _generate_hyde_answer(self, query: str) -> str:
        """Generate hypothetical answer using vLLM"""
        prompt = f"Answer this question briefly in 1-2 sentences: {query}\nAnswer:"
        
        outputs = self.llm_engine.generate(
            [prompt],
            max_tokens=50,
            temperature=0.3
        )
        
        answer = outputs[0].outputs[0].text.strip()
        print(f"    [HyDE] Generated: {answer[:60]}...")
        return answer
    
    def _reciprocal_rank_fusion(
        self,
        *result_lists,
        weights=None,
        k=60
    ) -> List:
        """RRF: Fuse multiple retrieval results"""
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        scores = {}
        
        for weight, results in zip(weights, result_lists):
            for rank, point in enumerate(results):
                point_id = point.id
                rrf_score = weight * (1.0 / (k + rank + 1))
                
                if point_id not in scores:
                    scores[point_id] = {
                        'score': rrf_score,
                        'point': point
                    }
                else:
                    scores[point_id]['score'] += rrf_score
        
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['point'] for item in sorted_results]
    
    def _format_results(self, points) -> List[Dict]:
        """Format Qdrant points to standard result format"""
        formatted = []
        for point in points:
            formatted.append({
                'text': point.payload.get('text', ''),
                'metadata': point.payload,
                'similarity': point.score
            })
        return formatted
    
    def _generate_sparse_vector_bm25(self, text: str):
        """BM25 fallback if SPLADE not available"""
        tokens = text.lower().split()
        token_counts = {}
        
        for token in tokens:
            token_hash = hash(token) % 100000
            token_counts[token_hash] = token_counts.get(token_hash, 0) + 1
        
        indices = list(token_counts.keys())
        values = list(token_counts.values())
        
        return SparseVector(indices=indices, values=values)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete all memories for a conversation"""
        try:
            conv_filter = Filter(
                must=[FieldCondition(
                    key="conversation_id",
                    match=MatchValue(value=conversation_id)
                )]
            )
            
            with self._lock:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=conv_filter
                )
            
            return True
        except Exception as e:
            print(f"Delete error: {e}")
            return False


def create_vector_db(backend: str = "qdrant", **kwargs):
    """Factory function"""
    if backend == "qdrant":
        return QdrantAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
