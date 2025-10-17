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
        if url:
            # Remote Qdrant (cloud or self-hosted)
            if api_key:
                self.client = QdrantClient(url=url, api_key=api_key)
                print(f"Connected to Qdrant Cloud: {url}")
            else:
                self.client = QdrantClient(url=url)
                print(f"Connected to Qdrant (no auth): {url}")
        else:
            # Local Qdrant
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
        self.model = self.dense_encoder  # Alias for compatibility
        # Get actual embedding dimension from the model
        self.dense_size = self.dense_encoder.get_sentence_embedding_dimension()
        print(f"✅ Qwen3 loaded (dim: {self.dense_size})")
        
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
    
    async def query_with_context_window(
        self,
        conversation_id: str,
        query_text: str,
        top_k: int = 3,
        context_window: int = 2,
        min_similarity: float = 0.65
    ) -> List[Dict]:
        """
        ENHANCED retrieval with context window and re-ranking

        Args:
            conversation_id: Conversation to search
            query_text: Query to search for
            top_k: Number of results to return
            context_window: Number of turns before/after each match (±window)
            min_similarity: Minimum similarity threshold for filtering

        Returns:
            List of dicts with 'text', 'similarity', 'metadata', 'context_window'
        """
        try:
            # Get base semantic matches
            base_results = await self.query(conversation_id, query_text, top_k=top_k * 2)

            if not base_results:
                return []

            # Re-rank with relevance scoring
            reranked = self._rerank_by_relevance(base_results, query_text, min_similarity)

            # Enrich with context window
            enriched_results = []
            for result in reranked[:top_k]:
                turn_number = result['metadata'].get('turn_number')

                if turn_number:
                    # Get surrounding context
                    context_turns = await self._get_turns_around(
                        conversation_id,
                        turn_number,
                        window=context_window
                    )

                    result['context_window'] = context_turns
                    result['has_context'] = len(context_turns) > 0

                enriched_results.append(result)

            return enriched_results

        except Exception as e:
            print(f"Enhanced query error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic query
            return await self.query(conversation_id, query_text, top_k)

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
    
    def _rerank_by_relevance(
        self,
        results: List[Dict],
        query: str,
        min_similarity: float = 0.65
    ) -> List[Dict]:
        """
        Re-rank results by combining semantic similarity and lexical overlap

        Args:
            results: Initial results from retrieval
            query: Original query text
            min_similarity: Minimum similarity threshold

        Returns:
            Re-ranked and filtered results
        """
        query_terms = set(query.lower().split())
        filtered = []

        for result in results:
            # Filter low-quality matches
            if result['similarity'] < min_similarity:
                continue

            # Calculate lexical overlap
            result_text = result['text'].lower()
            result_terms = set(result_text.split())
            overlap = len(query_terms & result_terms) / len(query_terms) if query_terms else 0

            # Combined scoring (70% semantic, 30% lexical)
            final_score = (0.7 * result['similarity']) + (0.3 * overlap)

            result['final_score'] = final_score
            result['lexical_overlap'] = overlap
            filtered.append(result)

        # Sort by final score
        filtered.sort(key=lambda x: x['final_score'], reverse=True)

        return filtered

    async def _get_turns_around(
        self,
        conversation_id: str,
        turn_number: int,
        window: int = 2
    ) -> List[Dict]:
        """
        Get turns before and after a specific turn number

        Args:
            conversation_id: Conversation ID
            turn_number: Central turn number
            window: Number of turns before/after

        Returns:
            List of turns with their text and metadata
        """
        try:
            start_turn = max(1, turn_number - window)
            end_turn = turn_number + window

            # Query for turns in range
            turns = []
            for i in range(start_turn, end_turn + 1):
                if i == turn_number:
                    continue  # Skip the main match (already included)

                # Scroll through collection to find turn
                turn_filter = Filter(
                    must=[
                        FieldCondition(
                            key="conversation_id",
                            match=MatchValue(value=conversation_id)
                        ),
                        FieldCondition(
                            key="turn_number",
                            match=MatchValue(value=i)
                        )
                    ]
                )

                with self._lock:
                    result = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=turn_filter,
                        limit=1
                    )

                if result[0]:  # result is tuple (points, next_offset)
                    point = result[0][0]
                    turns.append({
                        'turn_number': i,
                        'text': point.payload.get('text', ''),
                        'metadata': point.payload
                    })

            # Sort by turn number
            turns.sort(key=lambda x: x['turn_number'])
            return turns

        except Exception as e:
            print(f"Error getting context window: {e}")
            return []

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
