"""
Vector database adapters for conversation memory
HYBRID SEARCH: Nomic-Embed (dense) + BM25 (sparse) + Smart RRF Reranking
"""

from typing import List, Dict, Optional
import os
import threading


class QdrantAdapter:
    """Qdrant adapter with Hybrid Search (Dense + Sparse)"""

    def __init__(self, 
                persist_dir: str = "./data/qdrant_db",
                collection_name: str = "conversations",
                url: Optional[str] = None,
                api_key: Optional[str] = None,
                llm_engine = None):  # ✅ NEW: vLLM engine for HyDE
        """Initialize Qdrant with Adaptive Retrieval (Qwen3 + SPLADE + HyDE)"""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PayloadSchemaType
        from sentence_transformers import SentenceTransformer
        
        # Connect to Qdrant (Cloud or local)
        if url and api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
            print(f"Connected to Qdrant Cloud: {url}")
        else:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = QdrantClient(path=persist_dir)
            print(f"Connected to local Qdrant: {persist_dir}")

        self.collection_name = collection_name
        self._lock = threading.RLock()
        self.llm_engine = llm_engine  # ✅ Store for HyDE

        # 1. Qwen3 for dense embeddings (lightweight, fast)
        print("Loading Qwen3-Embedding-0.6B...")
        self.dense_encoder = SentenceTransformer(
            'Qwen/Qwen3-Embedding-0.6B',
            device='cuda'
        )
        self.dense_size = 768
        print("✅ Qwen3 loaded")

        # 2. SPLADE for semantic sparse retrieval
        print("Loading SPLADE for semantic term expansion...")
        try:
            from splade.models.transformer_rep import Splade
            self.splade = Splade(
                'naver/splade-cocondenser-ensembledistil',
                device='cuda'
            )
            print("✅ SPLADE loaded")
        except Exception as e:
            print(f"⚠️  SPLADE failed to load: {e}")
            print("    Falling back to BM25")
            self.splade = None

        # Create collection with hybrid vectors
        try:
            self.client.get_collection(collection_name)
            print(f"Qdrant collection '{collection_name}' exists")
        except:
            self.client.create_collection(
                collection_name=collection_name,
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
            print(f"Qdrant collection '{collection_name}' created (Qwen3-768 + SPLADE)")

        # Create conversation_id index
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="conversation_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(f"✅ Created index on 'conversation_id'")
        except Exception as idx_err:
            if "already exists" in str(idx_err).lower():
                print(f"✅ Index on 'conversation_id' already exists")
            else:
                print(f"Index warning: {idx_err}")

        info = self.client.get_collection(collection_name)
        print(f"✅ Adaptive retrieval ready: {info.points_count} documents (Qwen3 + SPLADE + HyDE)")



    def _generate_sparse_vector(self, text: str) -> Dict:
        """Generate BM25 sparse vector"""
        # Simple tokenization for BM25
        tokens = text.lower().split()
        
        # Count token frequencies
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Convert to sparse vector format (indices + values)
        indices = []
        values = []
        for token, count in token_counts.items():
            # Simple hash for token to index mapping
            idx = hash(token) % 100000
            indices.append(idx)
            values.append(float(count))
        
        return {"indices": indices, "values": values}

    def add(self, conversation_id: str, text: str, metadata: Optional[Dict] = None):
        """Add text to vector database with Qwen3 + SPLADE"""
        try:
            from qdrant_client.models import PointStruct, SparseVector
            import uuid
            
            # Dense embedding (Qwen3)
            dense_vector = self.dense_encoder.encode(text).tolist()
            
            # Sparse vector (SPLADE or BM25 fallback)
            if self.splade:
                splade_dict = self.splade(text, return_sparse=True)
                sparse_vector = SparseVector(
                    indices=list(splade_dict.keys()),
                    values=list(splade_dict.values())
                )
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


    async def query(self,
                conversation_id: str,
                query_text: str,
                top_k: int = 3) -> List[Dict]:
        """
        Adaptive retrieval strategy:
        - >0.7 similarity: Qwen3 only (fast path, 10ms)
        - 0.5-0.7: Qwen3 + SPLADE (medium, 25ms)
        - 0.3-0.5: All three + RRF (full search, 100ms)
        - <0.3: Full + HyDE emphasis (hard queries, 100ms)
        """
        try:
            if not query_text or not query_text.strip():
                return []

            from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector

            conv_filter = Filter(
                must=[FieldCondition(
                    key="conversation_id",
                    match=MatchValue(value=conversation_id)
                )]
            )

            # Stage 1: Try Qwen3 dense embedding (fast path)
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
                print(f"  [Adaptive] No results found")
                return []

            # Check confidence
            top_similarity = qwen_results.points[0].score

            # HIGH CONFIDENCE: Qwen3 is confident
            if top_similarity > 0.7:
                print(f"  [Adaptive] HIGH confidence ({top_similarity:.3f}) - Qwen3 only ⚡")
                return self._format_results(qwen_results.points[:top_k])

            # MEDIUM CONFIDENCE: Add SPLADE
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

            # LOW CONFIDENCE: Add HyDE
            elif top_similarity > 0.3:
                print(f"  [Adaptive] LOW confidence ({top_similarity:.3f}) - Adding SPLADE + HyDE")

                # SPLADE retrieval
                if self.splade:
                    splade_dict = self.splade(query_text, return_sparse=True)
                    sparse_query = SparseVector(
                        indices=list(splade_dict.keys()),
                        values=list(splade_dict.values())
                    )
                else:
                    sparse_query = self._generate_sparse_vector_bm25(query_text)

                # HyDE retrieval (if LLM available)
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
                    # No LLM, just Qwen3 + SPLADE
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

            # VERY LOW CONFIDENCE: Full search with HyDE emphasis
            else:
                print(f"  [Adaptive] VERY LOW confidence ({top_similarity:.3f}) - Full search + HyDE emphasis")

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
                            limit=top_k * 3
                        )

                        hyde_results = self.client.query_points(
                            collection_name=self.collection_name,
                            query=hyde_dense,
                            using="dense",
                            query_filter=conv_filter,
                            limit=top_k * 3
                        )

                    # Emphasize HyDE for very hard queries
                    fused = self._reciprocal_rank_fusion(
                        qwen_results.points,
                        splade_results.points,
                        hyde_results.points,
                        weights=[0.2, 0.3, 0.5]
                    )
                else:
                    with self._lock:
                        splade_results = self.client.query_points(
                            collection_name=self.collection_name,
                            query=sparse_query,
                            using="sparse",
                            query_filter=conv_filter,
                            limit=top_k * 3
                        )
                    fused = self._reciprocal_rank_fusion(
                        qwen_results.points,
                        splade_results.points
                    )

                return self._format_results(fused[:top_k])

        except Exception as e:
            print(f"Qdrant query error: {e}")
            import traceback
            traceback.print_exc()
            return []
    async def _generate_hyde_answer(self, query: str) -> str:
        """Generate hypothetical answer using vLLM"""
        prompt = f"Answer this question briefly in 1-2 sentences: {query}\nAnswer:"
        
        # Wrap in list for vLLM batch processing
        outputs = self.llm_engine.generate([prompt], sampling_params={
            'max_tokens': 50,
            'temperature': 0.3,
            'top_p': 0.9
        })
        
        answer = outputs[0].outputs[0].text.strip()
        print(f"    [HyDE] Generated: {answer[:60]}...")
        return answer

    def _reciprocal_rank_fusion(self, 
                                *result_lists,
                                weights=None,
                                k=60) -> List:
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
        
        # Sort by RRF score
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
        from qdrant_client.models import SparseVector
        tokens = text.lower().split()
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        indices = [hash(token) % 100000 for token in token_counts.keys()]
        values = list(token_counts.values())
        
        return SparseVector(indices=indices, values=values)


    def get_by_conversation(self, conversation_id: str) -> List[Dict]:
        """Get all turns for a conversation"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            with self._lock:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(
                            key="conversation_id", 
                            match=MatchValue(value=conversation_id)
                        )]
                    ),
                    limit=1000
                )

            formatted = []
            for point in results[0]:
                formatted.append({
                    'text': point.payload.get('text', ''),
                    'metadata': point.payload
                })

            return formatted

        except Exception as e:
            print(f"Qdrant get error: {e}")
            return []

    def delete_conversation(self, conversation_id: str):
        """Delete all turns for a conversation"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            with self._lock:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[FieldCondition(
                            key="conversation_id", 
                            match=MatchValue(value=conversation_id)
                        )]
                    )
                )
        except Exception as e:
            print(f"Qdrant delete error: {e}")


def create_vector_db(backend: str = "qdrant", **kwargs):
    """Factory function"""
    if backend == "qdrant":
        return QdrantAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

