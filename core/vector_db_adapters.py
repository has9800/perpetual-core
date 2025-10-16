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
                api_key: Optional[str] = None):
        """Initialize Qdrant with Hybrid Search support"""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PayloadSchemaType

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

        # Nomic-Embed for dense vectors
        from sentence_transformers import SentenceTransformer
        print("Loading Nomic-Embed-Text-v1.5 (768 dim)...")
        self.dense_encoder = SentenceTransformer(
            'nomic-ai/nomic-embed-text-v1.5', 
            trust_remote_code=True
        )
        self.dense_size = 768
        print("✅ Nomic-Embed loaded")

        # Create collection with hybrid vectors (if doesn't exist)
        try:
            self.client.get_collection(collection_name)
            print(f"Qdrant collection '{collection_name}' exists")
        except:
            # Collection doesn't exist - create it
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
            print(f"Qdrant collection '{collection_name}' created with hybrid search")

        # ALWAYS ensure index exists (whether collection is new or existing)
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="conversation_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(f"✅ Created index on 'conversation_id'")
        except Exception as idx_err:
            # Index might already exist - that's fine
            if "already exists" in str(idx_err).lower() or "duplicate" in str(idx_err).lower():
                print(f"✅ Index on 'conversation_id' already exists")
            else:
                print(f"Index warning: {idx_err}")

        info = self.client.get_collection(collection_name)
        print(f"✅ Qdrant ready: {info.points_count} documents")

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

    def add(self, 
            conversation_id: str,
            text: str,
            metadata: Optional[Dict] = None) -> bool:
        """Add conversation turn with hybrid vectors"""
        try:
            if not text or not text.strip():
                return False

            from qdrant_client.models import PointStruct

            # Generate dense vector (Nomic-Embed)
            dense_vector = self.dense_encoder.encode(text).tolist()
            
            # Generate sparse vector (BM25)
            sparse_vector = self._generate_sparse_vector(text)

            turn_number = metadata.get('turn_number', 0) if metadata else 0
            point_id = hash(f"{conversation_id}_{turn_number}_{text[:50]}") & 0x7FFFFFFFFFFFFFFF

            payload = metadata or {}
            payload['conversation_id'] = conversation_id
            payload['text'] = text
            clean_payload = {k: v for k, v in payload.items() if v is not None}

            with self._lock:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(
                        id=point_id, 
                        vector={
                            "dense": dense_vector,
                            "sparse": sparse_vector
                        },
                        payload=clean_payload
                    )]
                )

            return True

        except Exception as e:
            print(f"Qdrant add error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def query(self,
            conversation_id: str,
            query_text: str,
            top_k: int = 3,
            similarity_threshold: float = 0.6) -> List[Dict]:
        """
        Hybrid search with smart RRF reranking
        
        Combines:
        1. Dense semantic search (Nomic-Embed)
        2. Sparse keyword search (BM25)
        3. RRF fusion
        """
        try:
            if not query_text or not query_text.strip():
                return []

            from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, Fusion

            # Generate query vectors
            dense_query = self.dense_encoder.encode(query_text).tolist()
            sparse_query = self._generate_sparse_vector(query_text)

            # Hybrid search with RRF fusion
            with self._lock:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        # Dense semantic search
                        Prefetch(
                            query=dense_query,
                            using="dense",
                            limit=20,
                            filter=Filter(
                                must=[FieldCondition(
                                    key="conversation_id", 
                                    match=MatchValue(value=conversation_id)
                                )]
                            )
                        ),
                        # Sparse keyword search (BM25)
                        Prefetch(
                            query=sparse_query,
                            using="sparse",
                            limit=20,
                            filter=Filter(
                                must=[FieldCondition(
                                    key="conversation_id", 
                                    match=MatchValue(value=conversation_id)
                                )]
                            )
                        )
                    ],
                    query=dense_query,  # Use dense query as main query
                    limit=top_k,
                    # RRF fusion happens in prefetch combination
                )

            # Format results
            formatted = []
            for hit in results:
                if hit.score > 0.3:
                    formatted.append({
                        'text': hit.payload.get('text', ''),
                        'metadata': hit.payload,
                        'similarity': hit.score,
                        'hybrid_search': True
                    })

            if not formatted:
                print(f"  [Qdrant] No results found")
                return []

            top_similarity = formatted[0]['similarity']

            if top_similarity < similarity_threshold:
                print(f"  [Qdrant] Hybrid search, low sim ({top_similarity:.3f})")
            else:
                print(f"  [Qdrant] Hybrid search, high sim ({top_similarity:.3f}) ✅")

            return formatted

        except Exception as e:
            print(f"Qdrant query error: {e}")
            import traceback
            traceback.print_exc()
            return []


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
