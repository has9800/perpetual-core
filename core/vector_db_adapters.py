"""
Vector database adapters for conversation memory
Supports: Qdrant (production) with V2 features
UPDATED: Nomic-Embed + BGE Reranker with debug logging
"""

from typing import List, Dict, Optional
import os
import threading


class QdrantAdapter:
    """Qdrant adapter - Fast, production-ready vector DB with Nomic-Embed + Reranking"""

    def __init__(self, 
                 persist_dir: str = "./data/qdrant_db",
                 collection_name: str = "conversations"):
        """Initialize Qdrant client with Nomic-Embed and reranking support"""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        os.makedirs(persist_dir, exist_ok=True)

        # Create local persistent client
        self.client = QdrantClient(path=persist_dir)
        self.collection_name = collection_name

        # Thread lock for concurrent access protection
        self._lock = threading.RLock()

        # V2: Initialize Nomic-Embed for embeddings
        from sentence_transformers import SentenceTransformer
        print("Loading Nomic-Embed-Text-v1.5 (768 dim)...")
        self.encoder = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        self.vector_size = 768
        print("✅ Nomic-Embed loaded")

        # V2: Initialize BGE reranker (lazy load on first query)
        self._reranker = None

        # Create collection if doesn't exist
        try:
            self.client.get_collection(collection_name)
            print(f"Qdrant collection '{collection_name}' already exists")
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Qdrant collection '{collection_name}' created with Nomic-Embed (768 dim)")

        # Get collection info
        info = self.client.get_collection(collection_name)
        print(f"Qdrant initialized: {info.points_count} documents")

    def _get_reranker(self):
        """Lazy load BGE reranker"""
        if self._reranker is None:
            from reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker

    def add(self, 
            conversation_id: str,
            text: str,
            metadata: Optional[Dict] = None) -> bool:
        """Add conversation turn with Nomic-Embed embedding"""
        try:
            if not text or not text.strip():
                return False

            from qdrant_client.models import PointStruct

            # Generate embedding with Nomic-Embed
            vector = self.encoder.encode(text).tolist()

            # Create unique ID
            turn_number = metadata.get('turn_number', 0) if metadata else 0
            point_id = hash(f"{conversation_id}_{turn_number}") & 0x7FFFFFFFFFFFFFFF

            # Build payload
            payload = metadata or {}
            payload['conversation_id'] = conversation_id
            payload['text'] = text

            # Filter out None values
            clean_payload = {k: v for k, v in payload.items() if v is not None}

            # Thread-safe upsert
            with self._lock:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(id=point_id, vector=vector, payload=clean_payload)]
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
             use_reranking: bool = True) -> List[Dict]:
        """
        V2: Two-stage retrieval with Nomic-Embed + BGE Reranker
        Stage 1: Retrieve 20 candidates
        Stage 2: Rerank to top-K
        """
        try:
            if not query_text or not query_text.strip():
                return []

            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Generate query embedding with Nomic-Embed
            query_vector = self.encoder.encode(query_text).tolist()

            # STAGE 1: Retrieve more candidates for reranking
            retrieve_k = 20 if use_reranking else top_k

            # Thread-safe search
            with self._lock:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=retrieve_k,
                    query_filter=Filter(
                        must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
                    )
                )

            # Format results
            formatted = []
            for hit in results:
                if hit.score > 0.3:  # Similarity threshold
                    formatted.append({
                        'text': hit.payload.get('text', ''),
                        'metadata': hit.payload,
                        'similarity': hit.score
                    })

            print(f"  [Qdrant] Stage 1: Retrieved {len(formatted)} candidates (threshold >0.3)")

            # STAGE 2: Rerank with BGE
            if use_reranking and len(formatted) > top_k:
                print(f"  [Qdrant] Stage 2: Reranking {len(formatted)} → {top_k}")
                reranker = self._get_reranker()
                formatted = reranker.rerank(query_text, formatted, top_k=top_k)
                
                # Debug: Show rerank scores
                rerank_scores = [f.get('rerank_score', 0) for f in formatted[:3]]
                print(f"  [Qdrant] Top-3 rerank scores: {[f'{s:.3f}' for s in rerank_scores]}")
            elif len(formatted) > top_k:
                formatted = formatted[:top_k]
                print(f"  [Qdrant] No reranking (not enough candidates or disabled)")

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
                        must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
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
                        must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
                    )
                )
        except Exception as e:
            print(f"Qdrant delete error: {e}")


def create_vector_db(backend: str = "qdrant", **kwargs):
    """Factory function to create vector DB adapter"""
    if backend == "qdrant":
        return QdrantAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown vector DB backend: {backend}")
