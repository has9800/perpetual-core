"""
Vector database adapters for conversation memory
SIMPLE V2: Nomic-Embed + Smart Qdrant RRF Reranking (when similarity < threshold)
"""

from typing import List, Dict, Optional
import os
import threading


class QdrantAdapter:
    """Qdrant adapter with Nomic-Embed + smart RRF reranking"""

    def __init__(self, 
                 persist_dir: str = "./data/qdrant_db",
                 collection_name: str = "conversations"):
        """Initialize Qdrant with Nomic-Embed"""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        os.makedirs(persist_dir, exist_ok=True)

        self.client = QdrantClient(path=persist_dir)
        self.collection_name = collection_name
        self._lock = threading.RLock()

        # Nomic-Embed for embeddings
        from sentence_transformers import SentenceTransformer
        print("Loading Nomic-Embed-Text-v1.5 (768 dim)...")
        self.encoder = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        self.vector_size = 768
        print("✅ Nomic-Embed loaded")

        # Create collection if needed
        try:
            self.client.get_collection(collection_name)
            print(f"Qdrant collection '{collection_name}' exists")
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            print(f"Qdrant collection '{collection_name}' created")

        info = self.client.get_collection(collection_name)
        print(f"Qdrant ready: {info.points_count} documents")

    def add(self, 
            conversation_id: str,
            text: str,
            metadata: Optional[Dict] = None) -> bool:
        """Add conversation turn"""
        try:
            if not text or not text.strip():
                return False

            from qdrant_client.models import PointStruct

            vector = self.encoder.encode(text).tolist()
            turn_number = metadata.get('turn_number', 0) if metadata else 0
            point_id = hash(f"{conversation_id}_{turn_number}_{text[:50]}") & 0x7FFFFFFFFFFFFFFF

            payload = metadata or {}
            payload['conversation_id'] = conversation_id
            payload['text'] = text
            clean_payload = {k: v for k, v in payload.items() if v is not None}

            with self._lock:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(id=point_id, vector=vector, payload=clean_payload)]
                )

            return True

        except Exception as e:
            print(f"Qdrant add error: {e}")
            return False

    def query(self,
             conversation_id: str,
             query_text: str,
             top_k: int = 3,
             similarity_threshold: float = 0.6) -> List[Dict]:
        """
        Smart retrieval: Auto-rerank with Qdrant RRF if top similarity < threshold
        
        Args:
            similarity_threshold: If top result < this, trigger RRF reranking (default 0.6)
        """
        try:
            if not query_text or not query_text.strip():
                return []

            from qdrant_client.models import Filter, FieldCondition, MatchValue

            query_vector = self.encoder.encode(query_text).tolist()

            # First pass: Standard vector search
            with self._lock:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=Filter(
                        must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
                    )
                )

            # Format results
            formatted = []
            for hit in results:
                if hit.score > 0.3:
                    formatted.append({
                        'text': hit.payload.get('text', ''),
                        'metadata': hit.payload,
                        'similarity': hit.score
                    })

            if not formatted:
                print(f"  [Qdrant] No results found")
                return []

            top_similarity = formatted[0]['similarity']

            # Smart reranking: Only if top similarity is low
            if top_similarity < similarity_threshold:
                print(f"  [Qdrant] Low similarity ({top_similarity:.3f}) → Reranking with RRF")
                
                try:
                    from qdrant_client.models import Prefetch, Query
                    
                    # Retrieve more candidates for reranking
                    with self._lock:
                        reranked_results = self.client.query_points(
                            collection_name=self.collection_name,
                            prefetch=Prefetch(
                                query=query_vector,
                                limit=20,  # Get 20 candidates
                                filter=Filter(
                                    must=[FieldCondition(
                                        key="conversation_id", 
                                        match=MatchValue(value=conversation_id)
                                    )]
                                )
                            ),
                            query=Query(fusion="rrf"),  # Reciprocal Rank Fusion
                            limit=top_k
                        )
                    
                    # Reformat with rerank flag
                    formatted = []
                    for hit in reranked_results:
                        if hit.score > 0.3:
                            formatted.append({
                                'text': hit.payload.get('text', ''),
                                'metadata': hit.payload,
                                'similarity': hit.score,
                                'reranked': True
                            })
                    
                    if formatted:
                        print(f"  [Qdrant] Reranked → top similarity now {formatted[0]['similarity']:.3f}")
                
                except Exception as e:
                    print(f"  [Qdrant] RRF reranking failed: {e}, using original results")
            else:
                print(f"  [Qdrant] High similarity ({top_similarity:.3f}) → No reranking needed")

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
    """Factory function"""
    if backend == "qdrant":
        return QdrantAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
