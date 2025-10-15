"""
Vector database adapters for conversation memory
Supports: ChromaDB (development) and Qdrant (production)
FIXED: Thread-safe Qdrant operations
"""

from typing import List, Dict, Optional
import os
import threading
from reranker import rerank

class QdrantAdapter:
    """Qdrant adapter - Fast, production-ready vector DB with Nomic-Embed + Reranking"""

    def __init__(self, 
                 persist_dir: str = "./data/qdrant_db",
                 collection_name: str = "conversations"):
        """Initialize Qdrant client with Nomic-Embed and reranking support"""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        os.makedirs(persist_dir, exist_ok=True)

        # Create local persistent client
        self.client = QdrantClient(path=persist_dir)
        self.collection_name = collection_name

        # Thread lock for concurrent access protection
        self._lock = threading.RLock()

        # V2: Initialize Nomic-Embed for embeddings (UPGRADED from all-MiniLM-L6-v2)
        from sentence_transformers import SentenceTransformer
        print("Loading Nomic-Embed-Text-v1.5 (768 dim)...")
        self.encoder = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')
        self.vector_size = 768  # Nomic-Embed dimension (upgraded from 384)
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
        """Lazy load BGE reranker (only when first query needs it)"""
        if self._reranker is None:
            from reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker

    def add(self, 
            conversation_id: str,
            text: str,
            metadata: Optional[Dict] = None) -> bool:
        """Add conversation turn with Nomic-Embed embedding (thread-safe)"""
        try:
            if not text or not text.strip():
                return False

            from qdrant_client.models import PointStruct

            # Generate embedding with Nomic-Embed
            vector = self.encoder.encode(text).tolist()

            # Create unique ID
            turn_number = metadata.get('turn_number', 0) if metadata else 0
            point_id = hash(f"{conversation_id}_{turn_number}") & 0x7FFFFFFFFFFFFFFF  # Positive int64

            # Build payload (metadata)
            payload = metadata or {}
            payload['conversation_id'] = conversation_id
            payload['text'] = text

            # Filter out None values
            clean_payload = {
                k: v for k, v in payload.items() 
                if v is not None
            }

            # Thread-safe upsert
            with self._lock:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=clean_payload
                        )
                    ]
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
        V2: Two-stage retrieval with Nomic-Embed + BGE Reranker (thread-safe)
        
        Stage 1: Retrieve 20 candidates with Nomic-Embed
        Stage 2: Rerank to top-3 with BGE Reranker
        """
        try:
            if not query_text or not query_text.strip():
                return []

            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Generate query embedding with Nomic-Embed
            query_vector = self.encoder.encode(query_text).tolist()

            # STAGE 1: Retrieve more candidates (20) for reranking
            retrieve_k = 20 if use_reranking else top_k

            # Thread-safe search
            with self._lock:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=retrieve_k,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="conversation_id",
                                match=MatchValue(value=conversation_id)
                            )
                        ]
                    )
                )

            # Format results
            formatted = []
            for hit in results:
                # Only include if similarity > 0.3 (meaningful match)
                if hit.score > 0.3:
                    formatted.append({
                        'text': hit.payload.get('text', ''),
                        'metadata': hit.payload,
                        'similarity': hit.score
                    })

            # STAGE 2: Rerank top candidates with BGE
            if use_reranking and len(formatted) > top_k:
                reranker = self._get_reranker()
                formatted = reranker.rerank(query_text, formatted, top_k=top_k)
            elif len(formatted) > top_k:
                formatted = formatted[:top_k]

            return formatted

        except Exception as e:
            print(f"Qdrant query error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_by_conversation(self, conversation_id: str) -> List[Dict]:
        """Get all turns for a conversation (thread-safe)"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Thread-safe scroll
            with self._lock:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="conversation_id",
                                match=MatchValue(value=conversation_id)
                            )
                        ]
                    ),
                    limit=1000
                )

            formatted = []
            for point in results[0]:  # results is (points, next_page_offset)
                formatted.append({
                    'text': point.payload.get('text', ''),
                    'metadata': point.payload
                })

            return formatted

        except Exception as e:
            print(f"Qdrant get error: {e}")
            return []

    def delete_conversation(self, conversation_id: str):
        """Delete all turns for a conversation (thread-safe)"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            with self._lock:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="conversation_id",
                                match=MatchValue(value=conversation_id)
                            )
                        ]
                    )
                )
        except Exception as e:
            print(f"Qdrant delete error: {e}")


class ChromaDBAdapter:
    """ChromaDB adapter - Simple, good for development"""

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        """Initialize ChromaDB with sentence transformers"""
        import chromadb
        from chromadb.config import Settings

        os.makedirs(persist_dir, exist_ok=True)

        # Create client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Use sentence transformers embedding function
        from chromadb.utils import embedding_functions
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"ChromaDB initialized: {self.collection.count()} documents")

    def add(self, 
            conversation_id: str,
            text: str,
            metadata: Optional[Dict] = None) -> bool:
        """Add conversation turn with proper embedding"""
        try:
            if not text or not text.strip():
                return False

            # Create unique ID
            doc_id = f"{conversation_id}_{metadata.get('turn_number', 0)}"

            # Add metadata
            meta = metadata or {}
            meta['conversation_id'] = conversation_id

            # FILTER OUT None VALUES (ChromaDB doesn't accept them)
            clean_meta = {
                k: v for k, v in meta.items() 
                if v is not None
            }

            # Add to collection (embeddings generated automatically)
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[clean_meta]
            )

            return True

        except Exception as e:
            print(f"ChromaDB add error: {e}")
            return False

    def query(self,
             conversation_id: str,
             query_text: str,
             top_k: int = 3) -> List[Dict]:
        """Query with similarity threshold"""
        try:
            if not query_text or not query_text.strip():
                return []

            # Query collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where={"conversation_id": conversation_id}
            )

            # Format results with similarity scores
            formatted = []
            if results and results['documents'] and len(results['documents'][0]) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    similarity = 1.0 - distance  # Convert distance to similarity

                    # Only include if similarity > 0.3 (meaningful match)
                    if similarity > 0.3:
                        formatted.append({
                            'text': doc,
                            'metadata': results['metadatas'][0][i],
                            'similarity': similarity
                        })

            return formatted

        except Exception as e:
            print(f"ChromaDB query error: {e}")
            return []

    def get_by_conversation(self, conversation_id: str) -> List[Dict]:
        """Get all turns for a conversation"""
        try:
            results = self.collection.get(
                where={"conversation_id": conversation_id}
            )

            if not results or not results['documents']:
                return []

            formatted = []
            for i, doc in enumerate(results['documents']):
                formatted.append({
                    'text': doc,
                    'metadata': results['metadatas'][i]
                })

            return formatted

        except Exception as e:
            print(f"ChromaDB get error: {e}")
            return []

    def delete_conversation(self, conversation_id: str):
        """Delete all turns for a conversation"""
        try:
            results = self.collection.get(
                where={"conversation_id": conversation_id}
            )
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
        except Exception as e:
            print(f"ChromaDB delete error: {e}")


def create_vector_db(backend: str = "qdrant", **kwargs):
    """
    Factory function to create vector DB adapter

    Args:
        backend: "qdrant" (production) or "chromadb" (development)
    """
    if backend == "qdrant":
        return QdrantAdapter(**kwargs)
    elif backend == "chromadb":
        return ChromaDBAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown vector DB backend: {backend}. Use 'qdrant' or 'chromadb'")