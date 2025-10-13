"""
Vector database adapters for conversation memory
FIXED: Proper embeddings, persistence, and retrieval
"""

from typing import List, Dict, Optional
import os


class ChromaDBAdapter:
    """ChromaDB adapter with proper embedding support"""

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

            # Add to collection (embeddings generated automatically)
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[meta]
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


def create_vector_db(backend: str = "chromadb", **kwargs):
    """Factory function to create vector DB adapter"""
    if backend == "chromadb":
        return ChromaDBAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown vector DB backend: {backend}")