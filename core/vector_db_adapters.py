"""
Vector Database Adapters
Supports multiple vector DB backends with unified interface
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import time


class VectorDBAdapter(ABC):
    """Base adapter for vector databases"""

    @abstractmethod
    def add(self, doc_id: str, text: str, metadata: Dict) -> float:
        """Add document, return latency in ms"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int, filter: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Delete document"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """Get database statistics"""
        pass


class ChromaDBAdapter(VectorDBAdapter):
    """Adapter for ChromaDB"""

    def __init__(self, collection_name: str = "conversations", 
                 persist_directory: str = "./data/chroma_db"):
        """Initialize ChromaDB adapter"""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")

        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Conversation memory storage"}
        )

        self.add_count = 0
        self.search_count = 0

    def add(self, doc_id: str, text: str, metadata: Dict) -> float:
        """Add document to ChromaDB"""
        start = time.time()

        try:
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata]
            )
            self.add_count += 1

        except Exception as e:
            if "already exists" in str(e):
                self.collection.update(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[metadata]
                )
            else:
                raise

        return (time.time() - start) * 1000

    def search(self, query: str, top_k: int, filter: Optional[Dict] = None) -> List[Dict]:
        """Search ChromaDB"""
        try:
            where = filter if filter else None

            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where
            )

            self.search_count += 1

            formatted = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'similarity': 1.0 - results['distances'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                    })

            return formatted

        except Exception as e:
            print(f"ChromaDB search error: {e}")
            return []

    def delete(self, doc_id: str) -> bool:
        """Delete document"""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"ChromaDB delete error: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get stats"""
        return {
            'backend': 'ChromaDB',
            'collection': self.collection.name,
            'document_count': self.collection.count(),
            'total_adds': self.add_count,
            'total_searches': self.search_count
        }


def create_vector_db(backend: str = "chromadb", **kwargs) -> VectorDBAdapter:
    """
    Factory function to create vector DB adapter

    Args:
        backend: 'chromadb' or 'qdrant'
        **kwargs: Backend-specific arguments

    Returns:
        VectorDBAdapter instance
    """
    if backend.lower() == "chromadb":
        return ChromaDBAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'chromadb'")