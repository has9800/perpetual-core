"""
Reranker Module for V2
Two-stage retrieval: Nomic-Embed (stage 1) → BGE Reranker (stage 2)
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder


class BGEReranker:
    """BGE Reranker v2-m3 for precise reranking"""
    
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        """Initialize BGE reranker"""
        print(f"Loading BGE reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        print(f"✅ BGE reranker loaded")
    
    def rerank(self, 
               query: str, 
               candidates: List[Dict], 
               top_k: int = 3) -> List[Dict]:
        """
        Rerank candidates using cross-encoder
        
        Args:
            query: User query text
            candidates: List of dicts with 'text' and 'metadata' keys
            top_k: Number of top results to return
        
        Returns:
            Reranked list of top_k candidates
        """
        if not candidates:
            return []
        
        if len(candidates) <= top_k:
            return candidates
        
        # Create query-document pairs for cross-encoder
        pairs = [(query, candidate['text']) for candidate in candidates]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Attach scores to candidates
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(scores[i])
        
        # Sort by rerank score (descending)
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]


# Singleton instance (lazy load)
_reranker = None

def get_reranker() -> BGEReranker:
    """Get or create reranker instance"""
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker

def rerank(query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
    """Convenience function for reranking"""
    return get_reranker().rerank(query, candidates, top_k)
