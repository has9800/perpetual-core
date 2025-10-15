"""
Simple Context Manager - Short-Term + Long-Term Memory
KISS Principle: Just sliding window + Qdrant, no compression
"""

from typing import List, Dict, Tuple
from collections import deque
import time


class SimpleContextManager:
    """
    Simple two-tier memory:
    - Short-term: Last 15 turns in memory (sliding window)
    - Long-term: Everything in Qdrant for retrieval
    """
    
    def __init__(self, short_term_limit: int = 15):
        """
        Args:
            short_term_limit: Number of recent turns to keep in memory (default 15)
        """
        self.short_term_limit = short_term_limit
        
        # In-memory short-term storage: {conv_id: deque of turns}
        # deque with maxlen automatically evicts oldest when full
        self.short_term = {}
        
        # Statistics
        self.stats = {
            'contexts_built': 0,
            'total_generations': 0,
        }
    
    def add_turn(self, conversation_id: str, user_query: str, 
                 assistant_response: str, metadata: Dict = None):
        """
        Add turn to short-term memory
        Automatically evicts oldest turn if > short_term_limit
        """
        if conversation_id not in self.short_term:
            self.short_term[conversation_id] = deque(maxlen=self.short_term_limit)
        
        turn = {
            'user': user_query,
            'assistant': assistant_response,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        # deque automatically evicts oldest when appending beyond maxlen
        self.short_term[conversation_id].append(turn)
        self.stats['total_generations'] += 1
    
    def get_short_term_turns(self, conversation_id: str) -> List[Dict]:
        """Get all short-term turns (last N turns, up to limit)"""
        if conversation_id not in self.short_term:
            return []
        return list(self.short_term[conversation_id])
    
    def build_context(self, 
                     conversation_id: str,
                     retrieved_long_term: List[Dict] = None) -> Tuple[str, Dict]:
        """
        Build context from short-term + retrieved long-term
        
        Format:
        # Earlier context: (retrieved from Qdrant if relevant)
        User: ...
        Assistant: ...
        
        # Recent conversation: (last 15 turns from memory)
        User: ...
        Assistant: ...
        
        Returns:
            (formatted_context, metadata_dict)
        """
        self.stats['contexts_built'] += 1
        
        short_term = self.get_short_term_turns(conversation_id)
        
        context_parts = []
        
        # Add retrieved long-term context (only if high similarity)
        retrieved_count = 0
        if retrieved_long_term:
            high_quality = [ctx for ctx in retrieved_long_term if ctx.get('similarity', 0) > 0.5]
            if high_quality:
                context_parts.append("# Earlier context:")
                for ctx in high_quality[:3]:  # Top 3 most relevant
                    query = ctx.get('text', '')
                    response = ctx.get('metadata', {}).get('response', '')
                    if query and response:
                        context_parts.append(f"User: {query}")
                        context_parts.append(f"Assistant: {response}")
                        retrieved_count += 1
                context_parts.append("")
        
        # Add ALL short-term turns (always included)
        if short_term:
            context_parts.append("# Recent conversation:")
            for turn in short_term:
                context_parts.append(f"User: {turn['user']}")
                context_parts.append(f"Assistant: {turn['assistant']}")
            context_parts.append("")
        
        formatted = "\n".join(context_parts)
        
        metadata = {
            'short_term_turns': len(short_term),
            'long_term_retrieved': retrieved_count,
            'total_context_lines': len(context_parts)
        }
        
        return formatted, metadata
    
    def get_turn_count(self, conversation_id: str) -> int:
        """Get number of turns in short-term memory"""
        return len(self.short_term.get(conversation_id, []))
    
    def delete_conversation(self, conversation_id: str):
        """Delete conversation from short-term memory"""
        if conversation_id in self.short_term:
            del self.short_term[conversation_id]
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            **self.stats,
            'active_conversations': len(self.short_term),
            'total_short_term_turns': sum(len(turns) for turns in self.short_term.values()),
            'avg_turns_per_conversation': (
                sum(len(turns) for turns in self.short_term.values()) / len(self.short_term)
                if self.short_term else 0
            )
        }
