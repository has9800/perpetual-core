"""
Context Manager V2 - Application-Level Memory Management
Handles: sliding window (10-15 turns) + compression + token budgets + deduplication
"""

from typing import List, Dict, Tuple
from collections import OrderedDict
import time
from token_counter import count_tokens, count_messages_tokens, truncate_to_budget


class ContextManagerV2:
    """
    Manages conversation context with:
    - Last 10-15 turns kept uncompressed (in-memory)
    - Older turns compressed/deduplicated
    - Token budget enforcement (default 2000 tokens)
    - Offload to vector DB when stale
    """
    
    def __init__(self, 
                 token_budget: int = 2000,
                 recent_turns_limit: int = 15,
                 similarity_threshold: float = 0.85):
        """
        Initialize context manager
        
        Args:
            token_budget: Max tokens to send to LLM
            recent_turns_limit: Number of recent turns to keep uncompressed
            similarity_threshold: Similarity threshold for deduplication
        """
        self.token_budget = token_budget
        self.recent_turns_limit = recent_turns_limit
        self.similarity_threshold = similarity_threshold
        
        # In-memory storage: conversation_id -> list of turns
        self.conversations = {}  # {conv_id: [turn1, turn2, ...]}
        
        # Statistics
        self.stats = {
            'contexts_built': 0,
            'turns_deduplicated': 0,
            'turns_offloaded': 0,
            'tokens_saved': 0,
        }
    
    def add_turn(self, conversation_id: str, user_query: str, 
                 assistant_response: str, metadata: Dict = None):
        """
        Add a new turn to conversation
        
        Args:
            conversation_id: Unique conversation ID
            user_query: User's query
            assistant_response: Assistant's response
            metadata: Additional metadata
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        turn = {
            'user': user_query,
            'assistant': assistant_response,
            'timestamp': time.time(),
            'tokens': count_tokens(user_query) + count_tokens(assistant_response),
            'metadata': metadata or {}
        }
        
        self.conversations[conversation_id].append(turn)
    
    def get_context_for_llm(self, 
                           conversation_id: str,
                           retrieved_context: List[Dict] = None) -> Tuple[str, Dict]:
        """
        Build optimized context for LLM with token budget enforcement
        
        Args:
            conversation_id: Conversation ID
            retrieved_context: Optional retrieved older context from vector DB
        
        Returns:
            (formatted_context, metadata_dict)
        """
        self.stats['contexts_built'] += 1
        
        if conversation_id not in self.conversations:
            return "", {'recent_turns': 0, 'retrieved_turns': 0, 'total_tokens': 0}
        
        all_turns = self.conversations[conversation_id]
        
        # Step 1: Get last N turns (uncompressed)
        recent_turns = all_turns[-self.recent_turns_limit:]
        
        # Step 2: Get older turns for compression (turns before recent)
        older_turns = all_turns[:-self.recent_turns_limit] if len(all_turns) > self.recent_turns_limit else []
        
        # Step 3: Build context sections
        context_parts = []
        token_count = 0
        
        # Reserve tokens for retrieved context (if any)
        retrieved_token_budget = 500 if retrieved_context else 0
        recent_token_budget = self.token_budget - retrieved_token_budget
        
        # Add retrieved context (oldest, most relevant)
        retrieved_count = 0
        if retrieved_context:
            retrieved_parts = []
            for ctx in retrieved_context[:3]:  # Top 3 most relevant
                if token_count >= retrieved_token_budget:
                    break
                
                query = ctx.get('text', '')
                response = ctx.get('metadata', {}).get('response', '')
                similarity = ctx.get('similarity', 0) or ctx.get('rerank_score', 0)
                
                # Only include high-quality retrievals
                if similarity > 0.5 and query and response:
                    entry = f"[Earlier] User: {query[:150]}...\nAssistant: {response[:200]}..."
                    entry_tokens = count_tokens(entry)
                    
                    if token_count + entry_tokens <= retrieved_token_budget:
                        retrieved_parts.append(entry)
                        token_count += entry_tokens
                        retrieved_count += 1
            
            if retrieved_parts:
                context_parts.append("# Relevant earlier context:")
                context_parts.extend(retrieved_parts)
                context_parts.append("")
        
        # Add recent turns (most important - always include)
        recent_parts = []
        recent_count = 0
        for turn in recent_turns:
            entry = f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            entry_tokens = count_tokens(entry)
            
            # If over budget, truncate oldest recent turns
            if token_count + entry_tokens > self.token_budget:
                # Try to keep at least last 5 turns
                if recent_count >= 5:
                    break
                # Truncate this turn to fit
                remaining_budget = self.token_budget - token_count
                entry = truncate_to_budget(entry, remaining_budget)
                entry_tokens = count_tokens(entry)
            
            recent_parts.append(entry)
            token_count += entry_tokens
            recent_count += 1
        
        if recent_parts:
            context_parts.append("# Recent conversation:")
            context_parts.extend(recent_parts)
        
        formatted_context = "\n".join(context_parts)
        
        metadata = {
            'recent_turns': recent_count,
            'retrieved_turns': retrieved_count,
            'total_tokens': token_count,
            'budget_utilization': token_count / self.token_budget,
            'older_turns_available': len(older_turns)
        }
        
        return formatted_context, metadata
    
    def get_turns_for_offload(self, conversation_id: str) -> List[Dict]:
        """
        Get older turns that should be offloaded to vector DB
        
        Returns:
            List of turns older than recent_turns_limit
        """
        if conversation_id not in self.conversations:
            return []
        
        all_turns = self.conversations[conversation_id]
        
        # Only offload if we have more than recent_turns_limit + buffer
        if len(all_turns) <= self.recent_turns_limit + 5:
            return []
        
        # Get turns that are old enough to offload (keep recent + small buffer)
        turns_to_offload = all_turns[:-(self.recent_turns_limit + 5)]
        
        # Deduplicate before offloading (save storage space)
        deduplicated = self.deduplicate_turns(turns_to_offload)
        
        self.stats['turns_offloaded'] += len(deduplicated)
        
        return deduplicated
    
    def prune_offloaded_turns(self, conversation_id: str, num_turns: int):
        """
        Remove oldest N turns from memory (after successful offload to vector DB)
        
        Args:
            conversation_id: Conversation ID
            num_turns: Number of oldest turns to remove
        """
        if conversation_id not in self.conversations:
            return
        
        self.conversations[conversation_id] = self.conversations[conversation_id][num_turns:]
    
    def deduplicate_turns(self, turns: List[Dict]) -> List[Dict]:
        """
        Remove near-duplicate turns (simple text similarity)
        
        Args:
            turns: List of turn dicts
        
        Returns:
            Deduplicated list
        """
        if not turns:
            return turns
        
        # Simple deduplication: remove turns with very similar user queries
        unique_turns = []
        seen_queries = []
        
        for turn in turns:
            query = turn.get('user', '').lower().strip()
            
            # Check if very similar to any seen query
            is_duplicate = False
            for seen in seen_queries:
                # Simple string similarity (Jaccard)
                query_words = set(query.split())
                seen_words = set(seen.split())
                
                if query_words and seen_words:
                    similarity = len(query_words & seen_words) / len(query_words | seen_words)
                    if similarity > self.similarity_threshold:
                        is_duplicate = True
                        self.stats['turns_deduplicated'] += 1
                        break
            
            if not is_duplicate:
                unique_turns.append(turn)
                seen_queries.append(query)
        
        return unique_turns
    
    def get_recent_turns_count(self, conversation_id: str) -> int:
        """Get number of turns in memory for this conversation"""
        return len(self.conversations.get(conversation_id, []))
    
    def delete_conversation(self, conversation_id: str):
        """Delete conversation from memory"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            **self.stats,
            'active_conversations': len(self.conversations),
            'total_turns_in_memory': sum(len(turns) for turns in self.conversations.values())
        }
