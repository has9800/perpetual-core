"""
Memory Manager for Infinite Context
FIXED: Proper turn storage, retrieval, and conversation ID consistency
"""

from typing import List, Dict, Optional
import time
from collections import OrderedDict


class MemoryManager:
    """Manages conversation memory with vector DB and local cache"""

    def __init__(self,
                 vector_db,
                 cache_capacity: int = 1000,
                 ttl_days: int = 90):
        """Initialize memory manager"""
        self.vector_db = vector_db
        self.cache_capacity = cache_capacity
        self.ttl_seconds = ttl_days * 86400

        # LRU cache for recent turns (fast access)
        self.recent_cache = OrderedDict()

        # Conversation metadata
        self.conversation_turns = {}  # conversation_id -> turn_count

        # Statistics
        self.stats = {
            'total_turns': 0,
            'snapshots_created': 0,
            'retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0
        }

    def add_turn(self,
                conversation_id: str,
                text: str,
                metadata: Optional[Dict] = None) -> bool:
        """
        Add conversation turn to memory

        Args:
            conversation_id: Unique conversation ID (stays same for all turns)
            text: The user query (what we search for later)
            metadata: Additional data (should include 'response')
        """
        try:
            # Track turn number for this conversation
            if conversation_id not in self.conversation_turns:
                self.conversation_turns[conversation_id] = 0

            self.conversation_turns[conversation_id] += 1
            turn_number = self.conversation_turns[conversation_id]

            # Build metadata
            meta = metadata or {}
            meta.update({
                'conversation_id': conversation_id,
                'turn_number': turn_number,
                'timestamp': time.time()
            })

            # Store in vector DB (text = user query for searching)
            success = self.vector_db.add(
                conversation_id=conversation_id,
                text=text,
                metadata=meta
            )

            if success:
                # Add to recent cache
                cache_key = f"{conversation_id}_{turn_number}"
                self.recent_cache[cache_key] = {
                    'text': text,
                    'metadata': meta
                }

                # Evict old entries from cache
                if len(self.recent_cache) > self.cache_capacity:
                    self.recent_cache.popitem(last=False)
                    self.stats['evictions'] += 1

                self.stats['total_turns'] += 1
                self.stats['snapshots_created'] += 1

            return success

        except Exception as e:
            print(f"Memory add error: {e}")
            return False

    def retrieve_context(self,
                        conversation_id: str,
                        query: str,
                        top_k: int = 3) -> Dict:
        """
        Retrieve relevant context from conversation history

        Args:
            conversation_id: Conversation to search within
            query: User query to find similar past exchanges
            top_k: Number of relevant turns to retrieve

        Returns:
            Dict with 'results' list and 'metadata'
        """
        try:
            self.stats['retrievals'] += 1

            # Query vector DB for similar past exchanges
            results = self.vector_db.query(
                conversation_id=conversation_id,
                query_text=query,
                top_k=top_k
            )

            # Sort by similarity (highest first)
            results = sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)

            return {
                'results': results,
                'conversation_id': conversation_id,
                'query': query,
                'retrieved_count': len(results)
            }

        except Exception as e:
            print(f"Memory retrieval error: {e}")
            return {
                'results': [],
                'conversation_id': conversation_id,
                'error': str(e)
            }

    def get_recent_turns(self,
                        conversation_id: str,
                        limit: int = 5) -> List[str]:
        """Get N most recent turns from cache"""
        try:
            # Get all turns for this conversation from cache
            conv_turns = [
                (key, value) for key, value in self.recent_cache.items()
                if key.startswith(conversation_id)
            ]

            # Sort by turn number (most recent last)
            conv_turns.sort(key=lambda x: x[1]['metadata'].get('turn_number', 0))

            # Take last N turns
            recent = conv_turns[-limit:]

            # Format as strings
            formatted = []
            for _, turn in recent:
                text = turn['text']
                response = turn['metadata'].get('response', '')
                if response:
                    formatted.append(f"User: {text}\nAssistant: {response}")
                else:
                    formatted.append(f"User: {text}")

            if formatted:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1

            return formatted

        except Exception as e:
            print(f"Recent turns error: {e}")
            return []

    def get_conversation_length(self, conversation_id: str) -> int:
        """Get total number of turns in conversation"""
        return self.conversation_turns.get(conversation_id, 0)

    def delete_conversation(self, conversation_id: str):
        """Delete all data for a conversation"""
        try:
            # Delete from vector DB
            self.vector_db.delete_conversation(conversation_id)

            # Remove from cache
            keys_to_remove = [
                key for key in self.recent_cache
                if key.startswith(conversation_id)
            ]
            for key in keys_to_remove:
                del self.recent_cache[key]

            # Remove from metadata
            if conversation_id in self.conversation_turns:
                del self.conversation_turns[conversation_id]

        except Exception as e:
            print(f"Delete conversation error: {e}")

    def get_metrics(self) -> Dict:
        """Get memory system metrics"""
        return {
            'active_conversations': len(self.conversation_turns),
            'total_turns': self.stats['total_turns'],
            'snapshots_created': self.stats['snapshots_created'],
            'evictions': self.stats['evictions'],
            'retrievals': self.stats['retrievals'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
            ),
            'cache_size': len(self.recent_cache),
            'cache_utilization': len(self.recent_cache) / self.cache_capacity
        }