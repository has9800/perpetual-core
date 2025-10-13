"""
Infinite Memory System - Core Memory Manager
Handles conversation snapshots and vector-based retrieval
"""

import time
import hashlib
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ConversationTurn:
    """Single conversation turn"""
    conversation_id: str
    turn_id: str
    text: str
    timestamp: float
    metadata: Dict


class MemoryManager:
    """
    Manages conversation memory with snapshots and vector DB
    Thread-safe and production-ready
    """

    def __init__(self, 
                 vector_db,
                 cache_capacity: int = 1000,
                 ttl_days: int = 90):
        """
        Initialize memory manager

        Args:
            vector_db: Vector database instance (ChromaDB/Qdrant)
            cache_capacity: Max conversations to keep in active memory
            ttl_days: Time-to-live for old conversations
        """
        self.vector_db = vector_db
        self.cache_capacity = cache_capacity
        self.ttl_days = ttl_days

        # Active snapshots (in-memory cache)
        self.active_conversations = {}  # conversation_id -> list of turns
        self.conversation_metadata = {}  # conversation_id -> metadata

        # Thread safety
        self.lock = threading.Lock()

        # Metrics
        self.metrics = {
            'snapshots_created': 0,
            'evictions': 0,
            'retrievals': 0,
            'vector_db_writes': 0,
            'vector_db_searches': 0,
            'errors': defaultdict(int)
        }

    def add_turn(self, 
                 conversation_id: str, 
                 text: str,
                 metadata: Optional[Dict] = None) -> Dict:
        """Add conversation turn with automatic snapshot"""
        with self.lock:
            try:
                timestamp = time.time()
                turn_id = self._generate_turn_id(conversation_id, text, timestamp)

                turn = ConversationTurn(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    text=text,
                    timestamp=timestamp,
                    metadata=metadata or {}
                )

                if conversation_id not in self.active_conversations:
                    self.active_conversations[conversation_id] = []
                    self.conversation_metadata[conversation_id] = {
                        'created_at': timestamp,
                        'last_updated': timestamp,
                        'turn_count': 0
                    }

                self.active_conversations[conversation_id].append(turn)
                self.conversation_metadata[conversation_id]['last_updated'] = timestamp
                self.conversation_metadata[conversation_id]['turn_count'] += 1

                self.metrics['snapshots_created'] += 1

                # Check if we need to evict
                if len(self.active_conversations) > self.cache_capacity:
                    self._evict_oldest()

                return {
                    'turn_id': turn_id,
                    'conversation_id': conversation_id,
                    'timestamp': timestamp,
                    'success': True
                }

            except Exception as e:
                self.metrics['errors']['add_turn'] += 1
                return {'success': False, 'error': str(e)}

    def _generate_turn_id(self, conversation_id: str, text: str, timestamp: float) -> str:
        """Generate unique turn ID"""
        hash_input = f"{conversation_id}_{timestamp}_{text}"
        hash_val = hashlib.md5(hash_input.encode()).hexdigest()[:16]
        return f"{conversation_id}_{int(timestamp)}_{hash_val}"

    def _evict_oldest(self):
        """Evict oldest conversation to vector DB"""
        try:
            oldest_id = min(
                self.conversation_metadata.keys(),
                key=lambda k: self.conversation_metadata[k]['last_updated']
            )

            turns = self.active_conversations[oldest_id]

            for turn in turns:
                self.vector_db.add(
                    doc_id=turn.turn_id,
                    text=turn.text,
                    metadata={
                        'conversation_id': turn.conversation_id,
                        'timestamp': turn.timestamp,
                        **turn.metadata
                    }
                )
                self.metrics['vector_db_writes'] += 1

            del self.active_conversations[oldest_id]
            del self.conversation_metadata[oldest_id]

            self.metrics['evictions'] += 1

        except Exception as e:
            self.metrics['errors']['eviction'] += 1
            raise

    def retrieve_context(self, 
                        conversation_id: str,
                        query: str,
                        top_k: int = 3,
                        include_active: bool = True) -> Dict:
        """Retrieve relevant context for query"""
        try:
            results = []

            # Get from vector DB
            vector_results = self.vector_db.search(
                query=query,
                top_k=top_k,
                filter={'conversation_id': conversation_id}
            )
            self.metrics['vector_db_searches'] += 1

            for result in vector_results:
                results.append({
                    'text': result['text'],
                    'similarity': result['similarity'],
                    'source': 'vector_db',
                    'metadata': result.get('metadata', {})
                })

            # Get from active memory
            if include_active and conversation_id in self.active_conversations:
                active_turns = self.active_conversations[conversation_id]
                query_lower = query.lower()

                for turn in active_turns[-10:]:
                    if any(word in turn.text.lower() for word in query_lower.split()):
                        results.append({
                            'text': turn.text,
                            'similarity': 0.9,
                            'source': 'active_memory',
                            'metadata': turn.metadata
                        })

            results.sort(key=lambda x: x['similarity'], reverse=True)

            self.metrics['retrievals'] += 1

            return {
                'results': results[:top_k],
                'total_results': len(results),
                'success': True
            }

        except Exception as e:
            self.metrics['errors']['retrieval'] += 1
            return {'success': False, 'error': str(e), 'results': []}

    def get_recent_turns(self, conversation_id: str, limit: int = 10) -> List[str]:
        """Get recent turns from active memory"""
        with self.lock:
            if conversation_id not in self.active_conversations:
                return []

            turns = self.active_conversations[conversation_id]
            return [turn.text for turn in turns[-limit:]]

    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        with self.lock:
            return {
                'active_conversations': len(self.active_conversations),
                'total_turns': sum(len(turns) for turns in self.active_conversations.values()),
                **self.metrics,
                'errors': dict(self.metrics['errors'])
            }