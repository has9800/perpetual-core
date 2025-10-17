"""
Enhanced Memory Manager with Anchor Context and High-Quality Retrieval
Production-ready with token tracking and auto-switching
"""
from typing import List, Dict, Optional
import time
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class EnhancedMemoryManager:
    """
    Enhanced memory manager with:
    - Anchor context (always-included important info)
    - Context window retrieval
    - Re-ranking
    - Token-aware mode switching
    """

    def __init__(
        self,
        vector_db,
        token_tracker=None,
        cache_capacity: int = 1000,
        ttl_days: int = 90
    ):
        """Initialize enhanced memory manager"""
        self.vector_db = vector_db
        self.token_tracker = token_tracker
        self.cache_capacity = cache_capacity
        self.ttl_seconds = ttl_days * 86400

        # LRU cache for recent turns
        self.recent_cache = OrderedDict()

        # Conversation metadata
        self.conversation_turns = {}
        self.anchor_context = {}  # {conv_id: [anchor_items]}

        # Statistics
        self.stats = {
            'total_turns': 0,
            'retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'mode_switches': {'full': 0, 'balanced': 0, 'safe': 0}
        }

    def add_turn(
        self,
        conversation_id: str,
        text: str,
        metadata: Optional[Dict] = None,
        is_anchor: bool = False
    ) -> bool:
        """
        Add conversation turn to memory

        Args:
            conversation_id: Unique conversation ID
            text: The text to store
            metadata: Additional data
            is_anchor: Mark as anchor context (always included)
        """
        try:
            # Track turn number
            if conversation_id not in self.conversation_turns:
                self.conversation_turns[conversation_id] = 0

            self.conversation_turns[conversation_id] += 1
            turn_number = self.conversation_turns[conversation_id]

            # Build metadata
            meta = metadata or {}
            meta.update({
                'conversation_id': conversation_id,
                'turn_number': turn_number,
                'timestamp': time.time(),
                'is_anchor': is_anchor
            })

            # Store in vector DB
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

                # Evict old entries
                if len(self.recent_cache) > self.cache_capacity:
                    self.recent_cache.popitem(last=False)

                # Store anchor if marked
                if is_anchor:
                    if conversation_id not in self.anchor_context:
                        self.anchor_context[conversation_id] = []
                    self.anchor_context[conversation_id].append({
                        'text': text,
                        'turn_number': turn_number,
                        'timestamp': time.time()
                    })

                self.stats['total_turns'] += 1

            return success

        except Exception as e:
            logger.error(f"Memory add error: {e}")
            return False

    async def retrieve_context_enhanced(
        self,
        conversation_id: str,
        query: str,
        token_budget: int = 8000,
        strategy: str = "ui_builder"
    ) -> Dict:
        """
        Enhanced context retrieval with auto-switching and anchors

        Args:
            conversation_id: Conversation to search
            query: Current query
            token_budget: Maximum tokens for context
            strategy: Context strategy (ui_builder, code_editor, chat)

        Returns:
            Dict with 'success', 'context', 'mode_used', 'components'
        """
        try:
            self.stats['retrievals'] += 1

            # Determine mode based on token tracker
            if self.token_tracker:
                use_retrieval, mode = self.token_tracker.should_use_retrieval(conversation_id)
                config = self.token_tracker.get_recommended_config(conversation_id)
            else:
                use_retrieval = True
                mode = "balanced"
                config = {"recent_turns": 30, "semantic_top_k": 5}

            self.stats['mode_switches'][mode] = self.stats['mode_switches'].get(mode, 0) + 1

            # Build context components
            components = []
            used_tokens = 0

            # 1. ANCHOR CONTEXT (always include if exists)
            anchors = self.get_anchor_context(conversation_id)
            for anchor in anchors[:3]:  # Max 3 anchors
                anchor_tokens = self._estimate_tokens(anchor['text'])
                if used_tokens + anchor_tokens < token_budget * 0.2:  # Max 20% for anchors
                    components.append({
                        'text': anchor['text'],
                        'source': 'anchor',
                        'priority': 100,
                        'tokens': anchor_tokens
                    })
                    used_tokens += anchor_tokens

            # 2. RECENT CONTEXT (for conversational flow)
            if not use_retrieval:
                # FULL MODE: Include all recent turns
                recent_limit = min(config['recent_turns'], 9999)
            else:
                # RETRIEVAL MODE: Limited recent turns
                recent_limit = config['recent_turns']

            recent_turns = self.get_recent_turns(conversation_id, limit=recent_limit)
            recent_text = "\n".join(recent_turns)
            recent_tokens = self._estimate_tokens(recent_text)

            # Adjust if exceeds budget
            if used_tokens + recent_tokens > token_budget * 0.6:
                # Reduce recent turns
                while recent_turns and recent_tokens > token_budget * 0.4:
                    recent_turns.pop(0)
                    recent_text = "\n".join(recent_turns)
                    recent_tokens = self._estimate_tokens(recent_text)

            if recent_turns:
                components.append({
                    'text': recent_text,
                    'source': 'recent',
                    'priority': 80,
                    'tokens': recent_tokens
                })
                used_tokens += recent_tokens

            # 3. SEMANTIC MATCHES (with context window)
            if use_retrieval and config['semantic_top_k'] > 0:
                remaining_budget = token_budget - used_tokens

                semantic_results = await self.vector_db.query_with_context_window(
                    conversation_id=conversation_id,
                    query_text=query,
                    top_k=config['semantic_top_k'],
                    context_window=2,  # ±2 turns
                    min_similarity=0.65
                )

                for result in semantic_results:
                    # Build enriched text with context window
                    enriched_text = self._build_windowed_text(result)
                    result_tokens = self._estimate_tokens(enriched_text)

                    if used_tokens + result_tokens <= token_budget:
                        components.append({
                            'text': enriched_text,
                            'source': f"semantic_{result.get('final_score', result['similarity']):.2f}",
                            'priority': 70 + (result.get('final_score', result['similarity']) * 20),
                            'tokens': result_tokens
                        })
                        used_tokens += result_tokens
                    else:
                        break  # Budget exhausted

            # Sort components by priority
            components.sort(key=lambda x: x['priority'], reverse=True)

            # Build final context string
            context_string = "\n\n---\n\n".join([
                f"[{comp['source']}]\n{comp['text']}"
                for comp in components
            ])

            return {
                'success': True,
                'context': context_string,
                'mode_used': mode,
                'components': components,
                'total_tokens': used_tokens,
                'token_budget': token_budget,
                'conversation_id': conversation_id
            }

        except Exception as e:
            logger.error(f"Enhanced retrieval error: {e}")
            # Fallback to basic retrieval
            basic_results = await self.vector_db.query(conversation_id, query, top_k=5)
            fallback_text = "\n".join([r['text'] for r in basic_results])

            return {
                'success': False,
                'context': fallback_text,
                'mode_used': 'fallback',
                'error': str(e)
            }

    def _build_windowed_text(self, result: Dict) -> str:
        """Build text with context window"""
        parts = []

        # Context before
        if result.get('context_window'):
            context_turns = result['context_window']
            before = [t for t in context_turns if t['turn_number'] < result['metadata']['turn_number']]
            if before:
                parts.append("...\n" + "\n".join([t['text'] for t in before]))

        # Main match
        parts.append(f">>> {result['text']} <<<")

        # Context after
        if result.get('context_window'):
            after = [t for t in context_turns if t['turn_number'] > result['metadata']['turn_number']]
            if after:
                parts.append("\n".join([t['text'] for t in after]) + "\n...")

        return "\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (chars / 4)"""
        if not text:
            return 0
        return len(text) // 4

    def get_anchor_context(self, conversation_id: str) -> List[Dict]:
        """Get anchor context for conversation"""
        return self.anchor_context.get(conversation_id, [])

    def add_anchor(
        self,
        conversation_id: str,
        text: str,
        tag: Optional[str] = None
    ) -> bool:
        """
        Add anchor context (important info that should always be included)

        Args:
            conversation_id: Conversation ID
            text: Anchor text (e.g., project goals, system state)
            tag: Optional tag for categorization
        """
        try:
            if conversation_id not in self.anchor_context:
                self.anchor_context[conversation_id] = []

            self.anchor_context[conversation_id].append({
                'text': text,
                'tag': tag,
                'timestamp': time.time()
            })

            return True
        except Exception as e:
            logger.error(f"Error adding anchor: {e}")
            return False

    def get_recent_turns(
        self,
        conversation_id: str,
        limit: int = 20
    ) -> List[str]:
        """Get N most recent turns from cache"""
        try:
            # Get all turns for this conversation
            conv_turns = [
                (key, value) for key, value in self.recent_cache.items()
                if key.startswith(conversation_id)
            ]

            # Sort by turn number
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
                    formatted.append(text)

            if formatted:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1

            return formatted

        except Exception as e:
            logger.error(f"Recent turns error: {e}")
            self.stats['cache_misses'] += 1
            return []

    def get_conversation_length(self, conversation_id: str) -> int:
        """Get total number of turns"""
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

            # Remove metadata
            if conversation_id in self.conversation_turns:
                del self.conversation_turns[conversation_id]

            if conversation_id in self.anchor_context:
                del self.anchor_context[conversation_id]

            # Reset token tracking
            if self.token_tracker:
                self.token_tracker.reset_conversation(conversation_id)

        except Exception as e:
            logger.error(f"Delete conversation error: {e}")

    def get_metrics(self) -> Dict:
        """Get memory system metrics"""
        return {
            'active_conversations': len(self.conversation_turns),
            'total_turns': self.stats['total_turns'],
            'retrievals': self.stats['retrievals'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
            ),
            'cache_size': len(self.recent_cache),
            'mode_distribution': self.stats['mode_switches']
        }
