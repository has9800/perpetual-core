"""
Token Tracker Service with Redis
Tracks conversation token usage for auto-switching between full and retrieval modes
"""
import redis
from typing import Optional, Tuple, Dict
import time
import json
import logging

logger = logging.getLogger(__name__)


class TokenTracker:
    """
    Lightweight token tracker using Redis for scale
    Uses character-based approximation for speed
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        redis_url: Optional[str] = None,
        char_to_token_ratio: float = 3.8
    ):
        """
        Initialize token tracker

        Args:
            redis_client: Existing Redis client (optional)
            redis_url: Redis connection URL (optional, defaults to localhost)
            char_to_token_ratio: Average chars per token (default: 3.8)
        """
        if redis_client:
            self.redis = redis_client
        else:
            redis_url = redis_url or "redis://localhost:6379/0"
            try:
                self.redis = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                self.redis.ping()
                logger.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise

        self.char_to_token_ratio = char_to_token_ratio

        # Thresholds for mode switching
        self.thresholds = {
            'full': 5000,        # < 5k tokens: use full context
            'balanced': 20000,   # 5k-20k: use balanced retrieval
            'safe': float('inf') # 20k+: use safe mode
        }

    def quick_estimate(self, text: str) -> int:
        """
        Fast token estimation using character count

        Args:
            text: Input text

        Returns:
            Estimated token count (±10% accuracy)
        """
        if not text:
            return 0
        # Conservative estimate: chars / 4
        return len(text) // 4

    def track_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        actual_tokens: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Track tokens for a conversation turn

        Args:
            conversation_id: Unique conversation identifier
            user_message: User's message text
            assistant_message: Assistant's response text
            actual_tokens: Actual token count from LLM provider (optional)

        Returns:
            Dict with conversation metadata
        """
        try:
            # Estimate tokens for this turn
            if actual_tokens:
                turn_tokens = actual_tokens
            else:
                turn_tokens = (
                    self.quick_estimate(user_message) +
                    self.quick_estimate(assistant_message)
                )

            # Get current conversation data
            key = f"conv:{conversation_id}:tokens"
            conv_data = self.redis.get(key)

            if conv_data:
                data = json.loads(conv_data)
            else:
                data = {
                    "total_tokens": 0,
                    "turn_count": 0,
                    "created_at": time.time(),
                    "current_mode": "full"
                }

            # Update metrics
            data["total_tokens"] += turn_tokens
            data["turn_count"] += 1
            data["last_updated"] = time.time()
            data["last_turn_tokens"] = turn_tokens

            # Determine appropriate mode
            total = data["total_tokens"]
            if total < self.thresholds['full']:
                data["current_mode"] = "full"
            elif total < self.thresholds['balanced']:
                data["current_mode"] = "balanced"
            else:
                data["current_mode"] = "safe"

            # Save to Redis with 30 day TTL
            self.redis.setex(
                key,
                30 * 24 * 60 * 60,  # 30 days
                json.dumps(data)
            )

            return data

        except Exception as e:
            logger.error(f"Error tracking turn for {conversation_id}: {e}")
            # Return safe defaults on error
            return {
                "total_tokens": turn_tokens,
                "turn_count": 1,
                "current_mode": "full",
                "error": str(e)
            }

    def get_conversation_stats(self, conversation_id: str) -> Optional[Dict]:
        """
        Get current conversation statistics

        Args:
            conversation_id: Conversation to query

        Returns:
            Dict with stats or None if not found
        """
        try:
            key = f"conv:{conversation_id}:tokens"
            data = self.redis.get(key)

            if data:
                return json.loads(data)
            return None

        except Exception as e:
            logger.error(f"Error getting stats for {conversation_id}: {e}")
            return None

    def should_use_retrieval(
        self,
        conversation_id: str
    ) -> Tuple[bool, str]:
        """
        Determine if retrieval should be used and which mode

        Args:
            conversation_id: Conversation to check

        Returns:
            Tuple of (use_retrieval: bool, mode: str)
        """
        try:
            stats = self.get_conversation_stats(conversation_id)

            if not stats:
                # New conversation - use full context
                return False, "full"

            total_tokens = stats["total_tokens"]

            if total_tokens < self.thresholds['full']:
                return False, "full"
            elif total_tokens < self.thresholds['balanced']:
                return True, "balanced"
            else:
                return True, "safe"

        except Exception as e:
            logger.error(f"Error checking retrieval for {conversation_id}: {e}")
            # Safe default: use retrieval with balanced mode
            return True, "balanced"

    def get_recommended_config(
        self,
        conversation_id: str
    ) -> Dict[str, int]:
        """
        Get recommended memory_config based on conversation state

        Args:
            conversation_id: Conversation to analyze

        Returns:
            Dict with recent_turns and semantic_top_k values
        """
        use_retrieval, mode = self.should_use_retrieval(conversation_id)

        configs = {
            "full": {
                "recent_turns": 999999,  # Send everything
                "semantic_top_k": 0
            },
            "balanced": {
                "recent_turns": 30,
                "semantic_top_k": 5
            },
            "safe": {
                "recent_turns": 40,
                "semantic_top_k": 7
            }
        }

        return configs[mode]

    def reset_conversation(self, conversation_id: str) -> bool:
        """
        Reset token tracking for a conversation

        Args:
            conversation_id: Conversation to reset

        Returns:
            True if successful
        """
        try:
            key = f"conv:{conversation_id}:tokens"
            self.redis.delete(key)
            logger.info(f"Reset tracking for {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting {conversation_id}: {e}")
            return False

    def get_global_stats(self) -> Dict:
        """
        Get global statistics across all conversations

        Returns:
            Dict with aggregate stats
        """
        try:
            # Scan for all conversation keys
            keys = list(self.redis.scan_iter("conv:*:tokens"))

            total_conversations = len(keys)
            total_tokens = 0
            total_turns = 0
            mode_distribution = {"full": 0, "balanced": 0, "safe": 0}

            for key in keys:
                try:
                    data = json.loads(self.redis.get(key))
                    total_tokens += data.get("total_tokens", 0)
                    total_turns += data.get("turn_count", 0)
                    mode = data.get("current_mode", "full")
                    mode_distribution[mode] = mode_distribution.get(mode, 0) + 1
                except:
                    continue

            return {
                "total_conversations": total_conversations,
                "total_tokens_tracked": total_tokens,
                "total_turns": total_turns,
                "avg_tokens_per_conversation": total_tokens / total_conversations if total_conversations > 0 else 0,
                "avg_turns_per_conversation": total_turns / total_conversations if total_conversations > 0 else 0,
                "mode_distribution": mode_distribution
            }

        except Exception as e:
            logger.error(f"Error getting global stats: {e}")
            return {
                "error": str(e)
            }

    def cleanup_old_conversations(self, days: int = 30) -> int:
        """
        Clean up conversations older than specified days

        Args:
            days: Age threshold in days

        Returns:
            Number of conversations cleaned up
        """
        try:
            cutoff = time.time() - (days * 24 * 60 * 60)
            keys = list(self.redis.scan_iter("conv:*:tokens"))
            cleaned = 0

            for key in keys:
                try:
                    data = json.loads(self.redis.get(key))
                    last_updated = data.get("last_updated", data.get("created_at", 0))

                    if last_updated < cutoff:
                        self.redis.delete(key)
                        cleaned += 1
                except:
                    continue

            logger.info(f"Cleaned up {cleaned} old conversations")
            return cleaned

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0


def get_token_tracker(redis_url: Optional[str] = None) -> TokenTracker:
    """
    Factory function to create TokenTracker instance

    Args:
        redis_url: Redis connection URL

    Returns:
        TokenTracker instance
    """
    return TokenTracker(redis_url=redis_url)
