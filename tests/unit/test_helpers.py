"""
Unit tests for utility helpers
No external dependencies required
"""
import pytest
from utils.helpers import (
    resolve_conversation_id,
    count_tokens,
    truncate_text,
    sanitize_conversation_id
)
from models.requests import ChatCompletionRequest, Message


class TestConversationIDResolution:
    """Test conversation ID resolution logic"""

    def test_conversation_id_priority(self):
        """Test conversation_id takes priority"""
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="test")],
            conversation_id="conv_123",
            chat_id="chat_456",
            session_id="session_789"
        )
        user = {"user_id": "user_abc"}

        conv_id = resolve_conversation_id(request, user)
        assert "conv_123" in conv_id
        assert "user_abc" in conv_id

    def test_chat_id_fallback(self):
        """Test chat_id used if conversation_id not provided"""
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="test")],
            chat_id="chat_456"
        )
        user = {"user_id": "user_abc"}

        conv_id = resolve_conversation_id(request, user)
        assert "chat_456" in conv_id

    def test_session_id_fallback(self):
        """Test session_id used if conversation_id and chat_id not provided"""
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="test")],
            session_id="session_789"
        )
        user = {"user_id": "user_abc"}

        conv_id = resolve_conversation_id(request, user)
        assert "session_789" in conv_id

    def test_auto_generate_id(self):
        """Test UUID generated if no IDs provided"""
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="test")]
        )
        user = {"user_id": "user_abc"}

        conv_id = resolve_conversation_id(request, user)
        assert "user_abc" in conv_id
        assert len(conv_id) > 20  # Contains UUID

    def test_namespace_included(self):
        """Test namespace is included in conversation ID"""
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="test")],
            conversation_id="conv_123",
            namespace="production"
        )
        user = {"user_id": "user_abc"}

        conv_id = resolve_conversation_id(request, user)
        assert conv_id.startswith("production:")


class TestTokenCounting:
    """Test token counting utilities"""

    def test_count_tokens_approximation(self):
        """Test token counting gives reasonable approximation"""
        text = "Hello world, this is a test message."
        tokens = count_tokens(text)

        # Rough approximation: ~4 chars per token
        expected = len(text) // 4
        assert abs(tokens - expected) < 5

    def test_count_tokens_empty(self):
        """Test empty string returns 0 tokens"""
        assert count_tokens("") == 0

    def test_count_tokens_long_text(self):
        """Test long text token counting"""
        text = "word " * 1000
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count


class TestTextTruncation:
    """Test text truncation utilities"""

    def test_truncate_short_text(self):
        """Test short text not truncated"""
        text = "Short text"
        truncated = truncate_text(text, max_tokens=100)
        assert truncated == text

    def test_truncate_long_text(self):
        """Test long text is truncated"""
        text = "word " * 1000
        truncated = truncate_text(text, max_tokens=10)
        assert len(truncated) < len(text)
        assert truncated.endswith("...")

    def test_truncate_exact_length(self):
        """Test text at exact max length"""
        text = "a" * 400  # 100 tokens worth
        truncated = truncate_text(text, max_tokens=100)
        assert len(truncated) <= 400


class TestConversationIDSanitization:
    """Test conversation ID sanitization"""

    def test_sanitize_alphanumeric(self):
        """Test alphanumeric IDs pass through"""
        conv_id = "user_123_chat_abc"
        sanitized = sanitize_conversation_id(conv_id)
        assert sanitized == conv_id

    def test_sanitize_removes_special_chars(self):
        """Test special characters are removed"""
        conv_id = "user@123#chat$abc!"
        sanitized = sanitize_conversation_id(conv_id)
        assert "@" not in sanitized
        assert "#" not in sanitized
        assert "$" not in sanitized

    def test_sanitize_preserves_colons(self):
        """Test colons are preserved (for namespace:user:id format)"""
        conv_id = "namespace:user_123:chat_abc"
        sanitized = sanitize_conversation_id(conv_id)
        assert ":" in sanitized

    def test_sanitize_truncates_long_ids(self):
        """Test IDs longer than 256 chars are truncated"""
        conv_id = "a" * 500
        sanitized = sanitize_conversation_id(conv_id)
        assert len(sanitized) <= 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
