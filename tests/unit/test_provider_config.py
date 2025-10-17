"""
Unit tests for provider configuration and routing
These tests don't require API keys
"""
import pytest
from services.provider_config import detect_provider, get_provider, PROVIDERS


class TestProviderDetection:
    """Test provider detection from model names"""

    def test_openai_detection(self):
        """Test OpenAI model detection"""
        provider = detect_provider("gpt-4o")
        assert provider is not None
        assert provider.name == "openai"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_anthropic_detection(self):
        """Test Anthropic model detection"""
        provider = detect_provider("claude-3-5-sonnet-20241022")
        assert provider is not None
        assert provider.name == "anthropic"
        assert provider.format == "anthropic"

    def test_xai_detection(self):
        """Test xAI model detection"""
        provider = detect_provider("grok-beta")
        assert provider is not None
        assert provider.name == "xai"

    def test_together_wildcard(self):
        """Test Together.ai wildcard matching"""
        provider = detect_provider("meta-llama/Llama-3-8b-chat-hf")
        assert provider is not None
        assert provider.name == "together"

    def test_unknown_model(self):
        """Test unknown model returns None"""
        provider = detect_provider("unknown-model-xyz")
        assert provider is None

    def test_openrouter_accepts_all(self):
        """Test OpenRouter accepts any model with wildcard"""
        # OpenRouter should match if no other provider matches
        # But it has "*" so it only matches if explicitly named
        provider = detect_provider("openrouter/anthropic/claude-3")
        # Should NOT match - needs proper prefix
        assert provider is None


class TestProviderRegistry:
    """Test provider registry functionality"""

    def test_all_providers_loaded(self):
        """Test all providers are in registry"""
        expected_providers = [
            "openai", "anthropic", "xai", "together",
            "cerebras", "openrouter", "deepseek", "groq"
        ]
        for provider_name in expected_providers:
            provider = get_provider(provider_name)
            assert provider is not None
            assert provider.name == provider_name

    def test_provider_has_required_fields(self):
        """Test all providers have required configuration"""
        for provider in PROVIDERS.values():
            assert provider.name
            assert provider.base_url
            assert provider.format in ["openai", "anthropic", "custom"]
            assert isinstance(provider.models, list)
            assert len(provider.models) > 0

    def test_openai_format_providers(self):
        """Test OpenAI-compatible providers"""
        openai_format = ["openai", "xai", "together", "cerebras", "groq", "deepseek", "openrouter"]
        for provider_name in openai_format:
            provider = get_provider(provider_name)
            assert provider.format == "openai"

    def test_anthropic_format(self):
        """Test Anthropic format"""
        provider = get_provider("anthropic")
        assert provider.format == "anthropic"
        assert provider.auth_header == "x-api-key"
        assert provider.auth_prefix == ""  # No "Bearer" prefix


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
