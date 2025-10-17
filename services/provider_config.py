"""
Provider configuration for multi-provider routing
Maps models to their providers and endpoints
"""
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider"""
    name: str
    base_url: str
    format: Literal["openai", "anthropic", "custom"]
    models: List[str]  # Can include wildcards like "meta-llama/*"
    supports_streaming: bool = True
    auth_header: str = "Authorization"  # or "x-api-key" for Anthropic
    auth_prefix: str = "Bearer"  # or empty for Anthropic


# Provider configurations
PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        format="openai",
        models=[
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini"
        ],
        supports_streaming=True,
        auth_header="Authorization",
        auth_prefix="Bearer"
    ),

    "anthropic": ProviderConfig(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        format="anthropic",
        models=[
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022"
        ],
        supports_streaming=True,
        auth_header="x-api-key",
        auth_prefix=""  # Anthropic doesn't use Bearer prefix
    ),

    "xai": ProviderConfig(
        name="xai",
        base_url="https://api.x.ai/v1",
        format="openai",  # xAI uses OpenAI-compatible format
        models=["grok-beta", "grok-vision-beta"],
        supports_streaming=True,
        auth_header="Authorization",
        auth_prefix="Bearer"
    ),

    "together": ProviderConfig(
        name="together",
        base_url="https://api.together.xyz/v1",
        format="openai",
        models=[
            # Wildcards - match any model starting with prefix
            "meta-llama/*",
            "mistralai/*",
            "togethercomputer/*",
            "NousResearch/*",
            "Qwen/*"
        ],
        supports_streaming=True,
        auth_header="Authorization",
        auth_prefix="Bearer"
    ),

    "cerebras": ProviderConfig(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        format="openai",
        models=[
            "llama3.1-8b",
            "llama3.1-70b",
            "llama-3.3-70b"
        ],
        supports_streaming=True,
        auth_header="Authorization",
        auth_prefix="Bearer"
    ),

    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        format="openai",
        models=["*"],  # Accepts any model (acts as proxy to other providers)
        supports_streaming=True,
        auth_header="Authorization",
        auth_prefix="Bearer"
    ),

    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        format="openai",
        models=["deepseek-chat", "deepseek-coder"],
        supports_streaming=True,
        auth_header="Authorization",
        auth_prefix="Bearer"
    ),

    "groq": ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        format="openai",
        models=[
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        supports_streaming=True,
        auth_header="Authorization",
        auth_prefix="Bearer"
    )
}


def detect_provider(model: str) -> Optional[ProviderConfig]:
    """
    Detect provider from model name

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet", "meta-llama/Llama-3-8b")

    Returns:
        ProviderConfig if found, None otherwise
    """
    # First check exact matches
    for provider in PROVIDERS.values():
        if model in provider.models:
            return provider

    # Then check wildcard matches (e.g., "meta-llama/*")
    for provider in PROVIDERS.values():
        for pattern in provider.models:
            if "*" in pattern:
                # Convert wildcard to prefix match
                prefix = pattern.replace("/*", "/")
                if model.startswith(prefix):
                    return provider

                # Handle single wildcard (matches everything)
                if pattern == "*":
                    return provider

    return None


def get_provider(provider_name: str) -> Optional[ProviderConfig]:
    """Get provider config by name"""
    return PROVIDERS.get(provider_name)


def list_supported_models() -> Dict[str, List[str]]:
    """Get all supported models grouped by provider"""
    return {
        provider.name: provider.models
        for provider in PROVIDERS.values()
    }
