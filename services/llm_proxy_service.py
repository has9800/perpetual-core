"""
LLM Proxy Service
Forwards requests to external LLM APIs with multi-provider support
"""
import httpx
import os
from typing import List, Dict, Optional, AsyncGenerator
from supabase import Client
import json
import time
from services.provider_config import detect_provider, get_provider, ProviderConfig
from services.api_key_service import get_api_key_service


class LLMProxyService:
    """
    Proxy service for forwarding to external LLM APIs
    Supports OpenAI, Anthropic, xAI, Together, Cerebras, OpenRouter, and more
    """

    def __init__(self):
        """Initialize proxy service"""
        # Timeout configuration
        self.timeout = httpx.Timeout(120.0, connect=10.0)

        # API key service for encrypted storage
        self.api_key_service = get_api_key_service()

    async def forward_to_openai(
        self,
        messages: List[Dict],
        model: str = "gpt-4o-mini",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False
    ) -> Dict:
        """
        Forward request to OpenAI API

        Args:
            messages: List of message dicts with role and content
            model: OpenAI model name
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stream: Whether to stream response

        Returns:
            Response dict in OpenAI format
        """
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if stream:
            payload["stream"] = True

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.openai_base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()

    async def forward_to_openai_stream(
        self,
        messages: List[Dict],
        model: str = "gpt-4o-mini",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0
    ) -> AsyncGenerator[str, None]:
        """
        Forward request to OpenAI API with streaming

        Yields SSE-formatted chunks
        """
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.openai_base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        yield line + "\n"

    async def forward_to_anthropic(
        self,
        messages: List[Dict],
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict:
        """
        Forward request to Anthropic API

        Args:
            messages: List of message dicts
            model: Anthropic model name
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response

        Returns:
            Response dict in Anthropic format (converted to OpenAI format)
        """
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        # Convert OpenAI message format to Anthropic format
        # Separate system messages from user/assistant messages
        system_messages = [m for m in messages if m.get("role") == "system"]
        conversation_messages = [m for m in messages if m.get("role") != "system"]

        system_prompt = "\n".join([m["content"] for m in system_messages]) if system_messages else None

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": conversation_messages,
            "temperature": temperature
        }

        if system_prompt:
            payload["system"] = system_prompt

        if stream:
            payload["stream"] = True

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.anthropic_base_url}/messages",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            anthropic_response = response.json()

            # Convert Anthropic format to OpenAI format
            return self._convert_anthropic_to_openai(anthropic_response, model)

    def _convert_anthropic_to_openai(self, anthropic_response: Dict, model: str) -> Dict:
        """Convert Anthropic response format to OpenAI format"""
        # Extract text from content blocks
        text_content = ""
        if "content" in anthropic_response:
            for block in anthropic_response["content"]:
                if block.get("type") == "text":
                    text_content += block.get("text", "")

        # Build OpenAI-compatible response
        return {
            "id": anthropic_response.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text_content
                },
                "finish_reason": self._convert_stop_reason(anthropic_response.get("stop_reason"))
            }],
            "usage": {
                "prompt_tokens": anthropic_response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": anthropic_response.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    anthropic_response.get("usage", {}).get("input_tokens", 0) +
                    anthropic_response.get("usage", {}).get("output_tokens", 0)
                )
            }
        }

    def _convert_stop_reason(self, anthropic_stop_reason: Optional[str]) -> str:
        """Convert Anthropic stop reason to OpenAI format"""
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop"
        }
        return mapping.get(anthropic_stop_reason, "stop")

    async def forward_request(
        self,
        messages: List[Dict],
        model: str,
        user_id: str,
        supabase: Client,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False
    ) -> Dict:
        """
        Universal router - forwards to appropriate provider based on model name

        Args:
            messages: List of message dicts
            model: Model name (determines provider)
            user_id: User ID for API key lookup
            supabase: Supabase client
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stream: Whether to stream response

        Returns:
            Response dict in OpenAI format
        """
        # Detect provider from model name
        provider_config = detect_provider(model)

        if not provider_config:
            raise ValueError(f"Unknown model: {model}. No provider configured for this model.")

        # Get user's API key for this provider
        api_key = await self.api_key_service.get_user_api_key(
            supabase=supabase,
            user_id=user_id,
            provider=provider_config.name
        )

        if not api_key:
            raise ValueError(
                f"No API key found for provider '{provider_config.name}'. "
                f"Please add your {provider_config.name} API key in settings."
            )

        # Forward based on format
        if provider_config.format == "openai":
            return await self._forward_openai_format(
                provider_config=provider_config,
                api_key=api_key,
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream
            )
        elif provider_config.format == "anthropic":
            return await self._forward_anthropic_format(
                provider_config=provider_config,
                api_key=api_key,
                messages=messages,
                model=model,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                stream=stream
            )
        else:
            raise ValueError(f"Unsupported format: {provider_config.format}")

    async def _forward_openai_format(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False
    ) -> Dict:
        """Forward to OpenAI-compatible endpoint"""
        headers = {
            f"{provider_config.auth_header}": f"{provider_config.auth_prefix} {api_key}".strip(),
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if stream:
            payload["stream"] = True

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{provider_config.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()

    async def _forward_anthropic_format(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        messages: List[Dict],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict:
        """Forward to Anthropic API with format conversion"""
        headers = {
            f"{provider_config.auth_header}": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        # Convert OpenAI format to Anthropic format
        system_messages = [m for m in messages if m.get("role") == "system"]
        conversation_messages = [m for m in messages if m.get("role") != "system"]
        system_prompt = "\n".join([m["content"] for m in system_messages]) if system_messages else None

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": conversation_messages,
            "temperature": temperature
        }

        if system_prompt:
            payload["system"] = system_prompt

        if stream:
            payload["stream"] = True

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{provider_config.base_url}/messages",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            anthropic_response = response.json()

            # Convert to OpenAI format
            return self._convert_anthropic_to_openai(anthropic_response, model)


# Singleton instance
_llm_proxy_service: Optional[LLMProxyService] = None


def get_llm_proxy_service() -> LLMProxyService:
    """Get or create LLM proxy service instance"""
    global _llm_proxy_service
    if _llm_proxy_service is None:
        _llm_proxy_service = LLMProxyService()
    return _llm_proxy_service
