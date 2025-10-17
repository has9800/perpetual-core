"""
Utility helper functions
"""
from models.requests import ChatCompletionRequest, MemoryQueryRequest
from typing import Dict, Union
import uuid
import re


def resolve_conversation_id(
    request: Union[ChatCompletionRequest, MemoryQueryRequest, Dict],
    user_context: Dict
) -> str:
    """
    Resolve conversation_id from multiple possible fields
    
    Priority: conversation_id > chat_id > session_id > thread_id > generate new
    
    Args:
        request: Request object with possible ID fields
        user_context: User context from auth
        
    Returns:
        Resolved conversation ID in format: namespace:user_id:id
    """
    namespace = getattr(request, 'namespace', 'default')
    user_id = user_context['user_id']
    
    # Try each field in priority order
    for field in ['conversation_id', 'chat_id', 'session_id', 'thread_id']:
        if hasattr(request, field):
            value = getattr(request, field)
            if value:
                return f"{namespace}:{user_id}:{value}"
    
    # Generate new ID if none provided
    new_id = str(uuid.uuid4())
    return f"{namespace}:{user_id}:{new_id}"


def count_tokens(text: str) -> int:
    """
    Estimate token count (simple approximation)
    
    For production, use tiktoken or model-specific tokenizer
    
    Args:
        text: Text to count
        
    Returns:
        Estimated token count
    """
    # Simple approximation: ~4 chars per token
    # For production, use: tiktoken.encoding_for_model(model).encode(text)
    return len(text) // 4


def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncate text to max tokens
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        
    Returns:
        Truncated text
    """
    estimated_chars = max_tokens * 4
    if len(text) <= estimated_chars:
        return text
    
    return text[:estimated_chars] + "..."


def format_messages_for_vllm(messages: list) -> str:
    """
    Format messages list into single prompt for vLLM
    
    Args:
        messages: List of message dicts with role and content
        
    Returns:
        Formatted prompt string
    """
    formatted = []
    
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'system':
            formatted.append(f"System: {content}")
        elif role == 'user':
            formatted.append(f"User: {content}")
        elif role == 'assistant':
            formatted.append(f"Assistant: {content}")
    
    return "\n\n".join(formatted)


def sanitize_conversation_id(conv_id: str) -> str:
    """
    Sanitize conversation ID for storage
    
    Args:
        conv_id: Raw conversation ID
        
    Returns:
        Sanitized ID (alphanumeric, hyphens, underscores only)
    """
    # Remove invalid characters
    sanitized = re.sub(r'[^a-zA-Z0-9\-_:]', '', conv_id)
    
    # Limit length
    if len(sanitized) > 256:
        sanitized = sanitized[:256]
    
    return sanitized


def build_context_prompt(retrieved_memories: list, max_memories: int = 5) -> str:
    """
    Build context prompt from retrieved memories
    
    Args:
        retrieved_memories: List of memory dicts
        max_memories: Max number to include
        
    Returns:
        Formatted context string
    """
    if not retrieved_memories:
        return ""
    
    context_parts = [
        "Relevant context from previous conversation:"
    ]
    
    for i, memory in enumerate(retrieved_memories[:max_memories]):
        text = memory.get('text', '')
        similarity = memory.get('similarity', 0)
        
        context_parts.append(
            f"\n[Context {i+1}] (relevance: {similarity:.2f})\n{text}"
        )
    
    return "\n".join(context_parts)
