"""
Custom validation functions
"""
from fastapi import HTTPException
from typing import Dict, Optional


def validate_tier_access(
    user_context: Dict,
    required_tier: str
) -> bool:
    """
    Validate user has required tier
    
    Args:
        user_context: User context from auth
        required_tier: Required tier (free, pro, enterprise)
        
    Returns:
        True if user has access
        
    Raises:
        HTTPException if access denied
    """
    tier_hierarchy = {
        'free': 0,
        'pro': 1,
        'enterprise': 2
    }
    
    user_tier = user_context.get('tier', 'free')
    
    if tier_hierarchy.get(user_tier, 0) < tier_hierarchy.get(required_tier, 0):
        raise HTTPException(
            status_code=403,
            detail=f"This feature requires {required_tier} tier"
        )
    
    return True


def validate_conversation_id(conv_id: str) -> bool:
    """
    Validate conversation ID format
    
    Args:
        conv_id: Conversation ID
        
    Returns:
        True if valid
        
    Raises:
        HTTPException if invalid
    """
    if not conv_id:
        raise HTTPException(400, "conversation_id cannot be empty")
    
    if len(conv_id) > 256:
        raise HTTPException(400, "conversation_id too long (max 256 chars)")
    
    # Must contain namespace:user_id:id format
    parts = conv_id.split(':')
    if len(parts) != 3:
        raise HTTPException(
            400,
            "conversation_id must be in format namespace:user_id:id"
        )
    
    return True


def validate_token_limit(
    input_tokens: int,
    max_tokens: int,
    model_max: int = 4096
) -> bool:
    """
    Validate token limits
    
    Args:
        input_tokens: Input token count
        max_tokens: Requested max output tokens
        model_max: Model's max context
        
    Returns:
        True if valid
        
    Raises:
        HTTPException if invalid
    """
    if input_tokens + max_tokens > model_max:
        raise HTTPException(
            400,
            f"Total tokens ({input_tokens + max_tokens}) exceeds model max ({model_max})"
        )
    
    return True
