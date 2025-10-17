"""
Authentication service using Supabase
"""
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client
from typing import Dict, Optional
import hashlib
import time
from config.settings import get_settings

settings = get_settings()
security = HTTPBearer()


class AuthService:
    """Supabase authentication service"""
    
    def __init__(self):
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        self.api_key_cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def validate_api_key(
        self,
        credentials: HTTPAuthorizationCredentials = Security(security)
    ) -> Dict:
        """
        Validate API key and return user context
        
        Args:
            credentials: Bearer token from request
            
        Returns:
            User context dict with user_id, tier, rate limits
            
        Raises:
            HTTPException: If API key is invalid
        """
        api_key = credentials.credentials
        
        # Check cache first
        if api_key in self.api_key_cache:
            cached = self.api_key_cache[api_key]
            if time.time() - cached['cached_at'] < self.cache_ttl:
                return cached['user_context']
        
        # Hash the API key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Query Supabase
        try:
            result = self.supabase.table('api_keys').select(
                'user_id, tier, rate_limit_per_minute, rate_limit_per_day, is_active'
            ).eq('key_hash', key_hash).single().execute()
            
            if not result.data:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key"
                )
            
            key_data = result.data
            
            # Check if key is active
            if not key_data.get('is_active', False):
                raise HTTPException(
                    status_code=401,
                    detail="API key has been deactivated"
                )
            
            # Get user data
            user_result = self.supabase.table('users').select(
                'id, email, tier, current_balance_usd'
            ).eq('id', key_data['user_id']).single().execute()
            
            if not user_result.data:
                raise HTTPException(
                    status_code=401,
                    detail="User not found"
                )
            
            user_data = user_result.data
            
            # Build user context
            user_context = {
                'user_id': user_data['id'],
                'email': user_data['email'],
                'tier': user_data['tier'],
                'rate_limit_per_minute': key_data['rate_limit_per_minute'],
                'rate_limit_per_day': key_data['rate_limit_per_day'],
                'current_balance_usd': user_data['current_balance_usd'],
                'api_key_hash': key_hash
            }
            
            # Update last_used timestamp
            self.supabase.table('api_keys').update({
                'last_used': 'now()'
            }).eq('key_hash', key_hash).execute()
            
            # Cache the result
            self.api_key_cache[api_key] = {
                'user_context': user_context,
                'cached_at': time.time()
            }
            
            return user_context
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"Auth error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Authentication service error"
            )
    
    async def create_api_key(self, user_id: str, tier: str = "free") -> str:
        """
        Create new API key for user
        
        Args:
            user_id: User ID
            tier: Tier (free, pro, enterprise)
            
        Returns:
            New API key (plaintext, only shown once)
        """
        import secrets
        
        # Generate secure random key
        api_key = f"pk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Set rate limits based on tier
        rate_limits = {
            'free': {'per_minute': 10, 'per_day': 1000},
            'pro': {'per_minute': 60, 'per_day': 10000},
            'enterprise': {'per_minute': 1000, 'per_day': 1000000}
        }
        
        limits = rate_limits.get(tier, rate_limits['free'])
        
        # Store in Supabase
        self.supabase.table('api_keys').insert({
            'key_hash': key_hash,
            'user_id': user_id,
            'tier': tier,
            'rate_limit_per_minute': limits['per_minute'],
            'rate_limit_per_day': limits['per_day'],
            'is_active': True
        }).execute()
        
        return api_key
    
    async def revoke_api_key(self, api_key: str):
        """Revoke API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        self.supabase.table('api_keys').update({
            'is_active': False
        }).eq('key_hash', key_hash).execute()
        
        # Clear from cache
        if api_key in self.api_key_cache:
            del self.api_key_cache[api_key]


# Singleton instance
_auth_service: Optional[AuthService] = None

def get_auth_service() -> AuthService:
    """Get or create AuthService singleton"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
