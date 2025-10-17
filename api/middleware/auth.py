"""
Authentication middleware
Validates API keys on every request
"""
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from services.auth_service import get_auth_service
import time

security = HTTPBearer()


class AuthMiddleware(BaseHTTPMiddleware):
    """Authenticate requests using Supabase API keys"""
    
    # Public endpoints (no auth required)
    PUBLIC_PATHS = {
        '/health',
        '/metrics',
        '/docs',
        '/openapi.json',
        '/redoc'
    }
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public endpoints
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        
        # Get auth service
        auth_service = get_auth_service()
        
        # Extract Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return HTTPException(401, "Missing Authorization header")
        
        if not auth_header.startswith('Bearer '):
            return HTTPException(401, "Invalid Authorization header format")
        
        # Validate API key
        try:
            from fastapi.security import HTTPAuthorizationCredentials
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=auth_header.replace('Bearer ', '')
            )
            
            user_context = await auth_service.validate_api_key(credentials)
            
            # Attach user context to request state
            request.state.user = user_context
            request.state.auth_time = time.time()
            
        except HTTPException as e:
            raise e
        except Exception as e:
            print(f"Auth middleware error: {e}")
            raise HTTPException(500, "Authentication error")
        
        # Continue to route
        response = await call_next(request)
        return response
