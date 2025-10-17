"""
Rate limiting middleware
Enforces per-user rate limits from Supabase
"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import redis
import time
from config.settings import get_settings

settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit requests per user"""
    
    def __init__(self, app):
        super().__init__(app)
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for public endpoints
        if request.url.path in {'/health', '/metrics', '/docs', '/openapi.json'}:
            return await call_next(request)
        
        # Get user context from auth middleware
        user = getattr(request.state, 'user', None)
        
        if not user:
            # Auth middleware should have caught this
            return await call_next(request)
        
        user_id = user['user_id']
        
        # Check rate limits
        try:
            # Per-minute limit
            minute_key = f"rate_limit:minute:{user_id}:{int(time.time() / 60)}"
            minute_count = self.redis_client.incr(minute_key)
            
            if minute_count == 1:
                self.redis_client.expire(minute_key, 60)
            
            if minute_count > user['rate_limit_per_minute']:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {user['rate_limit_per_minute']} requests per minute"
                )
            
            # Per-day limit
            day_key = f"rate_limit:day:{user_id}:{int(time.time() / 86400)}"
            day_count = self.redis_client.incr(day_key)
            
            if day_count == 1:
                self.redis_client.expire(day_key, 86400)
            
            if day_count > user['rate_limit_per_day']:
                raise HTTPException(
                    status_code=429,
                    detail=f"Daily rate limit exceeded: {user['rate_limit_per_day']} requests per day"
                )
            
            # Add rate limit headers to response
            response = await call_next(request)
            response.headers['X-RateLimit-Limit-Minute'] = str(user['rate_limit_per_minute'])
            response.headers['X-RateLimit-Remaining-Minute'] = str(
                user['rate_limit_per_minute'] - minute_count
            )
            response.headers['X-RateLimit-Limit-Day'] = str(user['rate_limit_per_day'])
            response.headers['X-RateLimit-Remaining-Day'] = str(
                user['rate_limit_per_day'] - day_count
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"Rate limit error: {e}")
            # Don't block request on rate limit errors
            return await call_next(request)
