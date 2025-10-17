"""
Request/response logging middleware
Logs all requests for monitoring and debugging
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import json


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses"""
    
    async def dispatch(self, request: Request, call_next):
        # Start timer
        start_time = time.time()
        
        # Get user from state (if authenticated)
        user = getattr(request.state, 'user', None)
        user_id = user['user_id'] if user else 'anonymous'
        
        # Log request
        print(f"[{request.method}] {request.url.path} - User: {user_id}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log response
            print(f"[{request.method}] {request.url.path} - {response.status_code} - {latency_ms:.0f}ms")
            
            # Add custom headers
            response.headers['X-Request-ID'] = str(id(request))
            response.headers['X-Process-Time'] = f"{latency_ms:.2f}ms"
            
            # Store metrics in request state for billing
            request.state.latency_ms = latency_ms
            request.state.status_code = response.status_code
            
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"[{request.method}] {request.url.path} - ERROR: {str(e)} - {latency_ms:.0f}ms")
            raise
