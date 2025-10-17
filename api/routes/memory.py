"""
Memory management endpoints
Query, add, delete memories
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from models.requests import MemoryQueryRequest, MemoryAddRequest, MemoryDeleteRequest
from models.responses import MemoryQueryResponse, MemoryAddResponse, MemoryResult
from api.dependencies import get_vector_db, get_supabase
from services.cache_service import get_cache_service
from utils.helpers import resolve_conversation_id
import time

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/query", response_model=MemoryQueryResponse)
async def query_memory(
    request: Request,
    query_request: MemoryQueryRequest,
    vector_db = Depends(get_vector_db),
    cache_service = Depends(get_cache_service)
):
    """
    Query conversation memory without generating response
    
    Returns relevant memories for the query
    """
    start_time = time.time()
    
    # Get user from auth middleware
    user = request.state.user
    
    # Resolve conversation ID
    conversation_id = resolve_conversation_id(query_request, user)
    
    try:
        # Check cache first
        cached_results = cache_service.get(conversation_id, query_request.query)
        
        if cached_results:
            query_time = (time.time() - start_time) * 1000
            
            return MemoryQueryResponse(
                success=True,
                results=[MemoryResult(**r) for r in cached_results],
                total_found=len(cached_results),
                query_time_ms=round(query_time, 2),
                conversation_id=conversation_id
            )
        
        # Query vector DB
        results = await vector_db.query(
            conversation_id=conversation_id,
            query_text=query_request.query,
            top_k=query_request.top_k
        )
        
        query_time = (time.time() - start_time) * 1000
        
        # Format results
        formatted_results = [
            MemoryResult(
                text=r['text'],
                similarity=r['similarity'],
                metadata=r.get('metadata', {}),
                timestamp=r.get('metadata', {}).get('timestamp')
            )
            for r in results
        ]
        
        # Cache results
        cache_service.set(
            conversation_id,
            query_request.query,
            [r.dict() for r in formatted_results]
        )
        
        return MemoryQueryResponse(
            success=True,
            results=formatted_results,
            total_found=len(formatted_results),
            query_time_ms=round(query_time, 2),
            conversation_id=conversation_id
        )
        
    except Exception as e:
        print(f"Memory query error: {e}")
        raise HTTPException(500, f"Memory query failed: {str(e)}")


@router.post("/add", response_model=MemoryAddResponse)
async def add_memory(
    request: Request,
    add_request: MemoryAddRequest,
    vector_db = Depends(get_vector_db),
    cache_service = Depends(get_cache_service)
):
    """
    Manually add memory to conversation
    
    Useful for importing existing conversation history
    """
    user = request.state.user
    
    # Resolve conversation ID
    conversation_id = resolve_conversation_id(add_request, user)
    
    try:
        # Add to vector DB
        success = vector_db.add(
            conversation_id=conversation_id,
            text=add_request.text,
            metadata=add_request.metadata
        )
        
        if success:
            # Invalidate cache for this conversation
            cache_service.invalidate(conversation_id)
            
            return MemoryAddResponse(
                success=True,
                message="Memory added successfully",
                conversation_id=conversation_id
            )
        else:
            raise HTTPException(500, "Failed to add memory")
            
    except Exception as e:
        print(f"Add memory error: {e}")
        raise HTTPException(500, f"Failed to add memory: {str(e)}")


@router.delete("/delete")
async def delete_conversation(
    request: Request,
    delete_request: MemoryDeleteRequest,
    vector_db = Depends(get_vector_db),
    cache_service = Depends(get_cache_service)
):
    """
    Delete all memories for a conversation
    
    This is permanent and cannot be undone
    """
    user = request.state.user
    
    # Resolve conversation ID
    conversation_id = resolve_conversation_id(delete_request, user)
    
    try:
        # Delete from vector DB
        success = vector_db.delete_conversation(conversation_id)
        
        if success:
            # Invalidate cache
            cache_service.invalidate(conversation_id)
            
            return {
                "success": True,
                "message": f"Deleted all memories for conversation {conversation_id}"
            }
        else:
            raise HTTPException(500, "Failed to delete conversation")
            
    except Exception as e:
        print(f"Delete conversation error: {e}")
        raise HTTPException(500, f"Failed to delete: {str(e)}")
