"""
Chat completions endpoint (OpenAI compatible)
Main endpoint for chat with infinite memory
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from models.requests import ChatCompletionRequest
from models.responses import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatChoice,
    ChatMessage,
    Usage,
    StreamChoice
)
from api.dependencies import (
    get_vllm_engine,
    get_memory_manager,
    get_supabase
)
from services.billing_service import get_billing_service
from services.cache_service import get_cache_service
from api.routes.health import record_request_metrics
from utils.helpers import resolve_conversation_id, count_tokens
import time
import uuid
import json

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request,
    chat_request: ChatCompletionRequest,
    memory_manager = Depends(get_memory_manager),
    vllm_engine = Depends(get_vllm_engine),
    billing_service = Depends(get_billing_service),
    cache_service = Depends(get_cache_service),
    supabase = Depends(get_supabase)
):
    """
    Chat completions with infinite memory (OpenAI compatible)
    
    Retrieves relevant context from conversation history,
    generates response, and stores the exchange
    """
    start_time = time.time()
    
    # Get user from auth middleware
    user = request.state.user
    
    # Resolve conversation ID
    conversation_id = resolve_conversation_id(chat_request, user)
    
    # Extract user message
    user_message = chat_request.messages[-1].content
    
    try:
        # Step 1: Retrieve relevant memories (if enabled)
        retrieved_context = []
        retrieval_latency = 0
        
        if chat_request.use_memory:
            retrieval_start = time.time()
            
            # Check cache
            cached_context = cache_service.get(conversation_id, user_message)
            
            if cached_context:
                retrieved_context = cached_context
            else:
                # Query memory
                memory_results = await memory_manager.retrieve_context(
                    conversation_id=conversation_id,
                    query=user_message,
                    top_k=chat_request.memory_top_k
                )
                
                if memory_results['success']:
                    retrieved_context = memory_results['results']
                    
                    # Cache results
                    cache_service.set(
                        conversation_id,
                        user_message,
                        retrieved_context
                    )
            
            retrieval_latency = (time.time() - retrieval_start) * 1000
        
        # Step 2: Build prompt with context
        messages = chat_request.messages.copy()
        
        if retrieved_context:
            # Inject context before user message
            context_text = "\n\n".join([
                f"[Previous context {i+1}]: {r['text']}"
                for i, r in enumerate(retrieved_context[:3])
            ])
            
            context_message = {
                "role": "system",
                "content": f"Relevant context from conversation history:\n{context_text}"
            }
            
            # Insert before last message
            messages.insert(-1, context_message)
        
        # Step 3: Generate response
        generation_start = time.time()
        
        # Format for vLLM
        prompt_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in messages
        ])
        
        # Generate
        outputs = vllm_engine.generate(
            [prompt_text],
            sampling_params={
                'max_tokens': chat_request.max_tokens,
                'temperature': chat_request.temperature,
                'top_p': chat_request.top_p
            }
        )
        
        generated_text = outputs[0].outputs[0].text.strip()
        generation_latency = (time.time() - generation_start) * 1000
        
        # Step 4: Store exchange in memory
        if chat_request.use_memory:
            # Store user message
            memory_manager.vector_db.add(
                conversation_id=conversation_id,
                text=f"User: {user_message}",
                metadata={'role': 'user', 'timestamp': time.time()}
            )
            
            # Store assistant response
            memory_manager.vector_db.add(
                conversation_id=conversation_id,
                text=f"Assistant: {generated_text}",
                metadata={'role': 'assistant', 'timestamp': time.time()}
            )
            
            # Invalidate cache
            cache_service.invalidate(conversation_id)
        
        # Step 5: Calculate usage
        input_tokens = count_tokens(prompt_text)
        output_tokens = count_tokens(generated_text)
        total_tokens = input_tokens + output_tokens
        
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens
        )
        
        # Step 6: Track billing
        await billing_service.track_usage(
            user_id=user['user_id'],
            conversation_id=conversation_id,
            usage={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'retrieval_calls': 1 if chat_request.use_memory else 0,
                'retrieval_latency_ms': retrieval_latency,
                'model': chat_request.model,
                'endpoint': '/v1/chat/completions',
                'status_code': 200,
                'latency_ms': (time.time() - start_time) * 1000
            },
            supabase_client=supabase
        )
        
        # Step 7: Record metrics
        total_latency = (time.time() - start_time) * 1000
        record_request_metrics(total_tokens, total_latency)
        
        # Step 8: Build response
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        return ChatCompletionResponse(
            id=response_id,
            created=int(time.time()),
            model=chat_request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=generated_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=usage,
            perpetual_metadata={
                'conversation_id': conversation_id,
                'retrieval_latency_ms': round(retrieval_latency, 2),
                'generation_latency_ms': round(generation_latency, 2),
                'total_latency_ms': round(total_latency, 2),
                'memories_used': len(retrieved_context),
                'cached': len(retrieved_context) > 0 and retrieval_latency < 10
            }
        )
        
    except Exception as e:
        print(f"Chat completion error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Chat completion failed: {str(e)}")
