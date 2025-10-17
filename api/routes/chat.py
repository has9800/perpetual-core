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
from services.llm_proxy_service import get_llm_proxy_service
from api.routes.health import record_request_metrics
from utils.helpers import resolve_conversation_id, count_tokens
import time
import uuid
import json
import asyncio

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request,
    chat_request: ChatCompletionRequest,
    memory_manager = Depends(get_memory_manager),
    billing_service = Depends(get_billing_service),
    llm_proxy_service = Depends(get_llm_proxy_service),
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
        memory_settings = chat_request.get_memory_settings()

        if chat_request.use_memory and memory_settings['semantic_top_k'] > 0:
            retrieval_start = time.time()

            # Query memory with configurable top_k
            memory_results = await memory_manager.retrieve_context(
                conversation_id=conversation_id,
                query=user_message,
                top_k=memory_settings['semantic_top_k']
            )

            if memory_results['success']:
                retrieved_context = memory_results['results']

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

        # Convert Pydantic models to dicts for API forwarding
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Step 3: Forward to external LLM API
        generation_start = time.time()

        # Forward to appropriate provider (OpenAI, Anthropic, xAI, Together, etc.)
        llm_response = await llm_proxy_service.forward_request(
            messages=messages_dict,
            model=chat_request.model,
            user_id=user['user_id'],
            supabase=supabase,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            stream=False
        )

        generated_text = llm_response["choices"][0]["message"]["content"]
        generation_latency = (time.time() - generation_start) * 1000

        # Step 4: Store exchange in memory (non-blocking background task)
        if chat_request.use_memory:
            # Create background task for storing (doesn't block response)
            async def store_memory_background():
                try:
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
                except Exception as e:
                    print(f"Background memory storage error: {e}")

            # Fire and forget (non-blocking)
            asyncio.create_task(store_memory_background())
        
        # Step 5: Extract usage from LLM response
        usage_data = llm_response.get("usage", {})
        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", input_tokens + output_tokens)

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


@router.post("/completions/stream")
async def chat_completions_stream(
    request: Request,
    chat_request: ChatCompletionRequest,
    memory_manager = Depends(get_memory_manager),
    billing_service = Depends(get_billing_service),
    llm_proxy_service = Depends(get_llm_proxy_service),
    supabase = Depends(get_supabase)
):
    """
    Chat completions with streaming (OpenAI compatible)

    Returns Server-Sent Events (SSE) stream
    """
    # Get user from auth middleware
    user = request.state.user

    # Resolve conversation ID
    conversation_id = resolve_conversation_id(chat_request, user)

    # Extract user message
    user_message = chat_request.messages[-1].content

    async def generate_stream():
        """Generator for SSE streaming"""
        try:
            # Step 1: Retrieve relevant memories (if enabled)
            retrieved_context = []
            retrieval_latency = 0
            memory_settings = chat_request.get_memory_settings()

            if chat_request.use_memory and memory_settings['semantic_top_k'] > 0:
                retrieval_start = time.time()

                # Query memory with configurable top_k
                memory_results = await memory_manager.retrieve_context(
                    conversation_id=conversation_id,
                    query=user_message,
                    top_k=memory_settings['semantic_top_k']
                )

                if memory_results['success']:
                    retrieved_context = memory_results['results']

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

            # Convert Pydantic models to dicts for API forwarding
            messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Step 3: Stream from external LLM API
            # Route to appropriate streaming method
            if chat_request.model.startswith("gpt-") or chat_request.model.startswith("o1-"):
                # OpenAI streaming
                full_response = ""
                async for line in llm_proxy_service.forward_to_openai_stream(
                    messages=messages_dict,
                    model=chat_request.model,
                    max_tokens=chat_request.max_tokens,
                    temperature=chat_request.temperature,
                    top_p=chat_request.top_p
                ):
                    # Forward SSE chunks directly to client
                    yield line

                    # Parse to collect full response for storage
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        try:
                            chunk_data = json.loads(line[6:])
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                full_response += delta["content"]
                        except:
                            pass

                # Step 4: Store exchange in memory
                if chat_request.use_memory and full_response:
                    # Store user message
                    memory_manager.vector_db.add(
                        conversation_id=conversation_id,
                        text=f"User: {user_message}",
                        metadata={'role': 'user', 'timestamp': time.time()}
                    )

                    # Store assistant response
                    memory_manager.vector_db.add(
                        conversation_id=conversation_id,
                        text=f"Assistant: {full_response}",
                        metadata={'role': 'assistant', 'timestamp': time.time()}
                    )

                # Note: For streaming, we can't track exact token usage in real-time
                # Would need to implement token counting on collected response

            else:
                # For non-OpenAI models, fall back to non-streaming
                # (Anthropic streaming requires different handling)
                yield "data: {\"error\": \"Streaming not yet supported for this model\"}\n\n"
                yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"Streaming error: {e}")
            import traceback
            traceback.print_exc()
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
