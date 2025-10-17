"""
Quality A/B Benchmark for Perpetual AI Memory System

Tests retrieval quality vs full context across different scenarios:
1. Multi-hop reasoning (A→B→C dependencies)
2. Context-dependent queries (references to earlier info)
3. Code understanding (cross-file dependencies)
4. Long-form content generation (coherent narratives)

Runs locally on Vast.ai GPU with Mistral-7B quantized model
No API keys required - pure vLLM inference
"""
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_wrapper import VLLMEngine
from core.vector_db import QdrantAdapter
from core.enhanced_memory_manager import EnhancedMemoryManager
from services.token_tracker import TokenTracker
from difflib import SequenceMatcher
import numpy as np


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class QualityBenchmark:
    """A/B quality testing for memory retrieval"""

    def __init__(
        self,
        model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        qdrant_url: str = None,
        qdrant_api_key: str = None
    ):
        """
        Initialize with vLLM and Qdrant

        Args:
            model_name: vLLM model to use
            qdrant_url: Qdrant URL (env: QDRANT_URL or QDRANT_CLOUD_URL)
            qdrant_api_key: Qdrant API key (env: QDRANT_API_KEY) - optional for local
        """
        print(f"{Colors.CYAN}Initializing Quality Benchmark...{Colors.END}")

        # Initialize vLLM engine
        print(f"Loading model: {model_name}")
        self.llm = VLLMEngine(
            model_name=model_name,
            quantization="gptq",
            gpu_memory_utilization=0.9,
            max_model_len=4096
        )

        # Initialize Qdrant adapter
        # Support env vars or passed params
        import os
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLOUD_URL") or "http://localhost:6333"
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        print(f"Connecting to Qdrant at: {qdrant_url}")
        self.vector_db = QdrantAdapter(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name="quality_benchmark"
        )

        # Initialize token tracker with mock Redis (optional for benchmarking)
        try:
            import redis
            redis_client = redis.from_url("redis://localhost:6379/0", decode_responses=True)
            redis_client.ping()
            self.token_tracker = TokenTracker(redis_client=redis_client)
            print("✓ Connected to Redis for token tracking")
        except Exception as e:
            print(f"⚠️  Redis not available, running without token tracking: {e}")
            self.token_tracker = None

        # Initialize enhanced memory manager
        self.memory_manager = EnhancedMemoryManager(
            vector_db=self.vector_db,
            token_tracker=self.token_tracker,
            cache_capacity=1000
        )

        print(f"{Colors.GREEN}✓ Initialization complete{Colors.END}\n")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (0-1)"""
        return SequenceMatcher(None, text1, text2).ratio()

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        # Use Qdrant's embedding model
        emb1 = self.vector_db.model.encode([text1])[0]
        emb2 = self.vector_db.model.encode([text2])[0]

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        return float(dot_product / (norm1 * norm2))

    async def run_conversation(
        self,
        conversation: List[Dict[str, str]],
        test_query: Dict[str, str],
        use_retrieval: bool = True,
        memory_config: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """
        Run a conversation and test response quality

        Args:
            conversation: List of {"role": "user/assistant", "content": "..."}
            test_query: {"question": "...", "context_needed": ["turn 3", "turn 7"]}
            use_retrieval: If True, use retrieval; if False, use full context
            memory_config: {"recent_turns": 15, "semantic_top_k": 3}
        """
        if memory_config is None:
            memory_config = {"recent_turns": 15, "semantic_top_k": 3}

        conversation_id = f"test_{'retrieval' if use_retrieval else 'full'}_{int(time.time())}"

        # Step 1: Build conversation history in memory
        for i, turn in enumerate(conversation):
            if turn['role'] == 'user':
                # Store user message
                self.memory_manager.vector_db.add(
                    conversation_id=conversation_id,
                    text=f"User: {turn['content']}",
                    metadata={'role': 'user', 'turn': i, 'timestamp': time.time()}
                )
            elif turn['role'] == 'assistant':
                # Store assistant response
                self.memory_manager.vector_db.add(
                    conversation_id=conversation_id,
                    text=f"Assistant: {turn['content']}",
                    metadata={'role': 'assistant', 'turn': i, 'timestamp': time.time()}
                )

        # Step 2: Generate response to test query
        query = test_query['question']

        if use_retrieval:
            # Use retrieval (recent_turns + semantic_top_k)
            retrieval_start = time.time()

            memory_results = await self.memory_manager.retrieve_context(
                conversation_id=conversation_id,
                query=query,
                top_k=memory_config['semantic_top_k']
            )

            retrieval_latency = (time.time() - retrieval_start) * 1000

            retrieved_context = memory_results.get('results', [])

            # Build prompt with retrieved context
            context_text = "\n\n".join([
                f"[Previous context {i+1}]: {r['text']}"
                for i, r in enumerate(retrieved_context[:memory_config['semantic_top_k']])
            ])

            # Add recent turns (simulate sliding window)
            recent_turns = conversation[-memory_config['recent_turns']:]
            recent_text = "\n".join([
                f"{turn['role'].capitalize()}: {turn['content']}"
                for turn in recent_turns
            ])

            full_prompt = f"""Relevant context from conversation history:
{context_text}

Recent conversation:
{recent_text}

User: {query}
Assistant:"""
        else:
            # Use full context (baseline)
            retrieval_latency = 0
            retrieved_context = []

            # Include ALL conversation history
            full_prompt = "\n".join([
                f"{turn['role'].capitalize()}: {turn['content']}"
                for turn in conversation
            ])
            full_prompt += f"\nUser: {query}\nAssistant:"

        # Step 3: Generate response
        generation_start = time.time()

        outputs = self.llm.generate(
            prompts=[full_prompt],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "\n\n\n"]
        )

        response = outputs[0].outputs[0].text.strip()
        generation_latency = (time.time() - generation_start) * 1000

        # Step 4: Count tokens
        prompt_tokens = len(full_prompt.split())  # Rough approximation
        response_tokens = len(response.split())

        return {
            'response': response,
            'retrieval_latency_ms': retrieval_latency,
            'generation_latency_ms': generation_latency,
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
            'total_tokens': prompt_tokens + response_tokens,
            'retrieved_context_count': len(retrieved_context),
            'method': 'retrieval' if use_retrieval else 'full_context'
        }

    async def test_multi_hop_reasoning(self) -> Dict[str, Any]:
        """Test: Multi-hop reasoning (A→B→C dependencies)"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}Test 1: Multi-Hop Reasoning{Colors.END}")
        print("Scenario: Information spans multiple turns (A→B→C)\n")

        conversation = [
            {"role": "user", "content": "I'm planning a trip to Japan in March."},
            {"role": "assistant", "content": "Great! March is cherry blossom season in Japan. Which cities are you considering?"},
            {"role": "user", "content": "I'm thinking Tokyo and Kyoto."},
            {"role": "assistant", "content": "Excellent choices! Tokyo offers modern attractions while Kyoto has traditional temples."},
            {"role": "user", "content": "I love traditional architecture. How many days should I spend in Kyoto?"},
            {"role": "assistant", "content": "For traditional architecture enthusiasts, I'd recommend 3-4 days in Kyoto to see temples like Kinkaku-ji and Fushimi Inari."},
            {"role": "user", "content": "What about budget? I'm a student."},
            {"role": "assistant", "content": "As a student, budget around $80-100/day for accommodation, food, and local transport. Consider staying in hostels."},
            {"role": "user", "content": "Are there student discounts?"},
            {"role": "assistant", "content": "Yes! Many temples and museums offer student discounts. Bring your student ID."},
            {"role": "user", "content": "What should I pack for the weather?"},
            {"role": "assistant", "content": "March weather in Japan is mild (10-15°C). Pack layers, a light jacket, and comfortable walking shoes."},
        ]

        test_query = {
            "question": "Based on everything we discussed, create a 5-day itinerary for my trip that fits my budget and interests.",
            "context_needed": ["Tokyo", "Kyoto", "traditional architecture", "student budget", "March weather", "3-4 days Kyoto"]
        }

        # Run with retrieval
        print(f"{Colors.CYAN}Running with retrieval (15 recent + 3 semantic)...{Colors.END}")
        result_retrieval = await self.run_conversation(
            conversation,
            test_query,
            use_retrieval=True,
            memory_config={"recent_turns": 15, "semantic_top_k": 3}
        )

        # Run with full context
        print(f"{Colors.CYAN}Running with full context (baseline)...{Colors.END}")
        result_full = await self.run_conversation(
            conversation,
            test_query,
            use_retrieval=False
        )

        # Compare
        similarity = self.calculate_similarity(result_retrieval['response'], result_full['response'])
        semantic_similarity = self.calculate_semantic_similarity(result_retrieval['response'], result_full['response'])

        print(f"\n{Colors.GREEN}Results:{Colors.END}")
        print(f"  Retrieval tokens: {result_retrieval['total_tokens']}")
        print(f"  Full context tokens: {result_full['total_tokens']}")
        print(f"  Token savings: {Colors.BOLD}{100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens']):.1f}%{Colors.END}")
        print(f"  Text similarity: {Colors.BOLD}{similarity:.2%}{Colors.END}")
        print(f"  Semantic similarity: {Colors.BOLD}{semantic_similarity:.2%}{Colors.END}")

        return {
            'test_name': 'multi_hop_reasoning',
            'retrieval': result_retrieval,
            'full_context': result_full,
            'text_similarity': similarity,
            'semantic_similarity': semantic_similarity,
            'token_savings_pct': 100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens'])
        }

    async def test_context_dependent_queries(self) -> Dict[str, Any]:
        """Test: Context-dependent queries (references to earlier info)"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}Test 2: Context-Dependent Queries{Colors.END}")
        print("Scenario: Query references specific earlier information\n")

        conversation = [
            {"role": "user", "content": "I'm building a FastAPI app with authentication."},
            {"role": "assistant", "content": "Great! Are you using JWT tokens or session-based auth?"},
            {"role": "user", "content": "JWT tokens with Supabase."},
            {"role": "assistant", "content": "Perfect. Supabase provides JWT verification out of the box. You'll need to validate tokens in middleware."},
            {"role": "user", "content": "How do I implement the middleware?"},
            {"role": "assistant", "content": "Create a dependency that extracts the Authorization header, verifies the JWT with Supabase, and injects the user into request.state."},
            {"role": "user", "content": "What about rate limiting?"},
            {"role": "assistant", "content": "Use Redis with a sliding window algorithm. Store request counts per user_id with TTL."},
            {"role": "user", "content": "Should I use Redis for caching too?"},
            {"role": "assistant", "content": "Depends on your use case. Redis is great for caching API responses, but adds complexity. Start without it."},
            {"role": "user", "content": "What database should I use?"},
            {"role": "assistant", "content": "Since you're using Supabase, you already have PostgreSQL. Use it for your main data."},
            {"role": "user", "content": "How do I handle file uploads?"},
            {"role": "assistant", "content": "Supabase Storage is perfect for this. Upload files to buckets and store URLs in PostgreSQL."},
        ]

        test_query = {
            "question": "Remind me what authentication method I chose and why it works well with my database choice.",
            "context_needed": ["JWT tokens with Supabase", "PostgreSQL"]
        }

        # Run with retrieval
        print(f"{Colors.CYAN}Running with retrieval (15 recent + 3 semantic)...{Colors.END}")
        result_retrieval = await self.run_conversation(
            conversation,
            test_query,
            use_retrieval=True
        )

        # Run with full context
        print(f"{Colors.CYAN}Running with full context (baseline)...{Colors.END}")
        result_full = await self.run_conversation(
            conversation,
            test_query,
            use_retrieval=False
        )

        # Compare
        similarity = self.calculate_similarity(result_retrieval['response'], result_full['response'])
        semantic_similarity = self.calculate_semantic_similarity(result_retrieval['response'], result_full['response'])

        print(f"\n{Colors.GREEN}Results:{Colors.END}")
        print(f"  Retrieval tokens: {result_retrieval['total_tokens']}")
        print(f"  Full context tokens: {result_full['total_tokens']}")
        print(f"  Token savings: {Colors.BOLD}{100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens']):.1f}%{Colors.END}")
        print(f"  Text similarity: {Colors.BOLD}{similarity:.2%}{Colors.END}")
        print(f"  Semantic similarity: {Colors.BOLD}{semantic_similarity:.2%}{Colors.END}")

        return {
            'test_name': 'context_dependent_queries',
            'retrieval': result_retrieval,
            'full_context': result_full,
            'text_similarity': similarity,
            'semantic_similarity': semantic_similarity,
            'token_savings_pct': 100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens'])
        }

    async def test_code_understanding(self) -> Dict[str, Any]:
        """Test: Code understanding (cross-file dependencies)"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}Test 3: Code Understanding{Colors.END}")
        print("Scenario: Code discussion with cross-file references\n")

        conversation = [
            {"role": "user", "content": "I have a bug in my API. The /chat endpoint returns 500 errors."},
            {"role": "assistant", "content": "Let's debug this. What does the error message say?"},
            {"role": "user", "content": "It says 'NoneType object has no attribute get' in llm_proxy_service.py line 47."},
            {"role": "assistant", "content": "That suggests the provider_config is None. Check if detect_provider() is finding the model."},
            {"role": "user", "content": "Oh I see. The model name is 'gpt-4-custom' which isn't in the registry."},
            {"role": "assistant", "content": "That's the issue. Add 'gpt-4-custom' to the OpenAI provider's models list in provider_config.py."},
            {"role": "user", "content": "Fixed! But now I'm getting 'API key not found' errors."},
            {"role": "assistant", "content": "Check if the user has stored their OpenAI API key. Use GET /v1/provider-keys/list to verify."},
            {"role": "user", "content": "They have the key stored. It's encrypted in the database."},
            {"role": "assistant", "content": "Then the issue is likely in api_key_service.py decrypt_api_key(). Check if the encryption key env var is set."},
            {"role": "user", "content": "API_KEY_ENCRYPTION_KEY is set. Let me check the decryption logic."},
            {"role": "assistant", "content": "Make sure you're using the same Fernet key for encryption and decryption. Any mismatch causes InvalidToken errors."},
        ]

        test_query = {
            "question": "Summarize the debugging steps we took to fix the /chat endpoint, including which files we modified.",
            "context_needed": ["llm_proxy_service.py", "provider_config.py", "api_key_service.py", "detect_provider", "encryption"]
        }

        # Run with retrieval
        print(f"{Colors.CYAN}Running with retrieval (15 recent + 3 semantic)...{Colors.END}")
        result_retrieval = await self.run_conversation(
            conversation,
            test_query,
            use_retrieval=True
        )

        # Run with full context
        print(f"{Colors.CYAN}Running with full context (baseline)...{Colors.END}")
        result_full = await self.run_conversation(
            conversation,
            test_query,
            use_retrieval=False
        )

        # Compare
        similarity = self.calculate_similarity(result_retrieval['response'], result_full['response'])
        semantic_similarity = self.calculate_semantic_similarity(result_retrieval['response'], result_full['response'])

        print(f"\n{Colors.GREEN}Results:{Colors.END}")
        print(f"  Retrieval tokens: {result_retrieval['total_tokens']}")
        print(f"  Full context tokens: {result_full['total_tokens']}")
        print(f"  Token savings: {Colors.BOLD}{100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens']):.1f}%{Colors.END}")
        print(f"  Text similarity: {Colors.BOLD}{similarity:.2%}{Colors.END}")
        print(f"  Semantic similarity: {Colors.BOLD}{semantic_similarity:.2%}{Colors.END}")

        return {
            'test_name': 'code_understanding',
            'retrieval': result_retrieval,
            'full_context': result_full,
            'text_similarity': similarity,
            'semantic_similarity': semantic_similarity,
            'token_savings_pct': 100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens'])
        }

    async def test_long_form_generation(self) -> Dict[str, Any]:
        """Test: Long-form content generation (coherent narratives)"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}Test 4: Long-Form Content Generation{Colors.END}")
        print("Scenario: Generate coherent content based on scattered info\n")

        conversation = [
            {"role": "user", "content": "I need to write a blog post about AI memory systems."},
            {"role": "assistant", "content": "Great topic! What angle are you taking?"},
            {"role": "user", "content": "I want to focus on cost savings. Like how sending full context is expensive."},
            {"role": "assistant", "content": "Good angle. You could compare traditional approaches (sending entire history) vs retrieval-based systems."},
            {"role": "user", "content": "What are the key benefits of retrieval?"},
            {"role": "assistant", "content": "Main benefits: 1) Constant token usage per request, 2) Infinite conversation length, 3) 99% cost reduction."},
            {"role": "user", "content": "Are there any downsides?"},
            {"role": "assistant", "content": "Yes: 1) Slight quality degradation for complex reasoning, 2) Retrieval latency (20-50ms), 3) Need for vector database."},
            {"role": "user", "content": "Who is this most useful for?"},
            {"role": "assistant", "content": "Best for: customer support, personal assistants, long-running projects. Less ideal for: creative writing, complex analysis."},
            {"role": "user", "content": "What about implementation complexity?"},
            {"role": "assistant", "content": "Moderate complexity. Need: vector DB (Qdrant), embeddings model, hybrid search (dense+sparse), sliding window logic."},
            {"role": "user", "content": "Can you give me a hook for the intro?"},
            {"role": "assistant", "content": "How about: 'Stop paying for massive context windows. Here's how semantic memory reduces LLM costs by 99%.'"},
        ]

        test_query = {
            "question": "Write a 3-paragraph blog post intro covering the problem, solution, and who it's for. Use the information we discussed.",
            "context_needed": ["cost savings", "99% reduction", "benefits", "downsides", "use cases", "hook"]
        }

        # Run with retrieval
        print(f"{Colors.CYAN}Running with retrieval (15 recent + 3 semantic)...{Colors.END}")
        result_retrieval = await self.run_conversation(
            conversation,
            test_query,
            use_retrieval=True
        )

        # Run with full context
        print(f"{Colors.CYAN}Running with full context (baseline)...{Colors.END}")
        result_full = await self.run_conversation(
            conversation,
            test_query,
            use_retrieval=False
        )

        # Compare
        similarity = self.calculate_similarity(result_retrieval['response'], result_full['response'])
        semantic_similarity = self.calculate_semantic_similarity(result_retrieval['response'], result_full['response'])

        print(f"\n{Colors.GREEN}Results:{Colors.END}")
        print(f"  Retrieval tokens: {result_retrieval['total_tokens']}")
        print(f"  Full context tokens: {result_full['total_tokens']}")
        print(f"  Token savings: {Colors.BOLD}{100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens']):.1f}%{Colors.END}")
        print(f"  Text similarity: {Colors.BOLD}{similarity:.2%}{Colors.END}")
        print(f"  Semantic similarity: {Colors.BOLD}{semantic_similarity:.2%}{Colors.END}")

        return {
            'test_name': 'long_form_generation',
            'retrieval': result_retrieval,
            'full_context': result_full,
            'text_similarity': similarity,
            'semantic_similarity': semantic_similarity,
            'token_savings_pct': 100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens'])
        }

    def generate_ui_building_conversation(self, num_turns: int) -> List[Dict[str, str]]:
        """
        Generate realistic UI building conversation (Lovable-like)

        Args:
            num_turns: Number of conversation turns to generate

        Returns:
            List of conversation turns
        """
        conversation = []

        # Phase 1: Initial requirements
        conversation.extend([
            {"role": "user", "content": "I want to build a landing page for my SaaS product. It's a project management tool called TaskFlow."},
            {"role": "assistant", "content": "Great! I'll help you build a landing page for TaskFlow. What key features do you want to highlight?"},
            {"role": "user", "content": "The main features are: task boards, time tracking, team collaboration, and reporting."},
            {"role": "assistant", "content": "Perfect. Let's start with a hero section that emphasizes efficiency and team collaboration."},
        ])

        # Phase 2: Hero section
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "I want a clean, modern hero with a gradient background."},
                {"role": "assistant", "content": "I'll create a hero section with a blue-purple gradient, bold headline, subtitle, and CTA button."},
                {"role": "user", "content": "Can you add an illustration or screenshot?"},
                {"role": "assistant", "content": "Added a product screenshot on the right side with subtle shadow and rounded corners."},
            ])

        # Phase 3: Features section
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "Now add a features section with the 4 features I mentioned."},
                {"role": "assistant", "content": "Created a 2x2 grid with icons for each feature: task boards, time tracking, collaboration, and reporting."},
                {"role": "user", "content": "Can you make the icons animated on hover?"},
                {"role": "assistant", "content": "Added scale-up hover effect with smooth transitions for all feature icons."},
            ])

        # Phase 4: Pricing section
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "Add a pricing section. We have 3 tiers: Starter ($9), Pro ($29), Enterprise (custom)."},
                {"role": "assistant", "content": "Created pricing cards with all tiers. Pro tier is highlighted as 'Most Popular'."},
                {"role": "user", "content": "Can you add feature lists for each tier?"},
                {"role": "assistant", "content": "Added feature comparisons: Starter (5 projects, 10 users), Pro (unlimited projects, 50 users), Enterprise (everything + SSO)."},
            ])

        # Phase 5: Testimonials
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "Add a testimonials section with customer quotes."},
                {"role": "assistant", "content": "Created testimonial carousel with 3 customer quotes, photos, and company names."},
                {"role": "user", "content": "Make it auto-scroll every 5 seconds."},
                {"role": "assistant", "content": "Added auto-scroll with pause-on-hover. Users can also navigate with arrow buttons."},
            ])

        # Phase 6: Footer
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "Add a footer with social links and newsletter signup."},
                {"role": "assistant", "content": "Created footer with 4 columns: Product, Company, Resources, and Newsletter form."},
                {"role": "user", "content": "Can you add LinkedIn, Twitter, and GitHub icons?"},
                {"role": "assistant", "content": "Added social media icons with hover effects and external links."},
            ])

        # Phase 7: Mobile responsive
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "Make sure it's mobile responsive."},
                {"role": "assistant", "content": "Added responsive breakpoints. Features grid becomes 1 column, hero stacks vertically on mobile."},
                {"role": "user", "content": "The pricing cards are too cramped on mobile."},
                {"role": "assistant", "content": "Made pricing cards full-width on mobile with horizontal scroll for easy comparison."},
            ])

        # Phase 8: SEO and meta
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "Add SEO meta tags and Open Graph tags."},
                {"role": "assistant", "content": "Added title, description, OG tags, and Twitter Card meta for social sharing."},
                {"role": "user", "content": "What should the meta description be?"},
                {"role": "assistant", "content": "Suggested: 'TaskFlow - Modern project management with task boards, time tracking, and team collaboration. Start free today.'"},
            ])

        # Phase 9: Analytics
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "Add Google Analytics tracking."},
                {"role": "assistant", "content": "Added GA4 script in <head> with your tracking ID. Also added event tracking for CTA clicks."},
                {"role": "user", "content": "Track pricing card clicks too."},
                {"role": "assistant", "content": "Added click event tracking for all pricing CTA buttons with plan name as parameter."},
            ])

        # Phase 10: Forms and validation
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "The newsletter form needs validation."},
                {"role": "assistant", "content": "Added email validation with regex and error messages. Disabled submit until valid."},
                {"role": "user", "content": "Show a success message after signup."},
                {"role": "assistant", "content": "Added toast notification on successful signup with 'Check your email' message."},
            ])

        # Phase 11: Performance
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "The page loads slowly. Can you optimize images?"},
                {"role": "assistant", "content": "Converted images to WebP format and added lazy loading. Page load reduced from 3.2s to 1.1s."},
                {"role": "user", "content": "What about the fonts?"},
                {"role": "assistant", "content": "Using font-display: swap for Google Fonts to prevent flash of invisible text (FOIT)."},
            ])

        # Phase 12: Accessibility
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "Make it accessible for screen readers."},
                {"role": "assistant", "content": "Added ARIA labels, alt text for images, and semantic HTML. Keyboard navigation works."},
                {"role": "user", "content": "Test it with WAVE accessibility checker."},
                {"role": "assistant", "content": "WAVE shows 0 errors. Added skip-to-content link and increased color contrast ratios."},
            ])

        # Phase 13: Final tweaks
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "The CTA button color doesn't pop enough."},
                {"role": "assistant", "content": "Changed CTA from blue to bright orange (#FF6B35) with higher contrast. Conversion rate improved."},
                {"role": "user", "content": "Add a subtle animation when scrolling into view."},
                {"role": "assistant", "content": "Added fade-in-up animations using Intersection Observer for features and pricing sections."},
            ])

        # Phase 14: Deployment
        if len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "How do I deploy this to Vercel?"},
                {"role": "assistant", "content": "Connect GitHub repo to Vercel. It auto-deploys on push to main. Custom domain supported."},
                {"role": "user", "content": "Set up automatic deployments for pull requests."},
                {"role": "assistant", "content": "Enabled preview deployments. Each PR gets a unique URL for testing before merge."},
            ])

        # Additional padding turns if needed
        while len(conversation) < num_turns:
            conversation.extend([
                {"role": "user", "content": "Can you add a FAQ section?"},
                {"role": "assistant", "content": "Added collapsible FAQ with 6 common questions about pricing, features, and support."},
                {"role": "user", "content": "Make the FAQ items expand smoothly."},
                {"role": "assistant", "content": "Added CSS transitions for smooth expand/collapse with max-height animation."},
            ])

        return conversation[:num_turns]

    async def test_50_turn_conversation(self) -> Dict[str, Any]:
        """Test: 50-turn UI building conversation"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}Test 5: 50-Turn UI Building Conversation{Colors.END}")
        print("Scenario: Long conversation building a SaaS landing page (Lovable-like)\n")

        conversation = self.generate_ui_building_conversation(50)

        test_query = {
            "question": "Summarize all the sections we built, the performance optimizations we made, and which tier of pricing has SSO. Also remind me what color we changed the CTA button to.",
            "context_needed": ["hero", "features", "pricing", "SSO", "Enterprise", "WebP", "lazy loading", "orange CTA"]
        }

        # Memory config for 50 turns
        memory_config = {"recent_turns": 30, "semantic_top_k": 5}

        # Run with enhanced retrieval
        print(f"{Colors.CYAN}Running with enhanced retrieval (30 recent + 5 semantic)...{Colors.END}")
        result_retrieval = await self.run_conversation_enhanced(
            conversation,
            test_query,
            memory_config=memory_config
        )

        # Run with full context
        print(f"{Colors.CYAN}Running with full context (baseline)...{Colors.END}")
        result_full = await self.run_conversation(
            conversation,
            test_query,
            use_retrieval=False
        )

        # Compare
        similarity = self.calculate_similarity(result_retrieval['response'], result_full['response'])
        semantic_similarity = self.calculate_semantic_similarity(result_retrieval['response'], result_full['response'])

        print(f"\n{Colors.GREEN}Results:{Colors.END}")
        print(f"  Retrieval tokens: {result_retrieval['total_tokens']}")
        print(f"  Full context tokens: {result_full['total_tokens']}")
        print(f"  Token savings: {Colors.BOLD}{100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens']):.1f}%{Colors.END}")
        print(f"  Text similarity: {Colors.BOLD}{similarity:.2%}{Colors.END}")
        print(f"  Semantic similarity: {Colors.BOLD}{semantic_similarity:.2%}{Colors.END}")

        return {
            'test_name': '50_turn_ui_building',
            'retrieval': result_retrieval,
            'full_context': result_full,
            'text_similarity': similarity,
            'semantic_similarity': semantic_similarity,
            'token_savings_pct': 100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens'])
        }

    async def test_100_turn_conversation(self) -> Dict[str, Any]:
        """Test: 100-turn extended UI building conversation"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}Test 6: 100-Turn Extended UI Building Conversation{Colors.END}")
        print("Scenario: Very long conversation with multiple iterations and changes\n")

        conversation = self.generate_ui_building_conversation(100)

        test_query = {
            "question": "Create a comprehensive summary document covering: 1) All page sections, 2) Performance metrics before/after optimization, 3) Accessibility improvements, 4) Deployment setup. Be specific about the details.",
            "context_needed": ["all sections", "3.2s to 1.1s", "WebP", "WAVE", "ARIA", "Vercel", "preview deployments"]
        }

        # Memory config for 100 turns (more aggressive retrieval)
        memory_config = {"recent_turns": 40, "semantic_top_k": 7}

        # Run with enhanced retrieval
        print(f"{Colors.CYAN}Running with enhanced retrieval (40 recent + 7 semantic)...{Colors.END}")
        result_retrieval = await self.run_conversation_enhanced(
            conversation,
            test_query,
            memory_config=memory_config
        )

        # Run with full context
        print(f"{Colors.CYAN}Running with full context (baseline)...{Colors.END}")
        result_full = await self.run_conversation(
            conversation,
            test_query,
            use_retrieval=False
        )

        # Compare
        similarity = self.calculate_similarity(result_retrieval['response'], result_full['response'])
        semantic_similarity = self.calculate_semantic_similarity(result_retrieval['response'], result_full['response'])

        print(f"\n{Colors.GREEN}Results:{Colors.END}")
        print(f"  Retrieval tokens: {result_retrieval['total_tokens']}")
        print(f"  Full context tokens: {result_full['total_tokens']}")
        print(f"  Token savings: {Colors.BOLD}{100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens']):.1f}%{Colors.END}")
        print(f"  Text similarity: {Colors.BOLD}{similarity:.2%}{Colors.END}")
        print(f"  Semantic similarity: {Colors.BOLD}{semantic_similarity:.2%}{Colors.END}")

        return {
            'test_name': '100_turn_ui_building',
            'retrieval': result_retrieval,
            'full_context': result_full,
            'text_similarity': similarity,
            'semantic_similarity': semantic_similarity,
            'token_savings_pct': 100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens'])
        }

    async def run_conversation_enhanced(
        self,
        conversation: List[Dict[str, str]],
        test_query: Dict[str, str],
        memory_config: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """
        Run conversation using EnhancedMemoryManager with context window retrieval

        Args:
            conversation: List of conversation turns
            test_query: Test query with question
            memory_config: Memory configuration

        Returns:
            Result dict with response and metrics
        """
        if memory_config is None:
            memory_config = {"recent_turns": 30, "semantic_top_k": 5}

        conversation_id = f"test_enhanced_{int(time.time())}"

        # Build conversation history
        for i, turn in enumerate(conversation):
            text = f"{turn['role'].capitalize()}: {turn['content']}"
            self.memory_manager.add_turn(
                conversation_id=conversation_id,
                text=text,
                metadata={
                    'role': turn['role'],
                    'turn_number': i + 1,
                    'timestamp': time.time()
                }
            )

            # Track tokens if tracker available
            if self.token_tracker and turn['role'] == 'assistant':
                user_msg = conversation[i-1]['content'] if i > 0 else ""
                self.token_tracker.track_turn(
                    conversation_id=conversation_id,
                    user_message=user_msg,
                    assistant_message=turn['content']
                )

        # Retrieve context using enhanced manager
        query = test_query['question']
        retrieval_start = time.time()

        memory_results = await self.memory_manager.retrieve_context_enhanced(
            conversation_id=conversation_id,
            query=query,
            token_budget=8000,
            strategy="ui_builder"
        )

        retrieval_latency = (time.time() - retrieval_start) * 1000

        # Build prompt with enhanced context
        context_string = memory_results.get('context', '')

        full_prompt = f"""Context from conversation history:
{context_string}

Current question: {query}
Assistant:"""

        # Generate response
        generation_start = time.time()
        outputs = self.llm.generate(
            prompts=[full_prompt],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "\n\n\n"]
        )

        response = outputs[0].outputs[0].text.strip()
        generation_latency = (time.time() - generation_start) * 1000

        # Count tokens
        prompt_tokens = len(full_prompt.split())
        response_tokens = len(response.split())

        return {
            'response': response,
            'retrieval_latency_ms': retrieval_latency,
            'generation_latency_ms': generation_latency,
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
            'total_tokens': prompt_tokens + response_tokens,
            'mode_used': memory_results.get('mode_used', 'balanced'),
            'method': 'enhanced_retrieval'
        }

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all quality tests and aggregate results"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}  PERPETUAL AI - QUALITY A/B BENCHMARK{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")

        start_time = time.time()

        results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.llm.model_name,
            'tests': []
        }

        # Run all tests
        test_1 = await self.test_multi_hop_reasoning()
        results['tests'].append(test_1)

        test_2 = await self.test_context_dependent_queries()
        results['tests'].append(test_2)

        test_3 = await self.test_code_understanding()
        results['tests'].append(test_3)

        test_4 = await self.test_long_form_generation()
        results['tests'].append(test_4)

        # NEW: Long conversation tests
        test_5 = await self.test_50_turn_conversation()
        results['tests'].append(test_5)

        test_6 = await self.test_100_turn_conversation()
        results['tests'].append(test_6)

        # Calculate aggregate metrics
        avg_text_similarity = np.mean([t['text_similarity'] for t in results['tests']])
        avg_semantic_similarity = np.mean([t['semantic_similarity'] for t in results['tests']])
        avg_token_savings = np.mean([t['token_savings_pct'] for t in results['tests']])

        results['aggregate'] = {
            'avg_text_similarity': avg_text_similarity,
            'avg_semantic_similarity': avg_semantic_similarity,
            'avg_token_savings_pct': avg_token_savings,
            'total_duration_seconds': time.time() - start_time
        }

        # Print summary
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}  AGGREGATE RESULTS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}\n")

        print(f"{Colors.BOLD}Average Text Similarity:{Colors.END} {avg_text_similarity:.2%}")
        print(f"{Colors.BOLD}Average Semantic Similarity:{Colors.END} {avg_semantic_similarity:.2%}")
        print(f"{Colors.BOLD}Average Token Savings:{Colors.END} {avg_token_savings:.1f}%")
        print(f"{Colors.BOLD}Total Duration:{Colors.END} {time.time() - start_time:.1f}s")

        # Quality assessment
        quality_score = avg_semantic_similarity

        print(f"\n{Colors.BOLD}Quality Assessment:{Colors.END}")
        if quality_score >= 0.90:
            print(f"  {Colors.GREEN}✓ EXCELLENT (≥90%) - Production ready!{Colors.END}")
        elif quality_score >= 0.80:
            print(f"  {Colors.YELLOW}○ GOOD (≥80%) - Acceptable for most use cases{Colors.END}")
        elif quality_score >= 0.70:
            print(f"  {Colors.YELLOW}○ FAIR (≥70%) - Consider tuning memory_config{Colors.END}")
        else:
            print(f"  {Colors.RED}✗ POOR (<70%) - Retrieval needs improvement{Colors.END}")

        # Save results
        output_file = f"quality_benchmark_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{Colors.CYAN}Results saved to: {output_file}{Colors.END}")

        return results


async def main():
    """Main entry point"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Quality A/B Benchmark for Perpetual AI')
    parser.add_argument('--model', type=str, default='TheBloke/Mistral-7B-Instruct-v0.2-GPTQ',
                      help='Model name (default: Mistral-7B GPTQ)')
    parser.add_argument('--qdrant-url', type=str, default=None,
                      help='Qdrant URL (default: env QDRANT_URL/QDRANT_CLOUD_URL or http://localhost:6333)')
    parser.add_argument('--qdrant-api-key', type=str, default=None,
                      help='Qdrant API key (default: env QDRANT_API_KEY, optional for local)')

    args = parser.parse_args()

    # Show configuration
    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLOUD_URL") or "http://localhost:6333"
    print(f"\n{Colors.BOLD}Configuration:{Colors.END}")
    print(f"  Model: {args.model}")
    print(f"  Qdrant URL: {qdrant_url}")
    print(f"  Qdrant API Key: {'Set' if (args.qdrant_api_key or os.getenv('QDRANT_API_KEY')) else 'Not set (local mode)'}\n")

    benchmark = QualityBenchmark(
        model_name=args.model,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key
    )
    results = await benchmark.run_all_tests()

    # Exit with appropriate code
    quality_score = results['aggregate']['avg_semantic_similarity']
    exit_code = 0 if quality_score >= 0.80 else 1

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
