"""
Long Conversation Quality Benchmark - 50 and 100 Turn Tests
Tests realistic long-form conversations where token savings matter most
"""
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_wrapper import VLLMEngine
from core.vector_db import QdrantAdapter
from core.enhanced_memory_manager import EnhancedMemoryManager
from services.token_tracker import TokenTracker
from difflib import SequenceMatcher
import numpy as np


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


class LongConversationBenchmark:
    """Benchmark for 50 and 100 turn conversations"""

    def __init__(
        self,
        model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        redis_url: str = None
    ):
        print(f"{Colors.CYAN}Initializing Long Conversation Benchmark...{Colors.END}")

        # Initialize vLLM
        print(f"Loading model: {model_name}")
        self.llm = VLLMEngine(
            model_name=model_name,
            quantization="gptq",
            gpu_memory_utilization=0.9,
            max_model_len=4096
        )

        # Initialize Qdrant
        import os
        qdrant_url = qdrant_url or os.getenv("QDRANT_CLOUD_URL") or "http://localhost:6333"
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        print(f"Connecting to Qdrant at: {qdrant_url}")
        self.vector_db = QdrantAdapter(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name="long_conv_benchmark"
        )

        # Initialize token tracker (optional - works without Redis for benchmark)
        try:
            redis_url = redis_url or os.getenv("REDIS_URL")
            if redis_url:
                self.token_tracker = TokenTracker(redis_url=redis_url)
                print("✓ Token tracker initialized with Redis")
            else:
                self.token_tracker = None
                print("⚠️  Token tracker disabled (no Redis)")
        except Exception as e:
            print(f"⚠️  Token tracker disabled: {e}")
            self.token_tracker = None

        # Initialize enhanced memory manager
        self.memory_manager = EnhancedMemoryManager(
            vector_db=self.vector_db,
            token_tracker=self.token_tracker,
            cache_capacity=2000
        )

        print(f"{Colors.GREEN}✓ Initialization complete{Colors.END}\n")

    def generate_ui_building_conversation(self, num_turns: int) -> List[Dict[str, str]]:
        """Generate realistic UI building conversation (Lovable-like)"""
        # Base conversation
        conversation = [
            {"role": "user", "content": "I want to build a landing page for my SaaS product."},
            {"role": "assistant", "content": "Great! What's your SaaS product about?"},
            {"role": "user", "content": "It's a project management tool for remote teams."},
            {"role": "assistant", "content": "Perfect. For a PM tool landing page, we'll need: hero section, features, pricing, and CTA. What's your primary CTA?"},
            {"role": "user", "content": "Start Free Trial - 14 days, no credit card."},
            {"role": "assistant", "content": "Excellent hook. Let's start with a clean hero: headline, subheadline, CTA button, and hero image. What's your main value prop?"},
            {"role": "user", "content": "Ship projects 2x faster with AI-powered task prioritization."},
            {"role": "assistant", "content": "Strong! I'll create: large headline '2x Faster Project Delivery', subhead about AI prioritization, prominent CTA button in brand color. What color scheme?"},
            {"role": "user", "content": "Modern tech look: deep blue primary (#1E40AF), white, gray accents."},
            {"role": "assistant", "content": "Great choice. Deep blue conveys trust and professionalism. I'll use white backgrounds with blue CTAs. Now, what are your top 3 features to highlight?"},
            {"role": "user", "content": "1) AI task prioritization, 2) Real-time collaboration, 3) Automated progress tracking."},
            {"role": "assistant", "content": "Perfect trio. I'll create a 3-column feature section with icons. Each feature gets: icon, headline, 2-sentence description. Should we add screenshots or keep it minimal?"},
        ]

        # Extend conversation with iterative design decisions
        extensions = [
            {"role": "user", "content": "Add screenshots. Show the AI prioritization dashboard."},
            {"role": "assistant", "content": "I'll add a product screenshot showing the AI priority scores and task cards. Should it be full-width or contained with padding?"},
            {"role": "user", "content": "Full-width with subtle shadow for depth."},
            {"role": "assistant", "content": "Done. Shadow added (0 4px 6px rgba(0,0,0,0.1)). Now for pricing - how many tiers?"},
            {"role": "user", "content": "Three tiers: Starter ($29/mo), Professional ($99/mo), Enterprise (custom)."},
            {"role": "assistant", "content": "Classic SaaS structure. I'll create pricing cards with: price, features list, CTA button. Which tier should be highlighted?"},
            {"role": "user", "content": "Professional - most popular. Add a 'Most Popular' badge."},
            {"role": "assistant", "content": "Added badge in blue with white text above Professional tier. Each card lists 5-6 key features. What differentiates Starter from Pro?"},
            {"role": "user", "content": "Starter: 5 team members, 10 projects. Pro: unlimited members, unlimited projects, AI features."},
            {"role": "assistant", "content": "Clear value ladder. I've added those limits. For Enterprise: 'Custom solutions, dedicated support, SSO'. Sound good?"},
            {"role": "user", "content": "Perfect. Add a testimonials section with 3 customer quotes."},
            {"role": "assistant", "content": "I'll create a testimonials carousel with: customer photo, quote, name, company, role. Do you have real testimonials or should I use placeholders?"},
            {"role": "user", "content": "Use placeholders for now: tech companies, engineering managers as personas."},
            {"role": "assistant", "content": "Created 3 testimonials: 'Reduced planning time by 60%', 'Game-changer for remote teams', 'Best PM tool we've used'. Each with placeholder headshots and titles like 'VP Engineering at TechCorp'. Now, footer?"},
            {"role": "user", "content": "Footer with: company links, product links, social media, email signup."},
            {"role": "assistant", "content": "Building footer with 4 columns: Product (features, pricing, demo), Company (about, careers, contact), Resources (blog, docs, support), plus email newsletter signup. What social platforms?"},
            {"role": "user", "content": "Twitter, LinkedIn, GitHub."},
            {"role": "assistant", "content": "Added social icons. Newsletter CTA: 'Get product updates and tips'. One more thing - should we add a demo video section?"},
            {"role": "user", "content": "Yes! After hero, before features. Embedded YouTube video."},
            {"role": "assistant", "content": "Perfect placement. Added video section with: headline 'See It In Action', video embed (16:9), caption 'Watch how teams ship faster'. Video has play button overlay. What about mobile responsiveness?"},
            {"role": "user", "content": "Must be mobile-first. Stack columns on mobile, larger touch targets."},
            {"role": "assistant", "content": "Implemented responsive breakpoints: mobile (< 768px) stacks all columns, tablet (768-1024px) shows 2 columns, desktop (> 1024px) full grid. Touch targets are 44px minimum. Hamburger menu on mobile?"},
            {"role": "user", "content": "Yes, hamburger menu with slide-out navigation."},
            {"role": "assistant", "content": "Added mobile nav: hamburger icon top-right, slides in from right, includes all nav links plus CTA button. Smooth animation (300ms). Background overlay dims page. What about loading states?"},
            {"role": "user", "content": "Show skeleton loaders for images and content."},
            {"role": "assistant", "content": "Implemented skeleton screens: pulsing gray blocks for images, text lines with gradient animation. Images fade in when loaded. Improves perceived performance. Should we add analytics?"},
            {"role": "user", "content": "Yes, Google Analytics and event tracking on CTA clicks."},
            {"role": "assistant", "content": "Added GA4 tracking: pageview on load, events for 'cta_click', 'demo_play', 'pricing_view'. Each event includes source section. Now, SEO optimization?"},
            {"role": "user", "content": "Critical. Proper meta tags, Open Graph, structured data."},
            {"role": "assistant", "content": "Implemented: meta description (155 chars), OG image (1200x630), Twitter cards, JSON-LD structured data for SoftwareApplication. Title: 'ProjectPro - Ship 2x Faster with AI'. Alt text on all images. Semantic HTML (header, main, section, footer). What about performance?"},
            {"role": "user", "content": "Must load fast. Optimize images, lazy load below fold."},
            {"role": "assistant", "content": "Optimized: images converted to WebP with fallbacks, lazy loading on images below fold, minified CSS/JS, preload critical fonts. Estimated Lighthouse score: 95+. Should we add A/B testing capability?"},
            {"role": "user", "content": "Not yet. Focus on launch. Add a cookie consent banner though."},
            {"role": "assistant", "content": "Added GDPR-compliant cookie banner: bottom of page, 'Accept' and 'Decline' buttons, link to privacy policy. Dismisses on accept, stores choice in localStorage. What's the domain?"},
            {"role": "user", "content": "projectpro.io - already registered. Need SSL setup."},
            {"role": "assistant", "content": "For deployment: use Vercel or Netlify for automatic SSL (Let's Encrypt). Both support custom domains with one-click SSL. I'll add _redirects for HTTPS enforcement. Environment: production or staging first?"},
            {"role": "user", "content": "Staging first at staging.projectpro.io for review."},
            {"role": "assistant", "content": "Set up staging environment. Will deploy to staging subdomain. You can review there before production. Need any forms besides newsletter?"},
            {"role": "user", "content": "Add a 'Request Demo' form: name, email, company, company size dropdown."},
            {"role": "assistant", "content": "Created demo request form modal: triggered by 'Request Demo' CTA, fields with validation (email format, required fields), company size options (1-10, 11-50, 51-200, 200+). Where should form submissions go?"},
            {"role": "user", "content": "Send to our CRM via webhook. I'll give you the endpoint."},
            {"role": "assistant", "content": "Ready to integrate webhook. POST request with form data JSON. Need: endpoint URL, auth header if required. Should we add reCAPTCHA?"},
            {"role": "user", "content": "Yes, invisible reCAPTCHA v3 on form submission."},
            {"role": "assistant", "content": "Integrated reCAPTCHA v3: validates on submit, score threshold 0.5, fallback to visible challenge if score too low. Protects against bots without user friction. Any accessibility requirements?"},
            {"role": "user", "content": "WCAG 2.1 AA compliant. Keyboard navigation, screen reader support."},
            {"role": "assistant", "content": "Implemented a11y features: skip navigation link, proper ARIA labels, focus indicators (2px blue outline), keyboard nav for all interactive elements, alt text, semantic HTML, color contrast ratio > 4.5:1. Tested with screen reader. What about internationalization?"},
            {"role": "user", "content": "English only for MVP. But structure for future i18n."},
            {"role": "assistant", "content": "Structured for i18n: all copy in en.json, React i18n hooks ready, lang attribute in HTML. Easy to add languages later. Final review: any changes to hero headline or CTA?"},
        ]

        # Add extensions until we reach desired length
        conversation.extend(extensions[:num_turns - len(conversation)])

        # Pad if needed with generic design iterations
        while len(conversation) < num_turns:
            user_q = {"role": "user", "content": f"Adjust spacing in section {len(conversation)//2}. Looks too cramped."}
            assistant_a = {"role": "assistant", "content": f"Increased padding to 48px top/bottom, 24px left/right. Better visual breathing room. Section {len(conversation)//2} now has consistent spacing with rest of page."}
            conversation.extend([user_q, assistant_a])

        return conversation[:num_turns]

    async def test_conversation(
        self,
        conversation: List[Dict],
        test_query: Dict,
        test_name: str,
        memory_config: Dict = None
    ) -> Dict[str, Any]:
        """Test a single conversation with retrieval vs full context"""
        if memory_config is None:
            memory_config = {"recent_turns": 30, "semantic_top_k": 5}

        print(f"\n{Colors.BOLD}{Colors.BLUE}Test: {test_name}{Colors.END}")
        print(f"Conversation length: {len(conversation)} turns\n")

        # Test with ENHANCED retrieval
        print(f"{Colors.CYAN}Running with ENHANCED retrieval (30 recent + 5 semantic + context window)...{Colors.END}")
        result_retrieval = await self.run_with_enhanced_retrieval(
            conversation,
            test_query,
            memory_config
        )

        # Test with FULL context
        print(f"{Colors.CYAN}Running with FULL context (baseline)...{Colors.END}")
        result_full = await self.run_with_full_context(
            conversation,
            test_query
        )

        # Calculate similarities
        similarity = SequenceMatcher(None, result_retrieval['response'], result_full['response']).ratio()

        # Semantic similarity
        emb1 = self.vector_db.model.encode([result_retrieval['response']])[0]
        emb2 = self.vector_db.model.encode([result_full['response']])[0]
        semantic_similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

        print(f"\n{Colors.GREEN}Results:{Colors.END}")
        print(f"  Retrieval tokens: {result_retrieval['total_tokens']}")
        print(f"  Full context tokens: {result_full['total_tokens']}")
        print(f"  Token savings: {Colors.BOLD}{100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens']):.1f}%{Colors.END}")
        print(f"  Text similarity: {Colors.BOLD}{similarity:.2%}{Colors.END}")
        print(f"  Semantic similarity: {Colors.BOLD}{semantic_similarity:.2%}{Colors.END}")

        return {
            'test_name': test_name,
            'conversation_length': len(conversation),
            'retrieval': result_retrieval,
            'full_context': result_full,
            'text_similarity': similarity,
            'semantic_similarity': semantic_similarity,
            'token_savings_pct': 100 * (1 - result_retrieval['total_tokens'] / result_full['total_tokens'])
        }

    async def run_with_enhanced_retrieval(
        self,
        conversation: List[Dict],
        test_query: Dict,
        memory_config: Dict
    ) -> Dict:
        """Run conversation with enhanced retrieval"""
        conv_id = f"test_enh_{int(time.time())}"

        # Build conversation in memory
        for i, turn in enumerate(conversation):
            self.memory_manager.vector_db.add(
                conversation_id=conv_id,
                text=f"{turn['role'].capitalize()}: {turn['content']}",
                metadata={'role': turn['role'], 'turn_number': i, 'timestamp': time.time()}
            )

        # Get enhanced context
        retrieval_start = time.time()
        context_result = await self.memory_manager.retrieve_context_enhanced(
            conversation_id=conv_id,
            query=test_query['question'],
            token_budget=8000,
            strategy="ui_builder"
        )
        retrieval_latency = (time.time() - retrieval_start) * 1000

        # Build prompt
        prompt = f"{context_result['context']}\n\nUser: {test_query['question']}\nAssistant:"

        # Generate
        gen_start = time.time()
        outputs = self.llm.generate([prompt], max_tokens=512, temperature=0.7, stop=["User:", "\n\n\n"])
        response = outputs[0].outputs[0].text.strip()
        gen_latency = (time.time() - gen_start) * 1000

        return {
            'response': response,
            'retrieval_latency_ms': retrieval_latency,
            'generation_latency_ms': gen_latency,
            'prompt_tokens': len(prompt.split()),
            'response_tokens': len(response.split()),
            'total_tokens': len(prompt.split()) + len(response.split()),
            'mode_used': context_result.get('mode_used', 'unknown')
        }

    async def run_with_full_context(self, conversation: List[Dict], test_query: Dict) -> Dict:
        """Run conversation with full context"""
        # Build full context prompt
        full_prompt = "\n".join([
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in conversation
        ])
        full_prompt += f"\nUser: {test_query['question']}\nAssistant:"

        # Generate
        gen_start = time.time()
        outputs = self.llm.generate([full_prompt], max_tokens=512, temperature=0.7, stop=["User:", "\n\n\n"])
        response = outputs[0].outputs[0].text.strip()
        gen_latency = (time.time() - gen_start) * 1000

        return {
            'response': response,
            'retrieval_latency_ms': 0,
            'generation_latency_ms': gen_latency,
            'prompt_tokens': len(full_prompt.split()),
            'response_tokens': len(response.split()),
            'total_tokens': len(full_prompt.split()) + len(response.split())
        }

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run 50 and 100 turn benchmarks"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}  LONG CONVERSATION BENCHMARK{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")

        start_time = time.time()
        results = {'timestamp': datetime.now().isoformat(), 'model': self.llm.model_name, 'tests': []}

        # Test 1: 50-turn UI building
        conv_50 = self.generate_ui_building_conversation(50)
        test_50 = await self.test_conversation(
            conv_50,
            {"question": "Summarize all the key design decisions we made and list the main sections of the landing page."},
            "50-Turn UI Building Conversation",
            {"recent_turns": 30, "semantic_top_k": 5}
        )
        results['tests'].append(test_50)

        # Test 2: 100-turn UI building
        conv_100 = self.generate_ui_building_conversation(100)
        test_100 = await self.test_conversation(
            conv_100,
            {"question": "Create a comprehensive checklist of all features, integrations, and optimizations we implemented."},
            "100-Turn UI Building Conversation",
            {"recent_turns": 40, "semantic_top_k": 7}
        )
        results['tests'].append(test_100)

        # Aggregate metrics
        avg_text_sim = np.mean([t['text_similarity'] for t in results['tests']])
        avg_semantic_sim = np.mean([t['semantic_similarity'] for t in results['tests']])
        avg_token_savings = np.mean([t['token_savings_pct'] for t in results['tests']])

        results['aggregate'] = {
            'avg_text_similarity': avg_text_sim,
            'avg_semantic_similarity': avg_semantic_sim,
            'avg_token_savings_pct': avg_token_savings,
            'total_duration_seconds': time.time() - start_time
        }

        # Print summary
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}  AGGREGATE RESULTS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}\n")
        print(f"{Colors.BOLD}Average Text Similarity:{Colors.END} {avg_text_sim:.2%}")
        print(f"{Colors.BOLD}Average Semantic Similarity:{Colors.END} {avg_semantic_sim:.2%}")
        print(f"{Colors.BOLD}Average Token Savings:{Colors.END} {avg_token_savings:.1f}%")
        print(f"{Colors.BOLD}Total Duration:{Colors.END} {time.time() - start_time:.1f}s")

        # Quality assessment
        if avg_semantic_sim >= 0.90:
            print(f"\n  {Colors.GREEN}✓ EXCELLENT (≥90%) - Production ready!{Colors.END}")
        elif avg_semantic_sim >= 0.80:
            print(f"\n  {Colors.YELLOW}○ GOOD (≥80%) - Acceptable for most use cases{Colors.END}")
        else:
            print(f"\n  {Colors.RED}✗ NEEDS WORK (<80%) - Tune retrieval settings{Colors.END}")

        # Save results
        output_file = f"long_conv_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{Colors.CYAN}Results saved to: {output_file}{Colors.END}")

        return results


async def main():
    import argparse, os

    parser = argparse.ArgumentParser(description='Long Conversation Benchmark')
    parser.add_argument('--model', type=str, default='TheBloke/Mistral-7B-Instruct-v0.2-GPTQ')
    parser.add_argument('--qdrant-url', type=str, default=None)
    parser.add_argument('--qdrant-api-key', type=str, default=None)
    parser.add_argument('--redis-url', type=str, default=None)
    args = parser.parse_args()

    benchmark = LongConversationBenchmark(
        model_name=args.model,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        redis_url=args.redis_url
    )
    results = await benchmark.run_all_tests()

    exit_code = 0 if results['aggregate']['avg_semantic_similarity'] >= 0.80 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
