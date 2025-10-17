"""
Semantic Clustering using GPT-4o-mini for intelligent conversation grouping
Compares against k-means to prove semantic understanding improves multi-hop reasoning
"""
import os
import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from openai import AsyncOpenAI


@dataclass
class ClusterGroup:
    """A semantic cluster of conversation turns"""
    cluster_id: int
    label: str  # Semantic label (e.g., "Hero section design")
    turn_numbers: List[int]
    turns: List[Dict]  # Full turn data
    summary: str  # Brief summary of cluster content
    embedding_centroid: Optional[List[float]] = None


class SemanticClusterer:
    """
    Intelligent clustering using LLM (GPT-4o-mini or local Mistral)
    Provides semantic understanding vs pure vector similarity (k-means)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        vllm_engine = None
    ):
        """
        Initialize semantic clusterer

        Args:
            api_key: OpenAI API key (defaults to env OPENAI_API_KEY)
            model: Model to use for clustering (gpt-4o-mini recommended)
            vllm_engine: Local vLLM engine (alternative to OpenAI)
        """
        if vllm_engine:
            # Use local vLLM (FREE!)
            self.client = None
            self.vllm_engine = vllm_engine
            self.model = "local_vllm"
            self.is_local = True
            print(f"  [Clustering] Using local vLLM model")
        else:
            # Use OpenAI API
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass vllm_engine.")

            self.client = AsyncOpenAI(api_key=api_key)
            self.vllm_engine = None
            self.model = model
            self.is_local = False

        self.cluster_cache = {}  # Cache clusters per conversation

    async def cluster_conversation(
        self,
        conversation: List[Dict[str, str]],
        conversation_id: str,
        num_clusters: int = 5,
        embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[ClusterGroup], float]:
        """
        Cluster conversation turns using GPT-4o-mini semantic understanding

        Args:
            conversation: List of {"role": "user/assistant", "content": "..."}
            conversation_id: Unique conversation ID for caching
            num_clusters: Target number of semantic groups
            embeddings: Pre-computed embeddings (optional)

        Returns:
            Tuple of (cluster_groups, cost_usd)
        """
        # Check cache
        cache_key = f"{conversation_id}_{len(conversation)}_{num_clusters}"
        if cache_key in self.cluster_cache:
            print(f"  [Clustering] Using cached clusters for {conversation_id}")
            return self.cluster_cache[cache_key], 0.0

        start_time = time.time()

        # Format conversation for analysis
        formatted_turns = []
        for i, turn in enumerate(conversation):
            formatted_turns.append({
                "turn": i + 1,
                "role": turn["role"],
                "content": turn["content"]
            })

        # Build clustering prompt
        prompt = self._build_clustering_prompt(formatted_turns, num_clusters)

        # Call LLM for semantic clustering
        print(f"  [Clustering] Analyzing {len(conversation)} turns with {self.model}...")

        if self.is_local:
            # Use local vLLM (FREE)
            full_prompt = f"""You are an expert at analyzing conversations and grouping related topics semantically.

{prompt}

Return ONLY valid JSON, nothing else."""

            outputs = self.vllm_engine.generate(
                prompts=[full_prompt],
                max_tokens=1024,
                temperature=0.3,
                stop=["```", "\n\n\n"]
            )

            response_text = outputs[0].outputs[0].text.strip()

            # Try to extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
            cost = 0.0  # Local vLLM is free!
        else:
            # Use OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing conversations and grouping related topics semantically."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)

        # Build ClusterGroup objects
        clusters = []
        for cluster_data in result["clusters"]:
            cluster = ClusterGroup(
                cluster_id=cluster_data["cluster_id"],
                label=cluster_data["label"],
                turn_numbers=cluster_data["turn_numbers"],
                turns=[conversation[t-1] for t in cluster_data["turn_numbers"]],
                summary=cluster_data["summary"],
                embedding_centroid=None  # Can compute later if needed
            )
            clusters.append(cluster)

        latency = (time.time() - start_time) * 1000
        print(f"  [Clustering] Created {len(clusters)} semantic groups in {latency:.0f}ms (${cost:.4f})")

        # Cache result
        self.cluster_cache[cache_key] = (clusters, cost)

        return clusters, cost

    def _build_clustering_prompt(self, turns: List[Dict], num_clusters: int) -> str:
        """Build prompt for GPT-4o-mini clustering"""
        turns_text = "\n".join([
            f"Turn {t['turn']} ({t['role']}): {t['content']}"
            for t in turns
        ])

        prompt = f"""Analyze this conversation and group the turns into {num_clusters} semantic clusters based on topic/theme.

Conversation:
{turns_text}

Instructions:
1. Group turns that discuss the same topic together (e.g., "hero section design", "pricing tiers", "performance optimization")
2. Preserve logical dependencies (if turn 5 references turn 2, keep them together)
3. Create meaningful labels for each cluster
4. Provide a brief summary of each cluster's content

Return JSON in this format:
{{
  "clusters": [
    {{
      "cluster_id": 1,
      "label": "Descriptive label",
      "turn_numbers": [1, 2, 5],
      "summary": "Brief summary of this topic"
    }},
    ...
  ]
}}

Ensure all turns are assigned to exactly one cluster."""

        return prompt

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for GPT-4o-mini"""
        # GPT-4o-mini pricing (as of 2024)
        input_cost_per_1m = 0.150  # $0.15 per 1M input tokens
        output_cost_per_1m = 0.600  # $0.60 per 1M output tokens

        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m

        return input_cost + output_cost

    async def find_relevant_clusters(
        self,
        clusters: List[ClusterGroup],
        query: str,
        top_k: int = 2
    ) -> List[ClusterGroup]:
        """
        Find most relevant clusters for a query using LLM

        Args:
            clusters: List of semantic clusters
            query: User query
            top_k: Number of clusters to return

        Returns:
            List of relevant clusters
        """
        # Build prompt
        cluster_descriptions = "\n".join([
            f"Cluster {c.cluster_id}: {c.label} - {c.summary}"
            for c in clusters
        ])

        prompt = f"""Given this query and list of conversation clusters, identify the {top_k} most relevant clusters.

Query: {query}

Available clusters:
{cluster_descriptions}

Return JSON with cluster IDs in order of relevance:
{{
  "relevant_cluster_ids": [1, 3]
}}"""

        if self.is_local:
            # Use local vLLM
            full_prompt = f"""You are an expert at matching queries to relevant conversation topics.

{prompt}

Return ONLY valid JSON, nothing else."""

            outputs = self.vllm_engine.generate(
                prompts=[full_prompt],
                max_tokens=256,
                temperature=0.1,
                stop=["```", "\n\n\n"]
            )

            response_text = outputs[0].outputs[0].text.strip()

            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
        else:
            # Use OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at matching queries to relevant conversation topics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

        relevant_ids = result["relevant_cluster_ids"][:top_k]

        # Return clusters in order of relevance
        relevant_clusters = [
            c for c in clusters if c.cluster_id in relevant_ids
        ]

        # Sort by relevance order
        id_order = {cid: i for i, cid in enumerate(relevant_ids)}
        relevant_clusters.sort(key=lambda c: id_order.get(c.cluster_id, 999))

        return relevant_clusters


class KMeansClusterer:
    """
    Traditional k-means clustering for comparison
    Uses only embedding similarity (no semantic understanding)
    """

    def __init__(self):
        """Initialize k-means clusterer"""
        pass

    def cluster_conversation(
        self,
        conversation: List[Dict[str, str]],
        embeddings: np.ndarray,
        num_clusters: int = 5
    ) -> Tuple[List[ClusterGroup], float]:
        """
        Cluster using k-means on embeddings

        Args:
            conversation: List of conversation turns
            embeddings: Turn embeddings (n_turns x embedding_dim)
            num_clusters: Number of clusters

        Returns:
            Tuple of (cluster_groups, cost=0.0)
        """
        start_time = time.time()

        # Run k-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Group turns by cluster
        cluster_map = {}
        for i, label in enumerate(labels):
            if label not in cluster_map:
                cluster_map[label] = []
            cluster_map[label].append(i + 1)  # Turn numbers start at 1

        # Build ClusterGroup objects
        clusters = []
        for cluster_id, turn_numbers in cluster_map.items():
            # Generic label for k-means
            label = f"Cluster {cluster_id + 1}"

            # Simple summary: just first turn content
            turns = [conversation[t-1] for t in turn_numbers]
            summary = turns[0]["content"][:60] + "..." if turns else ""

            cluster = ClusterGroup(
                cluster_id=cluster_id,
                label=label,
                turn_numbers=turn_numbers,
                turns=turns,
                summary=summary,
                embedding_centroid=kmeans.cluster_centers_[cluster_id].tolist()
            )
            clusters.append(cluster)

        latency = (time.time() - start_time) * 1000
        print(f"  [K-means] Created {len(clusters)} clusters in {latency:.0f}ms (free)")

        return clusters, 0.0  # K-means is free

    def find_relevant_clusters(
        self,
        clusters: List[ClusterGroup],
        query_embedding: np.ndarray,
        top_k: int = 2
    ) -> List[ClusterGroup]:
        """
        Find relevant clusters using embedding distance

        Args:
            clusters: List of clusters with centroids
            query_embedding: Query embedding vector
            top_k: Number of clusters to return

        Returns:
            List of relevant clusters
        """
        # Calculate distances to centroids
        distances = []
        for cluster in clusters:
            if cluster.embedding_centroid is None:
                continue

            centroid = np.array(cluster.embedding_centroid)
            # Cosine similarity
            similarity = np.dot(query_embedding, centroid) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(centroid)
            )
            distances.append((cluster, similarity))

        # Sort by similarity (descending)
        distances.sort(key=lambda x: x[1], reverse=True)

        return [cluster for cluster, _ in distances[:top_k]]


def create_clusterer(method: str = "gpt4o", **kwargs):
    """
    Factory function to create clusterer

    Args:
        method: "gpt4o" or "kmeans"
        **kwargs: Additional args for clusterer

    Returns:
        Clusterer instance
    """
    if method == "gpt4o":
        return SemanticClusterer(**kwargs)
    elif method == "kmeans":
        return KMeansClusterer(**kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
