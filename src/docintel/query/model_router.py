"""
Model routing logic for selecting between GPT-4.1 and GPT-5-mini.

This module implements intelligent routing between Azure OpenAI models based on:
- Context size (GPT-4.1: 1M tokens, GPT-5-mini: 400K tokens)
- Image presence (GPT-5-mini for multimodal)
- Query type (reasoning vs simple Q&A)
- Cost optimization (GPT-4.1 cheaper for text-only)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelChoice(str, Enum):
    """Available Azure OpenAI models."""
    GPT_4_1 = "gpt-4.1"
    GPT_5_MINI = "gpt-5-mini"


@dataclass
class RoutingDecision:
    """Result of routing decision with reasoning."""
    model: ModelChoice
    reason: str
    estimated_tokens: int
    has_images: bool
    metadata: Dict[str, Any]


class ModelRouter:
    """
    Routes queries to the optimal model based on requirements.
    
    Routing Rules:
    1. Multimodal (images) → GPT-5-mini
    2. Large context (>400K tokens) → GPT-4.1
    3. Visual reasoning keywords → GPT-5-mini (if images available)
    4. Default text queries → GPT-4.1 (cheaper)
    """
    
    # Model specifications
    GPT_4_1_MAX_TOKENS = 1_047_576  # ~1M tokens
    GPT_5_MINI_MAX_TOKENS = 400_000  # 400K tokens
    
    # Safety buffer (leave room for output tokens)
    GPT_4_1_SAFE_INPUT = 1_000_000  # Leave 47K for output
    GPT_5_MINI_SAFE_INPUT = 350_000  # Leave 50K for output
    
    # Keywords that suggest visual reasoning
    VISUAL_KEYWORDS = [
        "figure", "chart", "diagram", "image", "graph", "plot",
        "table", "visualization", "flowchart", "schematic",
        "efficacy curve", "survival curve", "kaplan-meier"
    ]
    
    # Keywords that suggest advanced reasoning
    REASONING_KEYWORDS = [
        "analyze", "compare", "reasoning", "step by step",
        "explain why", "justify", "rationale", "evidence",
        "statistics", "stats", "statistical", "observed",
        "calculate", "compute", "derive", "interpret",
        "efficacy", "safety", "outcomes", "results",
        "correlation", "significance", "p-value", "confidence interval"
    ]
    
    def __init__(self, disable_routing: bool = False):
        """
        Initialize the model router.
        
        Args:
            disable_routing: If True, always fallback to GPT-4.1 (bypass routing logic)
        """
        self.routing_log: List[RoutingDecision] = []
        self.disable_routing = disable_routing
    
    def route(
        self,
        query_text: str,
        images: Optional[List[Path | str]] = None,
        context_docs: Optional[List[str]] = None,
        force_model: Optional[ModelChoice] = None
    ) -> RoutingDecision:
        """
        Determine which model to use for the query.
        
        Args:
            query_text: The user's query text
            images: Optional list of image paths/URLs
            context_docs: Optional list of context documents
            force_model: Optional override to force a specific model
            
        Returns:
            RoutingDecision with model choice and reasoning
        """
        images = images or []
        context_docs = context_docs or []
        
        # Estimate token count (needed for all paths)
        estimated_tokens = self._estimate_tokens(query_text, images, context_docs)
        has_images = len(images) > 0
        
        # Feature flag: disable routing and always use GPT-4.1 fallback
        if self.disable_routing:
            decision = RoutingDecision(
                model=ModelChoice.GPT_4_1,
                reason="routing_disabled_fallback",
                estimated_tokens=estimated_tokens,
                has_images=has_images,
                metadata={"routing_disabled": True}
            )
            self.routing_log.append(decision)
            logger.info(f"Routing disabled - falling back to {decision.model}")
            return decision
        
        # Override if model is forced
        if force_model:
            decision = RoutingDecision(
                model=force_model,
                reason="forced_override",
                estimated_tokens=estimated_tokens,
                has_images=has_images,
                metadata={"forced": True}
            )
            self.routing_log.append(decision)
            return decision
        
        # Rule 1: Has images → GPT-5-mini (multimodal capability)
        if has_images:
            decision = RoutingDecision(
                model=ModelChoice.GPT_5_MINI,
                reason="multimodal_query_with_images",
                estimated_tokens=estimated_tokens,
                has_images=True,
                metadata={
                    "image_count": len(images)
                }
            )
            self.routing_log.append(decision)
            logger.info(f"Routing to {decision.model}: {decision.reason} "
                       f"({len(images)} images)")
            return decision
        
        # Rule 2: Advanced reasoning → GPT-5-mini
        if self._has_reasoning_keywords(query_text):
            decision = RoutingDecision(
                model=ModelChoice.GPT_5_MINI,
                reason="advanced_reasoning_query",
                estimated_tokens=estimated_tokens,
                has_images=has_images,
                metadata={
                    "reasoning_keywords_found": True
                }
            )
            self.routing_log.append(decision)
            logger.info(f"Routing to {decision.model}: {decision.reason}")
            return decision
        
        # Rule 3: Default → GPT-4.1 (cheaper for simple text queries)
        decision = RoutingDecision(
            model=ModelChoice.GPT_4_1,
            reason="default_text_query",
            estimated_tokens=estimated_tokens,
            has_images=has_images,
            metadata={
                "cost_optimized": True
            }
        )
        self.routing_log.append(decision)
        logger.info(f"Routing to {decision.model}: {decision.reason}")
        return decision
    
    def _estimate_tokens(
        self,
        query_text: str,
        images: List[Path | str],
        context_docs: List[str]
    ) -> int:
        """
        Estimate total token count for the query.
        
        Rough estimation:
        - 1 token ≈ 4 characters (English text)
        - 1 image ≈ 1000 tokens (conservative, actual varies 765-1500)
        
        Args:
            query_text: The query text
            images: List of image paths
            context_docs: List of context documents
            
        Returns:
            Estimated token count
        """
        # Text tokens (rough estimate: 1 token per 4 characters)
        query_tokens = len(query_text) // 4
        
        # Context tokens (each doc counted separately)
        context_tokens = sum(len(doc) // 4 for doc in context_docs)
        
        # Image tokens (conservative estimate)
        image_tokens = len(images) * 1000
        
        # Total
        text_tokens = query_tokens + context_tokens
        total = text_tokens + image_tokens
        
        logger.debug(
            f"Token estimate: {total:,} "
            f"(query: {query_tokens:,}, context: {context_tokens:,}, images: {image_tokens:,})"
        )
        
        return total
    
    def _has_visual_keywords(self, text: str) -> bool:
        """Check if query contains visual reasoning keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.VISUAL_KEYWORDS)
    
    def _has_reasoning_keywords(self, text: str) -> bool:
        """Check if query contains advanced reasoning keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.REASONING_KEYWORDS)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics on routing decisions.
        
        Returns:
            Dictionary with routing statistics
        """
        if not self.routing_log:
            return {"total_queries": 0}
        
        total = len(self.routing_log)
        gpt41_count = sum(1 for d in self.routing_log if d.model == ModelChoice.GPT_4_1)
        gpt5_count = sum(1 for d in self.routing_log if d.model == ModelChoice.GPT_5_MINI)
        
        reasons = {}
        for decision in self.routing_log:
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
        
        return {
            "total_queries": total,
            "gpt_4_1_count": gpt41_count,
            "gpt_5_mini_count": gpt5_count,
            "gpt_4_1_percentage": (gpt41_count / total * 100) if total > 0 else 0,
            "gpt_5_mini_percentage": (gpt5_count / total * 100) if total > 0 else 0,
            "routing_reasons": reasons,
            "avg_tokens": sum(d.estimated_tokens for d in self.routing_log) // total if total > 0 else 0
        }
    
    def clear_log(self) -> None:
        """Clear the routing log."""
        self.routing_log.clear()
        logger.debug("Routing log cleared")


# Singleton instance
_router: Optional[ModelRouter] = None


def get_router(disable_routing: bool = False) -> ModelRouter:
    """
    Get or create the singleton model router instance.
    
    Args:
        disable_routing: If True, router always falls back to GPT-4.1
    
    Returns:
        ModelRouter instance
    """
    global _router
    if _router is None:
        _router = ModelRouter(disable_routing=disable_routing)
    return _router
