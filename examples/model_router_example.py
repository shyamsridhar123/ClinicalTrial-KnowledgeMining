#!/usr/bin/env python3
"""
Example: Using the Model Router for Clinical Trial Queries

This script demonstrates how to use the model router and LLM client
to intelligently route queries between GPT-4.1 and GPT-5-mini.

Run with:
    pixi run python examples/model_router_example.py
"""

from __future__ import annotations

import logging
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docintel.query.model_router import get_router, ModelChoice
from docintel.query.llm_client import get_llm_client
from docintel.query.token_utils import estimate_query_tokens

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def example_1_simple_text_query():
    """Example 1: Simple text query → Routes to GPT-4.1 (cheaper)."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Text Query")
    print("="*80)
    
    router = get_router()
    client = get_llm_client()
    
    query = "What is the primary endpoint of NCT02826161?"
    
    # Route the query
    decision = router.route(query_text=query)
    
    print(f"\n📊 Routing Decision:")
    print(f"  Model: {decision.model.value}")
    print(f"  Reason: {decision.reason}")
    print(f"  Estimated tokens: {decision.estimated_tokens:,}")
    
    # Query the model
    messages = [
        {"role": "system", "content": "You are a clinical trial data analyst."},
        {"role": "user", "content": query}
    ]
    
    response = client.query(
        messages=messages,
        model=decision.model,
        max_tokens=500,
        temperature=0.7
    )
    
    print(f"\n💬 Response:")
    print(f"  {response.content[:200]}...")
    print(f"\n📈 Token Usage:")
    print(f"  Prompt: {response.usage['prompt_tokens']}")
    print(f"  Completion: {response.usage['completion_tokens']}")
    print(f"  Total: {response.usage['total_tokens']}")


def example_2_visual_query_with_image():
    """Example 2: Query with image → Routes to GPT-5-mini (multimodal)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Visual Query with Image")
    print("="*80)
    
    router = get_router()
    client = get_llm_client()
    
    query = "Analyze the efficacy data shown in this figure."
    image_path = Path("data/processing/figures/NCT02826161/Prot_000/Prot_000_figure_05.png")
    
    if not image_path.exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   Skipping this example...")
        return
    
    # Route the query with image
    decision = router.route(
        query_text=query,
        images=[image_path]
    )
    
    print(f"\n📊 Routing Decision:")
    print(f"  Model: {decision.model.value}")
    print(f"  Reason: {decision.reason}")
    print(f"  Estimated tokens: {decision.estimated_tokens:,}")
    print(f"  Has images: {decision.has_images}")
    
    # Query with multimodal input
    response = client.query_with_images(
        text=query,
        images=[image_path],
        model=decision.model,
        system_prompt="You are a clinical trial analyst specializing in figure interpretation.",
        max_tokens=2000
    )
    
    print(f"\n💬 Response:")
    print(f"  {response.content[:300]}...")
    print(f"\n📈 Token Usage:")
    print(f"  Total: {response.usage['total_tokens']:,} tokens")


def example_3_visual_reasoning_without_image():
    """Example 3: Visual reasoning keywords → Routes to GPT-5-mini."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Visual Reasoning Query (Keywords)")
    print("="*80)
    
    router = get_router()
    
    query = "Compare the Kaplan-Meier survival curves between treatment and control groups."
    
    # Route query (no image, but visual keywords present)
    decision = router.route(query_text=query)
    
    print(f"\n📊 Routing Decision:")
    print(f"  Model: {decision.model.value}")
    print(f"  Reason: {decision.reason}")
    print(f"  Metadata: {decision.metadata}")
    
    print(f"\n💡 Note: Query contains visual keywords ('Kaplan-Meier', 'curves')")
    print(f"   Router suggests GPT-5-mini for advanced reasoning capability.")


def example_4_large_context():
    """Example 4: Large context → Routes to GPT-4.1 (1M token window)."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Large Context Query")
    print("="*80)
    
    router = get_router()
    
    # Simulate large context documents
    large_docs = ["x" * 200_000 for _ in range(3)]  # ~450K tokens
    
    query = "Summarize the key findings across all three protocols."
    
    # Route with large context
    decision = router.route(
        query_text=query,
        context_docs=large_docs
    )
    
    print(f"\n📊 Routing Decision:")
    print(f"  Model: {decision.model.value}")
    print(f"  Reason: {decision.reason}")
    print(f"  Estimated tokens: {decision.estimated_tokens:,}")
    
    print(f"\n💡 Note: Context exceeds GPT-5-mini's 400K limit.")
    print(f"   Only GPT-4.1 (1M tokens) can handle this query.")


def example_5_advanced_reasoning():
    """Example 5: Advanced reasoning keywords → Routes to GPT-5-mini."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Advanced Reasoning Query")
    print("="*80)
    
    router = get_router()
    
    query = """
    Analyze step by step why the trial failed to meet its primary endpoint.
    Compare the baseline characteristics and explain the rationale for
    the observed efficacy differences.
    """
    
    # Route query with reasoning keywords
    decision = router.route(query_text=query)
    
    print(f"\n📊 Routing Decision:")
    print(f"  Model: {decision.model.value}")
    print(f"  Reason: {decision.reason}")
    print(f"  Metadata: {decision.metadata}")
    
    print(f"\n💡 Note: Query contains reasoning keywords:")
    print(f"   'analyze step by step', 'compare', 'explain', 'rationale'")
    print(f"   GPT-5-mini's reasoning capability is optimal here.")


def example_6_token_estimation():
    """Example 6: Accurate token counting with tiktoken."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Token Estimation")
    print("="*80)
    
    query = "What are the inclusion and exclusion criteria for this trial?"
    context = ["Protocol document content..."] * 100
    
    # Estimate tokens
    estimate = estimate_query_tokens(
        query_text=query,
        context_docs=context,
        model="gpt-4.1"
    )
    
    print(f"\n📊 Token Breakdown:")
    print(f"  Query tokens: {estimate['query_tokens']:,}")
    print(f"  Context tokens: {estimate['context_tokens']:,}")
    print(f"  Image tokens: {estimate['image_tokens']:,}")
    print(f"  Overhead: {estimate['overhead_tokens']:,}")
    print(f"  Total: {estimate['total_tokens']:,}")
    
    print(f"\n✅ Fits in GPT-4.1? {estimate['total_tokens'] <= 1_000_000}")
    print(f"✅ Fits in GPT-5-mini? {estimate['total_tokens'] <= 350_000}")


def show_routing_statistics():
    """Show routing statistics after all examples."""
    print("\n" + "="*80)
    print("ROUTING STATISTICS")
    print("="*80)
    
    router = get_router()
    stats = router.get_routing_stats()
    
    print(f"\n📊 Total Queries: {stats['total_queries']}")
    print(f"\n🤖 Model Distribution:")
    print(f"  GPT-4.1: {stats['gpt_4_1_count']} ({stats['gpt_4_1_percentage']:.1f}%)")
    print(f"  GPT-5-mini: {stats['gpt_5_mini_count']} ({stats['gpt_5_mini_percentage']:.1f}%)")
    
    print(f"\n📋 Routing Reasons:")
    for reason, count in stats['routing_reasons'].items():
        print(f"  {reason}: {count}")
    
    print(f"\n📈 Average Context: {stats['avg_tokens']:,} tokens")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("MODEL ROUTER EXAMPLES FOR CLINICAL TRIAL QUERIES")
    print("="*80)
    
    try:
        # Run examples
        example_1_simple_text_query()
        example_2_visual_query_with_image()
        example_3_visual_reasoning_without_image()
        example_4_large_context()
        example_5_advanced_reasoning()
        example_6_token_estimation()
        
        # Show statistics
        show_routing_statistics()
        
        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
