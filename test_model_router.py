#!/usr/bin/env python3
"""
Test script for model router functionality.

Validates routing logic, token counting, and API parameter handling
without making actual API calls (uses mock responses).

Run with:
    pixi run python test_model_router.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docintel.query.model_router import ModelRouter, ModelChoice, RoutingDecision
from docintel.query.token_utils import TokenCounter, estimate_query_tokens


def test_routing_simple_text():
    """Test 1: Simple text query should route to GPT-4.1 (default)."""
    print("\n" + "="*80)
    print("TEST 1: Simple Text Query Routing")
    print("="*80)
    
    router = ModelRouter()
    
    query = "What is the primary endpoint of this clinical trial?"
    decision = router.route(query_text=query)
    
    assert decision.model == ModelChoice.GPT_4_1, f"Expected GPT-4.1, got {decision.model}"
    assert decision.reason == "default_text_query", f"Expected default_text_query, got {decision.reason}"
    assert not decision.has_images, "Should not have images"
    
    print(f"✅ PASS: Routes to {decision.model.value}")
    print(f"   Reason: {decision.reason}")
    print(f"   Estimated tokens: {decision.estimated_tokens:,}")


def test_routing_with_images():
    """Test 2: Query with images should route to GPT-5-mini."""
    print("\n" + "="*80)
    print("TEST 2: Query with Images Routing")
    print("="*80)
    
    router = ModelRouter()
    
    query = "Analyze this figure."
    images = [Path("test_image.png")]
    decision = router.route(query_text=query, images=images)
    
    assert decision.model == ModelChoice.GPT_5_MINI, f"Expected GPT-5-mini, got {decision.model}"
    assert decision.reason == "multimodal_query_with_images", f"Unexpected reason: {decision.reason}"
    assert decision.has_images, "Should have images"
    
    print(f"✅ PASS: Routes to {decision.model.value}")
    print(f"   Reason: {decision.reason}")
    print(f"   Image count: {decision.metadata['image_count']}")


def test_routing_large_context():
    """Test 3: Simple text query routes to GPT-4.1 (default)."""
    print("\n" + "="*80)
    print("TEST 3: Text Query Without Special Keywords")
    print("="*80)
    
    router = ModelRouter()
    
    query = "Summarize these documents."
    decision = router.route(query_text=query)
    
    assert decision.model == ModelChoice.GPT_4_1, f"Expected GPT-4.1, got {decision.model}"
    assert decision.reason == "default_text_query", f"Unexpected reason: {decision.reason}"
    
    print(f"✅ PASS: Routes to {decision.model.value}")
    print(f"   Reason: {decision.reason}")


def test_routing_visual_keywords():
    """Test 4: Visual keywords should route to GPT-5-mini."""
    print("\n" + "="*80)
    print("TEST 4: Visual Keywords Routing")
    print("="*80)
    
    router = ModelRouter()
    
    queries = [
        "What does Figure 3 show?",
        "Analyze the Kaplan-Meier survival curve.",
        "Compare the bar chart in the efficacy section.",
        "Describe the flowchart diagram."
    ]
    
    for query in queries:
        decision = router.route(query_text=query)
        
        assert decision.model == ModelChoice.GPT_5_MINI, \
            f"Expected GPT-5-mini for '{query}', got {decision.model}"
        assert decision.reason == "visual_reasoning_query", \
            f"Unexpected reason: {decision.reason}"
        
        print(f"✅ PASS: '{query[:50]}...'")
        print(f"   Routes to {decision.model.value} ({decision.reason})")


def test_routing_reasoning_keywords():
    """Test 5: Reasoning keywords should route to GPT-5-mini."""
    print("\n" + "="*80)
    print("TEST 5: Reasoning Keywords Routing")
    print("="*80)
    
    router = ModelRouter()
    
    queries = [
        "Analyze step by step why the trial failed.",
        "Compare the baseline characteristics and justify the differences.",
        "Explain the rationale for the observed efficacy.",
        "Provide evidence for this hypothesis."
    ]
    
    for query in queries:
        decision = router.route(query_text=query)
        
        assert decision.model == ModelChoice.GPT_5_MINI, \
            f"Expected GPT-5-mini for '{query}', got {decision.model}"
        assert decision.reason == "advanced_reasoning_query", \
            f"Unexpected reason: {decision.reason}"
        
        print(f"✅ PASS: '{query[:50]}...'")
        print(f"   Routes to {decision.model.value} ({decision.reason})")


def test_routing_forced_override():
    """Test 6: Force model override should respect user choice."""
    print("\n" + "="*80)
    print("TEST 6: Forced Model Override")
    print("="*80)
    
    router = ModelRouter()
    
    # Query that would normally go to GPT-4.1
    query = "Simple text query"
    
    # Force it to use GPT-5-mini
    decision = router.route(query_text=query, force_model=ModelChoice.GPT_5_MINI)
    
    assert decision.model == ModelChoice.GPT_5_MINI, f"Expected GPT-5-mini, got {decision.model}"
    assert decision.reason == "forced_override", f"Unexpected reason: {decision.reason}"
    assert decision.metadata.get("forced") == True, "Should have forced flag"
    
    print(f"✅ PASS: Forced override to {decision.model.value}")
    print(f"   Reason: {decision.reason}")


def test_token_counting_text():
    """Test 7: Token counting for text."""
    print("\n" + "="*80)
    print("TEST 7: Text Token Counting")
    print("="*80)
    
    counter = TokenCounter()
    
    # Test simple text
    text = "Hello, world!"
    tokens = counter.count_text_tokens(text)
    
    # With tiktoken: ~3 tokens, without: ~3 tokens (13 chars / 4)
    assert 2 <= tokens <= 5, f"Unexpected token count: {tokens}"
    
    print(f"✅ PASS: '{text}' → {tokens} tokens")
    
    # Test longer text
    long_text = "This is a longer sentence that should have more tokens." * 10
    long_tokens = counter.count_text_tokens(long_text)
    
    assert long_tokens > tokens, "Longer text should have more tokens"
    
    print(f"✅ PASS: Longer text ({len(long_text)} chars) → {long_tokens} tokens")


def test_token_counting_messages():
    """Test 8: Token counting for chat messages."""
    print("\n" + "="*80)
    print("TEST 8: Message Token Counting")
    print("="*80)
    
    counter = TokenCounter()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the primary endpoint?"}
    ]
    
    tokens = counter.count_message_tokens(messages)
    
    # Should include message overhead + role tokens + content tokens
    assert tokens > 10, f"Message tokens too low: {tokens}"
    
    print(f"✅ PASS: Messages → {tokens} tokens")
    print(f"   (includes message formatting overhead)")


def test_token_counting_images():
    """Test 9: Token counting for images."""
    print("\n" + "="*80)
    print("TEST 9: Image Token Counting")
    print("="*80)
    
    counter = TokenCounter()
    
    # Test with 1 image
    image_tokens = counter.count_image_tokens(1)
    assert image_tokens == 1000, f"Expected 1000 tokens per image, got {image_tokens}"
    
    print(f"✅ PASS: 1 image → {image_tokens} tokens")
    
    # Test with multiple images
    multi_image_tokens = counter.count_image_tokens(5)
    assert multi_image_tokens == 5000, f"Expected 5000 tokens, got {multi_image_tokens}"
    
    print(f"✅ PASS: 5 images → {multi_image_tokens} tokens")


def test_context_estimation():
    """Test 10: Full context size estimation."""
    print("\n" + "="*80)
    print("TEST 10: Context Size Estimation")
    print("="*80)
    
    query = "What are the inclusion criteria?"
    context_docs = ["Document 1 content"] * 10
    images = [Path("image1.png"), Path("image2.png")]
    system_prompt = "You are a clinical trial analyst."
    
    estimate = estimate_query_tokens(
        query_text=query,
        images=images,
        context_docs=context_docs,
        model="gpt-4.1"
    )
    
    assert estimate["total_tokens"] > 0, "Total tokens should be positive"
    assert estimate["query_tokens"] > 0, "Query tokens should be positive"
    assert estimate["context_tokens"] > 0, "Context tokens should be positive"
    assert estimate["image_tokens"] == 2000, "Should have 2000 tokens for 2 images"
    assert estimate["image_count"] == 2, "Should count 2 images"
    assert estimate["context_doc_count"] == 10, "Should count 10 context docs"
    
    print(f"✅ PASS: Context estimation breakdown:")
    print(f"   Query: {estimate['query_tokens']} tokens")
    print(f"   Context: {estimate['context_tokens']} tokens")
    print(f"   Images: {estimate['image_tokens']} tokens (2 images)")
    print(f"   Total: {estimate['total_tokens']:,} tokens")


def test_context_fit_checks():
    """Test 11: Context window fit checking."""
    print("\n" + "="*80)
    print("TEST 11: Context Window Fit Checks")
    print("="*80)
    
    counter = TokenCounter()
    
    # Test small context (fits in both models)
    small_tokens = 10_000
    assert counter.fits_in_context(small_tokens, "gpt-4.1"), "Should fit in GPT-4.1"
    assert counter.fits_in_context(small_tokens, "gpt-5-mini"), "Should fit in GPT-5-mini"
    print(f"✅ PASS: {small_tokens:,} tokens fits in both models")
    
    # Test medium context (fits in GPT-4.1 only)
    medium_tokens = 500_000
    assert counter.fits_in_context(medium_tokens, "gpt-4.1"), "Should fit in GPT-4.1"
    assert not counter.fits_in_context(medium_tokens, "gpt-5-mini"), "Should NOT fit in GPT-5-mini"
    print(f"✅ PASS: {medium_tokens:,} tokens fits in GPT-4.1 only")
    
    # Test huge context (doesn't fit anywhere)
    huge_tokens = 1_200_000
    assert not counter.fits_in_context(huge_tokens, "gpt-4.1"), "Should NOT fit in GPT-4.1"
    assert not counter.fits_in_context(huge_tokens, "gpt-5-mini"), "Should NOT fit in GPT-5-mini"
    print(f"✅ PASS: {huge_tokens:,} tokens exceeds all limits")


def test_routing_statistics():
    """Test 12: Routing statistics tracking."""
    print("\n" + "="*80)
    print("TEST 12: Routing Statistics")
    print("="*80)
    
    router = ModelRouter()
    
    # Generate various routing decisions
    router.route("Simple query 1")
    router.route("Simple query 2")
    router.route("What does Figure 1 show?")
    router.route("Analyze this chart", images=[Path("test.png")])
    router.route("Compare the results step by step")
    
    stats = router.get_routing_stats()
    
    assert stats["total_queries"] == 5, f"Expected 5 queries, got {stats['total_queries']}"
    assert stats["gpt_4_1_count"] == 2, f"Expected 2 GPT-4.1 queries, got {stats['gpt_4_1_count']}"
    assert stats["gpt_5_mini_count"] == 3, f"Expected 3 GPT-5-mini queries, got {stats['gpt_5_mini_count']}"
    
    print(f"✅ PASS: Statistics tracking works")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   GPT-4.1: {stats['gpt_4_1_count']} ({stats['gpt_4_1_percentage']:.1f}%)")
    print(f"   GPT-5-mini: {stats['gpt_5_mini_count']} ({stats['gpt_5_mini_percentage']:.1f}%)")
    print(f"   Routing reasons: {stats['routing_reasons']}")


def test_edge_case_empty_inputs():
    """Test 13: Edge case with empty inputs."""
    print("\n" + "="*80)
    print("TEST 13: Edge Cases - Empty Inputs")
    print("="*80)
    
    router = ModelRouter()
    counter = TokenCounter()
    
    # Empty query
    decision = router.route(query_text="")
    assert decision.model == ModelChoice.GPT_4_1, "Empty query should default to GPT-4.1"
    print(f"✅ PASS: Empty query routes to {decision.model.value}")
    
    # Empty text token counting
    tokens = counter.count_text_tokens("")
    assert tokens == 0, "Empty text should have 0 tokens"
    print(f"✅ PASS: Empty text → {tokens} tokens")
    
    # Empty messages
    tokens = counter.count_message_tokens([])
    assert tokens == 0, "Empty messages should have 0 tokens"
    print(f"✅ PASS: Empty messages → {tokens} tokens")


def test_edge_case_images_with_context():
    """Test 14: Images always route to GPT-5-mini regardless of context."""
    print("\n" + "="*80)
    print("TEST 14: Images Always Route to GPT-5-mini")
    print("="*80)
    
    router = ModelRouter()
    
    # Query with images and some context
    images = [Path("test.png")]
    
    decision = router.route(
        query_text="Analyze this figure",
        images=images,
        context_docs=["Some context"]
    )
    
    # Images always route to GPT-5-mini
    assert decision.model == ModelChoice.GPT_5_MINI, \
        "Images should always route to GPT-5-mini"
    
    print(f"✅ PASS: Images route to {decision.model.value}")
    print(f"   Model: {decision.model.value}")
    print(f"   Reason: {decision.reason}")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*80)
    print("MODEL ROUTER TEST SUITE")
    print("="*80)
    print("\nTesting routing logic, token counting, and edge cases...")
    
    tests = [
        test_routing_simple_text,
        test_routing_with_images,
        test_routing_large_context,
        test_routing_visual_keywords,
        test_routing_reasoning_keywords,
        test_routing_forced_override,
        test_token_counting_text,
        test_token_counting_messages,
        test_token_counting_images,
        test_context_estimation,
        test_context_fit_checks,
        test_routing_statistics,
        test_edge_case_empty_inputs,
        test_edge_case_images_with_context,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAIL: {test_func.__name__}")
            print(f"   {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {test_func.__name__}")
            print(f"   {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\n✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
        return 1
    else:
        print(f"\n🎉 All tests passed!")
        return 0


def main():
    """Main entry point."""
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
