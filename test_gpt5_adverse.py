#!/usr/bin/env python3
"""Compare GPT-4.1 vs GPT-5-mini on adverse effects extraction (short and long queries)."""

import asyncio
import sys
sys.path.insert(0, 'src')

from query_clinical_trials import ClinicalTrialQA
from docintel.embeddings.client import EmbeddingClient
from docintel.config import EmbeddingSettings
from docintel.query.model_router import ModelRouter
from docintel.query.llm_client import get_llm_client

QUERIES = {
    "SHORT": "any adverse reactions or side effects from niraparib?",
    "LONG": "What is niraparib and how should niraparib be administered and what are the observed adverse effects?"
}

EXPECTED_ADVERSE_EVENTS = ['Nausea', 'Fatigue', 'Thrombocytopenia', 'Anemia', 'Constipation', 'Vomiting']

async def test_model(model_name: str, query: str, query_type: str):
    """Test a model on adverse effects extraction."""
    qa = ClinicalTrialQA()
    qa.embedding_client = EmbeddingClient(EmbeddingSettings())
    
    # Set routing based on model
    if model_name == "GPT-4.1":
        qa.router = ModelRouter(disable_routing=True)  # Force GPT-4.1
    else:
        qa.router = ModelRouter(disable_routing=False)  # Allow GPT-5-mini
    
    qa.llm_client = get_llm_client()
    
    result = await qa.answer_question(query)
    answer = result['answer']
    
    # Check for adverse effects
    found_events = []
    for event in EXPECTED_ADVERSE_EVENTS:
        if event.lower() in answer.lower():
            found_events.append(event)
    
    return {
        'model': result['model_used'],
        'routing_reason': result['routing_reason'],
        'query_type': query_type,
        'found_count': len(found_events),
        'found_events': found_events,
        'total_expected': len(EXPECTED_ADVERSE_EVENTS),
        'success': len(found_events) >= 4,  # Need at least 4 out of 6
        'processing_time_ms': result['processing_time_ms']
    }

async def run_comparison():
    """Run full comparison of both models on both query types."""
    print("=" * 80)
    print("ADVERSE EFFECTS EXTRACTION: GPT-4.1 vs GPT-5-mini")
    print("=" * 80)
    print(f"\nExpected adverse events: {', '.join(EXPECTED_ADVERSE_EVENTS)}")
    print(f"Success threshold: 4+ events detected\n")
    
    results = []
    
    for query_type, query in QUERIES.items():
        print(f"\n{'='*80}")
        print(f"TESTING: {query_type} QUERY")
        print(f"Query: \"{query}\"")
        print('='*80)
        
        for model in ["GPT-4.1", "GPT-5-mini"]:
            print(f"\n🤖 Testing {model}...")
            try:
                result = await test_model(model, query, query_type)
                results.append(result)
                
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                print(f"   {status} - Found {result['found_count']}/{result['total_expected']} events")
                print(f"   Events: {', '.join(result['found_events']) if result['found_events'] else 'None'}")
                print(f"   Time: {result['processing_time_ms']:.1f}ms")
                
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                results.append({
                    'model': model,
                    'query_type': query_type,
                    'success': False,
                    'error': str(e)
                })
    
    # Summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Query Type':<12} | {'Model':<12} | {'Found':<8} | {'Status':<8} | {'Time (ms)':<10}")
    print("-" * 80)
    
    for r in results:
        if 'error' not in r:
            status = "PASS ✅" if r['success'] else "FAIL ❌"
            found = f"{r['found_count']}/{r['total_expected']}"
            time_str = f"{r['processing_time_ms']:.0f}"
            print(f"{r['query_type']:<12} | {r['model']:<12} | {found:<8} | {status:<8} | {time_str:<10}")
        else:
            print(f"{r['query_type']:<12} | {r['model']:<12} | ERROR")
    
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    
    # Analyze results
    gpt41_results = [r for r in results if r.get('model') == 'gpt-4.1' and 'error' not in r]
    gpt5_results = [r for r in results if r.get('model') == 'gpt-5-mini' and 'error' not in r]
    
    gpt41_success_rate = sum(1 for r in gpt41_results if r['success']) / len(gpt41_results) * 100 if gpt41_results else 0
    gpt5_success_rate = sum(1 for r in gpt5_results if r['success']) / len(gpt5_results) * 100 if gpt5_results else 0
    
    print(f"\nGPT-4.1 Success Rate: {gpt41_success_rate:.0f}% ({sum(1 for r in gpt41_results if r['success'])}/{len(gpt41_results)} queries)")
    print(f"GPT-5-mini Success Rate: {gpt5_success_rate:.0f}% ({sum(1 for r in gpt5_results if r['success'])}/{len(gpt5_results)} queries)")
    
    if gpt41_success_rate > gpt5_success_rate:
        print(f"\n🏆 WINNER: GPT-4.1 (better at extracting adverse events from context)")
        print("   Recommendation: Keep disable_routing=True for clinical queries")
    elif gpt5_success_rate > gpt41_success_rate:
        print(f"\n🏆 WINNER: GPT-5-mini (better at extracting adverse events)")
        print("   Recommendation: Enable routing for reasoning queries")
    else:
        print(f"\n🤝 TIE: Both models perform similarly")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_comparison())
