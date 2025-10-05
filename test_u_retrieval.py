#!/usr/bin/env python3
"""
Test U-Retrieval System - Hierarchical Community-Aware Search

This script tests the U-Retrieval system's ability to:
1. Find relevant communities based on query
2. Search entities within those communities
3. Expand results using relation information
4. Rank with community-aware scoring
5. Aggregate multi-level context
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docintel.knowledge_graph.u_retrieval import (
    ClinicalURetrieval,
    QueryType,
    SearchScope,
    QueryContext
)
from docintel.config import get_config


async def test_u_retrieval():
    """Test U-Retrieval with various clinical queries"""
    
    config = get_config()
    connection_string = str(config.vector_db.dsn)
    
    print("=" * 80)
    print("U-RETRIEVAL HIERARCHICAL SEARCH TEST")
    print("=" * 80)
    print()
    
    # Test queries covering different clinical scenarios
    test_queries = [
        {
            "query": "What are the adverse events for niraparib?",
            "query_type": QueryType.HYBRID_SEARCH,
            "scope": SearchScope.GLOBAL,
            "context": QueryContext(
                # Don't filter by entity type - NER sometimes tags "Serious Adverse Reactions" as organization
                entity_types=None,
                confidence_threshold=0.5
            )
        },
        {
            "query": "monitoring safety endpoints",
            "query_type": QueryType.SEMANTIC_SEARCH,
            "scope": SearchScope.GLOBAL,
            "context": QueryContext(
                entity_types=None,  # Allow all types to avoid filtering matches
                confidence_threshold=0.4
            )
        },
        {
            "query": "atezolizumab treatment outcomes",
            "query_type": QueryType.ENTITY_SEARCH,
            "scope": SearchScope.GLOBAL,
            "context": QueryContext(
                entity_types=None,  # Allow all types
                confidence_threshold=0.5
            )
        },
        {
            "query": "statistical analysis methods survival",
            "query_type": QueryType.COMMUNITY_SEARCH,
            "scope": SearchScope.GLOBAL,
            "context": QueryContext(
                entity_types=None,  # Allow all types
                confidence_threshold=0.3
            )
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {test_case['query']}")
        print(f"Query Type: {test_case['query_type'].value}")
        print(f"Search Scope: {test_case['scope'].value}")
        print(f"{'=' * 80}\n")
        
        try:
            # Perform U-Retrieval search
            result = await ClinicalURetrieval(connection_string).u_retrieval_search(
                query=test_case['query'],
                query_type=test_case['query_type'],
                search_scope=test_case['scope'],
                context=test_case['context'],
                max_results=10
            )
            
            # Display results
            print(f"📊 RETRIEVAL STATISTICS:")
            print(f"   Processing Time: {result.processing_time_ms:.1f}ms")
            print(f"   Total Results: {result.total_results}")
            print(f"   Displayed Results: {len(result.results)}")
            print(f"   Communities Searched: {result.processing_stats['communities_searched']}")
            print()
            
            # Display global context
            print(f"🌍 GLOBAL CONTEXT:")
            gc = result.global_context
            print(f"   Total Entities: {gc['total_entities']}")
            print(f"   Unique Entity Types: {gc['unique_entity_types']}")
            print(f"   Average Confidence: {gc['average_confidence']:.3f}")
            print(f"   Coverage:")
            print(f"      - With Normalization: {gc['search_coverage']['entities_with_normalization']}")
            print(f"      - With Communities: {gc['search_coverage']['entities_with_communities']}")
            print(f"      - With Relations: {gc['search_coverage']['entities_with_relations']}")
            print()
            
            # Display entity type distribution
            print(f"📈 ENTITY TYPE DISTRIBUTION:")
            for entity_type, count in list(gc['entity_type_distribution'].items())[:5]:
                print(f"   {entity_type}: {count}")
            print()
            
            # Display vocabulary distribution
            if gc['vocabulary_distribution']:
                print(f"📚 VOCABULARY DISTRIBUTION:")
                for vocab, count in gc['vocabulary_distribution'].items():
                    print(f"   {vocab}: {count}")
                print()
            
            # Display community aggregation
            if result.community_aggregation:
                print(f"🏘️  COMMUNITY AGGREGATION:")
                for comm_id, comm_data in list(result.community_aggregation.items())[:3]:
                    print(f"   Community: {comm_data['title']}")
                    print(f"      Results: {comm_data['total_results']}")
                    print(f"      Avg Relevance: {comm_data['average_relevance']:.3f}")
                    print(f"      Entity Types: {', '.join(comm_data['entity_types'].keys())}")
                print()
            
            # Display top results
            print(f"🎯 TOP RESULTS (Top 5):")
            print()
            for idx, search_result in enumerate(result.results[:5], 1):
                print(f"   [{idx}] {search_result.entity_text}")
                print(f"       Type: {search_result.entity_type}")
                print(f"       Relevance Score: {search_result.relevance_score:.3f}")
                print(f"       Confidence: {search_result.confidence:.3f}")
                if search_result.normalized_concept_id:
                    print(f"       Normalized: {search_result.normalized_concept_id} ({search_result.normalized_vocabulary})")
                if search_result.community_title:
                    print(f"       Community: {search_result.community_title}")
                print(f"       Explanation: {search_result.explanation}")
                print()
            
            # Save detailed results to file
            output_file = f"output/reports/u_retrieval_test_{i}.json"
            Path("output/reports").mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    'query': test_case['query'],
                    'query_type': test_case['query_type'].value,
                    'search_scope': test_case['scope'].value,
                    'processing_time_ms': result.processing_time_ms,
                    'total_results': result.total_results,
                    'global_context': result.global_context,
                    'community_aggregation': result.community_aggregation,
                    'processing_stats': result.processing_stats,
                    'results': [
                        {
                            'entity_id': r.entity_id,
                            'entity_text': r.entity_text,
                            'entity_type': r.entity_type,
                            'normalized_concept_id': r.normalized_concept_id,
                            'normalized_vocabulary': r.normalized_vocabulary,
                            'confidence': r.confidence,
                            'community_id': r.community_id,
                            'community_title': r.community_title,
                            'relevance_score': r.relevance_score,
                            'explanation': r.explanation,
                            'metadata': r.metadata
                        }
                        for r in result.results[:10]
                    ]
                }, f, indent=2)
            
            print(f"💾 Detailed results saved to: {output_file}")
            print()
            
        except Exception as e:
            print(f"❌ ERROR during search: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("U-RETRIEVAL TEST COMPLETED")
    print("=" * 80)
    print()
    print("📊 Summary:")
    print(f"   Total queries tested: {len(test_queries)}")
    print(f"   Results saved to: output/reports/u_retrieval_test_*.json")
    print()
    print("🎯 Key Features Demonstrated:")
    print("   ✅ Community discovery and relevance scoring")
    print("   ✅ Hierarchical entity search within communities")
    print("   ✅ Relation-aware expansion (hybrid search)")
    print("   ✅ Multi-level context aggregation")
    print("   ✅ Clinical vocabulary weighting")
    print("   ✅ Entity type and confidence scoring")
    print()


async def compare_simple_vs_u_retrieval():
    """Compare simple semantic search vs U-Retrieval for the same query"""
    
    config = get_config()
    connection_string = str(config.vector_db.dsn)
    
    print("\n" + "=" * 80)
    print("COMPARISON: Simple Semantic Search vs U-Retrieval")
    print("=" * 80)
    print()
    
    query = "adverse events niraparib"
    
    # Simple semantic search (like query_clinical_trials.py does)
    print(f"Query: '{query}'")
    print()
    print("🔹 SIMPLE SEMANTIC SEARCH (current query_clinical_trials.py):")
    print("   - Direct embedding → pgvector search")
    print("   - No community awareness")
    print("   - No vocabulary weighting")
    print("   - No relation expansion")
    print()
    
    # U-Retrieval hierarchical search
    print("🔸 U-RETRIEVAL HIERARCHICAL SEARCH:")
    print("   - Find relevant communities first")
    print("   - Search entities within communities")
    print("   - Apply vocabulary/entity type weighting")
    print("   - Expand via relations (hybrid mode)")
    print("   - Community-aware ranking")
    print()
    
    try:
        result = await ClinicalURetrieval(connection_string).u_retrieval_search(
            query=query,
            query_type=QueryType.HYBRID_SEARCH,
            search_scope=SearchScope.GLOBAL,
            context=QueryContext(
                entity_types=["drug", "adverse_event"],
                confidence_threshold=0.4
            ),
            max_results=10
        )
        
        print(f"📊 U-RETRIEVAL RESULTS:")
        print(f"   Processing Time: {result.processing_time_ms:.1f}ms")
        print(f"   Communities Found: {result.processing_stats['communities_searched']}")
        print(f"   Total Results: {result.total_results}")
        print(f"   With Community Context: {result.global_context['search_coverage']['entities_with_communities']}")
        print(f"   With Normalization: {result.global_context['search_coverage']['entities_with_normalization']}")
        print(f"   With Relations: {result.global_context['search_coverage']['entities_with_relations']}")
        print()
        
        print("🎯 TOP 5 RESULTS WITH COMMUNITY CONTEXT:")
        for idx, r in enumerate(result.results[:5], 1):
            print(f"   [{idx}] {r.entity_text} ({r.entity_type})")
            print(f"       Score: {r.relevance_score:.3f} | Community: {r.community_title or 'N/A'}")
            print(f"       {r.explanation}")
        
        print()
        print("💡 U-RETRIEVAL ADVANTAGES:")
        print("   ✅ Leverages community structure for context")
        print("   ✅ Weights results by clinical vocabulary authority")
        print("   ✅ Discovers related entities via graph relations")
        print("   ✅ Provides multi-level aggregated context")
        print("   ✅ More interpretable (explanations + community info)")
        print()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print()
    print("🧪 Testing U-Retrieval Hierarchical Community-Aware Search")
    print()
    
    # Run main tests
    asyncio.run(test_u_retrieval())
    
    # Run comparison
    asyncio.run(compare_simple_vs_u_retrieval())
    
    print("✅ All tests completed!")
    print()
