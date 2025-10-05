"""
Compare U-Retrieval with and without graph expansion.

This test verifies that HYBRID_SEARCH (with graph expansion) returns 
more results than ENTITY_SEARCH (without expansion).
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from docintel.knowledge_graph.u_retrieval import ClinicalURetrieval, QueryType, SearchScope


async def test_expansion_comparison():
    """Compare entity search vs hybrid search (with graph expansion)"""
    print("=" * 70)
    print("TEST: Graph Expansion Comparison")
    print("=" * 70)
    
    conn_string = "postgresql://dbuser:dbpass123@localhost:5432/docintel"
    u_retrieval = ClinicalURetrieval(conn_string)
    
    try:
        query = "adverse events"
        print(f"\nQuery: '{query}'")
        print()
        
        # Test 1: Entity search only (no graph expansion)
        print("Test 1: ENTITY_SEARCH (no graph expansion)")
        print("-" * 70)
        result_entity = await u_retrieval.u_retrieval_search(
            query=query,
            query_type=QueryType.ENTITY_SEARCH,
            search_scope=SearchScope.GLOBAL,
            max_results=50
        )
        print(f"Results: {result_entity.total_results}")
        print(f"Time: {result_entity.processing_time_ms:.1f}ms")
        
        # Count graph-expanded results (should be 0)
        graph_expanded = sum(1 for r in result_entity.results 
                           if r.metadata.get('relation_type') == 'graph_expansion')
        print(f"Graph-expanded results: {graph_expanded}")
        
        # Test 2: Hybrid search (with graph expansion)
        print()
        print("Test 2: HYBRID_SEARCH (with graph expansion)")
        print("-" * 70)
        result_hybrid = await u_retrieval.u_retrieval_search(
            query=query,
            query_type=QueryType.HYBRID_SEARCH,
            search_scope=SearchScope.GLOBAL,
            max_results=50
        )
        print(f"Results: {result_hybrid.total_results}")
        print(f"Time: {result_hybrid.processing_time_ms:.1f}ms")
        
        # Count graph-expanded results
        graph_expanded = sum(1 for r in result_hybrid.results 
                           if r.metadata.get('relation_type') == 'graph_expansion')
        print(f"Graph-expanded results: {graph_expanded}")
        
        # Show sample graph-expanded results
        if graph_expanded > 0:
            print()
            print("Sample graph-expanded results:")
            count = 0
            for r in result_hybrid.results:
                if r.metadata.get('relation_type') == 'graph_expansion':
                    print(f"  - {r.entity_text} ({r.entity_type})")
                    print(f"    Hops: {r.metadata.get('hop_distance', '?')}")
                    print(f"    Path: {r.metadata.get('predicate_path', 'N/A')}")
                    count += 1
                    if count >= 5:
                        break
        
        # Analysis
        print()
        print("=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        
        if result_hybrid.total_results > result_entity.total_results:
            improvement = result_hybrid.total_results - result_entity.total_results
            print(f"✅ HYBRID search found {improvement} more results than ENTITY search")
            print(f"   Improvement: {improvement/result_entity.total_results*100:.1f}%")
        else:
            print(f"⚠️  HYBRID search did not find more results")
            print(f"   This may indicate graph expansion isn't triggering")
        
        if graph_expanded > 0:
            print(f"✅ Graph expansion is working! Found {graph_expanded} expanded entities")
            return True
        else:
            print(f"⚠️  No graph-expanded results found")
            print(f"   Possible causes:")
            print(f"   - Initial entities have no outgoing relations")
            print(f"   - AGE Cypher query isn't executing")
            print(f"   - Metadata not being set correctly")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await u_retrieval.close()


def main():
    """Run comparison test"""
    print("\n🧪 Testing Graph Expansion in U-Retrieval\n")
    
    success = asyncio.run(test_expansion_comparison())
    
    print()
    if success:
        print("🎉 Graph expansion is working correctly!")
    else:
        print("⚠️  Graph expansion may not be functioning - review output above")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
