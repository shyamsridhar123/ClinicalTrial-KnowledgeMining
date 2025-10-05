"""
Test U-Retrieval with AGE graph expansion.

Simple sync test (non-async) to verify U-Retrieval works.
"""

import psycopg
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from docintel.knowledge_graph.u_retrieval import ClinicalURetrieval, QueryType, SearchScope

async def test_u_retrieval_basic():
    """Test basic U-Retrieval functionality"""
    print("=" * 60)
    print("TEST: U-Retrieval with AGE Graph Expansion")
    print("=" * 60)
    
    conn_string = "postgresql://dbuser:dbpass123@localhost:5432/docintel"
    u_retrieval = ClinicalURetrieval(conn_string)
    
    try:
        # Test query
        query = "What are adverse events for Afatinib?"
        print(f"\nQuery: {query}")
        print()
        
        # Perform U-Retrieval search
        result = await u_retrieval.u_retrieval_search(
            query=query,
            query_type=QueryType.HYBRID_SEARCH,
            search_scope=SearchScope.GLOBAL,
            max_results=50
        )
        
        print(f"✅ U-Retrieval completed!")
        print(f"   Total results: {result.total_results}")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        print()
        
        # Show sample results
        if result.results:
            print(f"Sample results (showing first 10):")
            for i, res in enumerate(result.results[:10], 1):
                print(f"\n{i}. {res.entity_text} ({res.entity_type})")
                print(f"   Score: {res.relevance_score:.3f}")
                print(f"   Explanation: {res.explanation}")
                if res.metadata.get('hop_distance'):
                    print(f"   Hop distance: {res.metadata['hop_distance']}")
                    print(f"   Path: {res.metadata.get('predicate_path', 'N/A')}")
        else:
            print("⚠️  No results returned")
        
        print()
        print("=" * 60)
        return result.total_results > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await u_retrieval.close()


def main():
    """Run U-Retrieval test"""
    print("\n🧪 Testing U-Retrieval with AGE Graph Expansion\n")
    
    success = asyncio.run(test_u_retrieval_basic())
    
    if success:
        print("\n🎉 U-Retrieval test passed!")
        print("   Ready for integration into query_clinical_trials.py")
    else:
        print("\n⚠️  U-Retrieval test failed - see errors above")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
