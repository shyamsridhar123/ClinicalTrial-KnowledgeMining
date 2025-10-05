import asyncio
import os
from dotenv import load_dotenv
from docintel.knowledge_graph.u_retrieval import ClinicalURetrieval, QueryType, SearchScope

load_dotenv()

async def main():
    dsn = os.getenv('DOCINTEL_VECTOR_DB_DSN')
    
    u_retrieval = ClinicalURetrieval(dsn)
    
    result = await u_retrieval.u_retrieval_search(
        query="What are adverse events for Afatinib?",
        query_type=QueryType.HYBRID_SEARCH,
        search_scope=SearchScope.GLOBAL,
        max_results=50
    )
    
    print(f"\n✅ Found {len(result.results)} entities")
    print(f"   Graph expanded: {sum(1 for e in result.results if e.metadata.get('relation_type') == 'graph_expansion')}")
    print(f"   Processing time: {result.processing_time_ms:.1f}ms\n")
    
    # Check first 10 entities for chunk_ids
    print("First 10 entities:")
    for i, entity in enumerate(result.results[:10], 1):
        print(f"  [{i}] {entity.entity_text} ({entity.entity_type})")
        print(f"      chunk_id: {entity.chunk_id}")
        print(f"      metadata: {entity.metadata}")
    
    await u_retrieval.close()

if __name__ == "__main__":
    asyncio.run(main())
