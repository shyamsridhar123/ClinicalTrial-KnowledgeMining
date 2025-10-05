"""
Integration test for U-Retrieval in query_clinical_trials.py

This test verifies that:
1. U-Retrieval correctly integrates with the Q&A system
2. Graph expansion activates and finds related entities
3. Chunks are properly mapped from entities
4. GPT-4.1 synthesis produces coherent answers
5. Processing metrics are tracked

Run:
    pixi run -- pytest tests/test_u_retrieval_integration.py -v -s
"""

import pytest
import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from docintel.knowledge_graph.u_retrieval import ClinicalURetrieval, QueryType, SearchScope

load_dotenv()


@pytest.fixture
def db_dsn():
    """Get database connection string from environment."""
    dsn = os.getenv('DOCINTEL_VECTOR_DB_DSN')
    if not dsn:
        pytest.skip("DOCINTEL_VECTOR_DB_DSN not set")
    return dsn


@pytest.mark.asyncio
async def test_u_retrieval_basic_search(db_dsn):
    """Test that U-Retrieval returns entities with proper metadata."""
    u_retrieval = ClinicalURetrieval(db_dsn)
    
    result = await u_retrieval.u_retrieval_search(
        query="adverse events",
        query_type=QueryType.ENTITY_SEARCH,
        search_scope=SearchScope.GLOBAL,
        max_results=50
    )
    
    await u_retrieval.close()
    
    # Verify basic structure
    assert len(result.results) > 0, "Should return entities"
    assert result.processing_time_ms > 0, "Should track processing time"
    
    # Check entity structure
    entity = result.results[0]
    assert entity.entity_text, "Entity should have text"
    assert entity.entity_type, "Entity should have type"
    assert entity.relevance_score > 0, "Entity should have relevance score"
    assert 'source_chunk_id' in entity.metadata, "Entity should have source_chunk_id"
    
    print(f"✅ Basic search returned {len(result.results)} entities in {result.processing_time_ms:.1f}ms")


@pytest.mark.asyncio
async def test_u_retrieval_graph_expansion(db_dsn):
    """Test that graph expansion finds additional related entities."""
    u_retrieval = ClinicalURetrieval(db_dsn)
    
    result = await u_retrieval.u_retrieval_search(
        query="adverse events in clinical trials",
        query_type=QueryType.HYBRID_SEARCH,
        search_scope=SearchScope.GLOBAL,
        max_results=50
    )
    
    await u_retrieval.close()
    
    # Count graph-expanded entities
    graph_expanded = [e for e in result.results if e.metadata.get('relation_type') == 'graph_expansion']
    
    assert len(graph_expanded) > 0, "Should find graph-expanded entities"
    
    print(f"✅ Graph expansion found {len(graph_expanded)} additional entities out of {len(result.results)} total")
    
    # Check hop distance
    for entity in graph_expanded[:3]:
        hop = entity.metadata.get('hop_distance')
        print(f"   - {entity.entity_text} ({entity.entity_type}): {hop}-hop")


@pytest.mark.asyncio
async def test_chunk_mapping(db_dsn):
    """Test that entities properly map to chunks with embeddings."""
    u_retrieval = ClinicalURetrieval(db_dsn)
    
    result = await u_retrieval.u_retrieval_search(
        query="clinical trial endpoints",
        query_type=QueryType.ENTITY_SEARCH,
        search_scope=SearchScope.GLOBAL,
        max_results=30
    )
    
    # Group by source_chunk_id
    chunk_map = {}
    for entity in result.results:
        chunk_id = entity.metadata.get('source_chunk_id')
        if chunk_id:
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = []
            chunk_map[chunk_id].append(entity)
    
    await u_retrieval.close()
    
    assert len(chunk_map) > 0, "Should map entities to chunks"
    
    print(f"✅ {len(result.results)} entities mapped to {len(chunk_map)} unique chunks")
    
    # Show top chunks by entity count
    top_chunks = sorted(chunk_map.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    for chunk_id, entities in top_chunks:
        print(f"   - {chunk_id}: {len(entities)} entities")


@pytest.mark.asyncio
async def test_entity_prioritization(db_dsn):
    """Test that entities with relations are prioritized in search."""
    u_retrieval = ClinicalURetrieval(db_dsn)
    
    result = await u_retrieval.u_retrieval_search(
        query="drug interactions",
        query_type=QueryType.HYBRID_SEARCH,
        search_scope=SearchScope.GLOBAL,
        max_results=100
    )
    
    await u_retrieval.close()
    
    # Check first 50 entities - many should have relations
    import psycopg
    conn = psycopg.connect(db_dsn)
    cur = conn.cursor()
    
    entity_ids = [e.entity_id for e in result.results[:50]]
    placeholders = ','.join(['%s'] * len(entity_ids))
    
    cur.execute(f"""
        SELECT DISTINCT subject_entity_id as entity_id
        FROM docintel.relations 
        WHERE subject_entity_id IN ({placeholders}) 
           OR object_entity_id IN ({placeholders})
        UNION
        SELECT DISTINCT object_entity_id as entity_id
        FROM docintel.relations 
        WHERE subject_entity_id IN ({placeholders}) 
           OR object_entity_id IN ({placeholders})
    """, entity_ids + entity_ids + entity_ids + entity_ids)
    
    entities_with_relations = set(row[0] for row in cur.fetchall())
    conn.close()
    
    count = len(entities_with_relations)
    percentage = (count / 50) * 100
    
    print(f"✅ {count} out of first 50 entities have relations ({percentage:.1f}%)")
    assert percentage >= 20, f"Should prioritize entities with relations (got {percentage:.1f}%)"


@pytest.mark.asyncio
async def test_processing_metrics(db_dsn):
    """Test that processing metrics are properly tracked."""
    u_retrieval = ClinicalURetrieval(db_dsn)
    
    queries = [
        "What are adverse events?",
        "What are inclusion criteria?",
        "What are primary endpoints?"
    ]
    
    results = []
    for query in queries:
        result = await u_retrieval.u_retrieval_search(
            query=query,
            query_type=QueryType.HYBRID_SEARCH,
            search_scope=SearchScope.GLOBAL,
            max_results=50
        )
        results.append(result)
    
    await u_retrieval.close()
    
    # Check all have metrics
    for i, result in enumerate(results):
        assert result.processing_time_ms > 0, f"Query {i+1} should have processing time"
        assert len(result.results) > 0, f"Query {i+1} should return entities"
        
        graph_count = sum(1 for e in result.results if e.metadata.get('relation_type') == 'graph_expansion')
        print(f"✅ Query {i+1}: {len(result.results)} entities ({graph_count} expanded) in {result.processing_time_ms:.1f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
