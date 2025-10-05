"""
Test AGE graph multi-hop traversal.

Verifies that AGE Cypher queries work for finding related entities.
"""

import psycopg
import json
import re
from typing import List, Tuple

def test_basic_cypher():
    """Test basic AGE Cypher query"""
    print("=" * 60)
    print("TEST 1: Basic AGE Cypher Query")
    print("=" * 60)
    
    conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
    cur = conn.cursor()
    
    cur.execute("LOAD 'age'")
    cur.execute("SET search_path = ag_catalog, '$user', public")
    
    # Count nodes and edges
    cur.execute("""
        SELECT * FROM ag_catalog.cypher('clinical_graph', $$
            MATCH (n:Entity)
            RETURN count(n) as cnt
        $$) as (cnt agtype)
    """)
    node_count = int(str(cur.fetchone()[0]))
    
    cur.execute("""
        SELECT * FROM ag_catalog.cypher('clinical_graph', $$
            MATCH ()-[r:RELATES_TO]->()
            RETURN count(r) as cnt
        $$) as (cnt agtype)
    """)
    edge_count = int(str(cur.fetchone()[0]))
    
    print(f"✅ Graph has {node_count} nodes and {edge_count} edges")
    
    conn.close()
    return node_count > 0 and edge_count > 0


def test_find_entity():
    """Test finding a specific entity in the graph"""
    print("\n" + "=" * 60)
    print("TEST 2: Find Specific Entity (Afatinib - most connected)")
    print("=" * 60)
    
    conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
    cur = conn.cursor()
    
    # First find the entity_id from PostgreSQL (most connected entity)
    cur.execute("""
        SELECT e.entity_text, e.entity_type, e.entity_id, COUNT(*) as relation_count
        FROM docintel.entities e
        JOIN docintel.relations r ON (e.entity_id = r.subject_entity_id OR e.entity_id = r.object_entity_id)
        GROUP BY e.entity_id, e.entity_text, e.entity_type
        ORDER BY relation_count DESC
        LIMIT 1
    """)
    
    result = cur.fetchone()
    
    if result:
        entity_text, entity_type, entity_id, relation_count = result
        entity_id_str = str(entity_id)
        
        print(f"✅ Found most connected entity:")
        print(f"   - {entity_text} ({entity_type})")
        print(f"   - Relations: {relation_count}")
        print(f"   - Entity ID: {entity_id_str[:50]}...")
        
        # Verify it exists in AGE graph
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, '$user', public")
        
        cur.execute(f"""
            SELECT * FROM ag_catalog.cypher('clinical_graph', $$
                MATCH (n:Entity)
                WHERE n.entity_id = '{entity_id_str}'
                RETURN n.entity_text as entity_text
            $$) as (entity_text agtype)
        """)
        
        age_result = cur.fetchone()
        if age_result:
            print(f"   - ✅ Confirmed in AGE graph")
        else:
            print(f"   - ⚠️  Not found in AGE graph (sync issue)")
        
        conn.close()
        return entity_id_str
    else:
        print("❌ No entities with relations found")
        conn.close()
        return None


def test_1hop_traversal(entity_id: str):
    """Test 1-hop traversal from a seed entity"""
    print("\n" + "=" * 60)
    print("TEST 3: 1-Hop Graph Traversal")
    print("=" * 60)
    
    if not entity_id:
        print("⚠️  Skipping - no seed entity")
        return False
    
    conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
    cur = conn.cursor()
    
    cur.execute("LOAD 'age'")
    cur.execute("SET search_path = ag_catalog, '$user', public")
    
    # 1-hop traversal
    cypher = f"""
        MATCH (start:Entity)-[r:RELATES_TO]->(target:Entity)
        WHERE start.entity_id = '{entity_id}'
        RETURN 
            target.entity_text as entity_text,
            target.entity_type as entity_type,
            r.predicate as predicate,
            r.confidence as confidence
        LIMIT 20
    """
    
    try:
        # Execute Cypher query directly (AGE doesn't support parameterized Cypher)
        cur.execute(f"""
            SELECT * FROM ag_catalog.cypher('clinical_graph', $$
            {cypher}
            $$) as (entity_text agtype, entity_type agtype, predicate agtype, confidence agtype)
        """)
        
        results = cur.fetchall()
        
        if results:
            print(f"✅ Found {len(results)} entities 1-hop away:")
            for entity_text, entity_type, predicate, confidence in results[:10]:
                # Strip agtype quotes
                text = str(entity_text).strip('"')
                etype = str(entity_type).strip('"')
                pred = str(predicate).strip('"')
                conf = str(confidence)
                print(f"   - {text} ({etype}) via '{pred}' [confidence: {conf}]")
            
            conn.close()
            return True
        else:
            print("⚠️  No 1-hop neighbors found (entity may be isolated)")
            conn.close()
            return False
    
    except Exception as e:
        print(f"❌ Error during 1-hop traversal: {e}")
        conn.close()
        return False


def test_multihop_traversal(entity_id: str):
    """Test multi-hop traversal (1-2 hops)"""
    print("\n" + "=" * 60)
    print("TEST 4: Multi-Hop Graph Traversal (1-2 hops)")
    print("=" * 60)
    
    if not entity_id:
        print("⚠️  Skipping - no seed entity")
        return False
    
    conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
    cur = conn.cursor()
    
    cur.execute("LOAD 'age'")
    cur.execute("SET search_path = ag_catalog, '$user', public")
    
    # Multi-hop traversal with path info (AGE doesn't support list comprehension)
    cypher = f"""
        MATCH path = (start:Entity)-[r:RELATES_TO*1..2]->(target:Entity)
        WHERE start.entity_id = '{entity_id}'
        RETURN 
            target.entity_text as entity_text,
            target.entity_type as entity_type,
            length(path) as hop_distance,
            relationships(path) as predicates
        LIMIT 30
    """
    
    try:
        # Execute Cypher query directly (AGE doesn't support parameterized Cypher)
        cur.execute(f"""
            SELECT * FROM ag_catalog.cypher('clinical_graph', $$
            {cypher}
            $$) as (entity_text agtype, entity_type agtype, hop_distance agtype, predicates agtype)
        """)
        
        results = cur.fetchall()
        
        if results:
            print(f"✅ Found {len(results)} entities within 2 hops:")
            
            hop1_count = sum(1 for r in results if '1' in str(r[2]))
            hop2_count = sum(1 for r in results if '2' in str(r[2]))
            print(f"   • 1-hop: {hop1_count} entities")
            print(f"   • 2-hop: {hop2_count} entities")
            print()
            
            # Show sample 2-hop results
            print("Sample 2-hop paths:")
            for entity_text, entity_type, hop_dist, predicates in results[:10]:
                if '2' in str(hop_dist):
                    text = str(entity_text).strip('"')
                    etype = str(entity_type).strip('"')
                    
                    # Parse predicates from agtype array
                    try:
                        preds_str = str(predicates)
                        preds = re.findall(r'"([^"]+)"', preds_str)
                        path = ' → '.join(preds) if preds else 'related'
                    except:
                        path = 'related'
                    
                    print(f"   - {text} ({etype}) via path: {path}")
            
            conn.close()
            return True
        else:
            print("⚠️  No multi-hop neighbors found")
            conn.close()
            return False
    
    except Exception as e:
        print(f"❌ Error during multi-hop traversal: {e}")
        conn.close()
        return False


def test_relation_types():
    """Test distribution of relation types in the graph"""
    print("\n" + "=" * 60)
    print("TEST 5: Relation Type Distribution")
    print("=" * 60)
    
    conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
    cur = conn.cursor()
    
    cur.execute("LOAD 'age'")
    cur.execute("SET search_path = ag_catalog, '$user', public")
    
    # Count relations by predicate type
    cur.execute("""
        SELECT * FROM ag_catalog.cypher('clinical_graph', $$
            MATCH ()-[r:RELATES_TO]->()
            RETURN r.predicate as predicate, count(*) as cnt
        $$) as (predicate agtype, cnt agtype)
    """)
    
    results = cur.fetchall()
    
    if results:
        print(f"✅ Relation type distribution:")
        # Parse and sort
        relation_counts = []
        for predicate, cnt in results:
            pred = str(predicate).strip('"')
            count = int(str(cnt))
            relation_counts.append((pred, count))
        
        relation_counts.sort(key=lambda x: x[1], reverse=True)
        
        for pred, count in relation_counts[:15]:
            print(f"   - {pred}: {count:,} relations")
        
        conn.close()
        return True
    else:
        print("❌ No relations found")
        conn.close()
        return False


def main():
    """Run all AGE graph traversal tests"""
    print("\n" + "🧪" * 30)
    print("AGE GRAPH TRAVERSAL TEST SUITE")
    print("🧪" * 30)
    
    results = {}
    
    # Test 1: Basic connectivity
    results['basic'] = test_basic_cypher()
    
    # Test 2: Find entity
    entity_id = test_find_entity()
    results['find_entity'] = entity_id is not None
    
    # Test 3: 1-hop traversal
    results['1hop'] = test_1hop_traversal(entity_id) if entity_id else False
    
    # Test 4: Multi-hop traversal
    results['multihop'] = test_multihop_traversal(entity_id) if entity_id else False
    
    # Test 5: Relation types
    results['relation_types'] = test_relation_types()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All AGE graph traversal tests passed!")
        print("   The graph is ready for U-Retrieval integration.")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
