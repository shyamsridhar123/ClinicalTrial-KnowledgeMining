#!/usr/bin/env python3
"""
Debug Community-Based Search System

Analyzes why community search fails to retrieve entities for queries like "What is niraparib?"

Usage:
    pixi run -- python scripts/debug_community_search.py "What is niraparib?"
    pixi run -- python scripts/debug_community_search.py "What are adverse events?"
"""

import sys
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psycopg
from psycopg.rows import dict_row

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class CommunitySearchDebugger:
    """Debug community-based entity retrieval"""
    
    def __init__(self, db_dsn: str):
        self.db_dsn = db_dsn
        self.conn = None
    
    async def connect(self):
        """Connect to database"""
        self.conn = await psycopg.AsyncConnection.connect(
            self.db_dsn,
            row_factory=dict_row,
            autocommit=True
        )
    
    async def close(self):
        """Close connection"""
        if self.conn:
            await self.conn.close()
    
    async def debug_query(self, query: str):
        """Run comprehensive debug analysis for a query"""
        
        print("=" * 80)
        print(f"COMMUNITY SEARCH DEBUG: {query}")
        print("=" * 80)
        print()
        
        # Step 1: Analyze query terms
        print("📋 STEP 1: Query Analysis")
        print("-" * 80)
        query_lower = query.lower()
        stop_words = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'is', 'are'}
        query_terms = [term for term in query_lower.split() if term not in stop_words and len(term) > 2]
        
        print(f"Original query: {query}")
        print(f"Lowercase: {query_lower}")
        print(f"All terms: {query_lower.split()}")
        print(f"Meaningful terms (after stop word removal): {query_terms}")
        print()
        
        # Step 2: Check communities
        print("📊 STEP 2: Community Analysis")
        print("-" * 80)
        await self._analyze_communities(query_terms)
        print()
        
        # Step 3: Check AGE node ID mapping
        print("🔗 STEP 3: AGE Node ID Mapping")
        print("-" * 80)
        await self._analyze_age_mapping()
        print()
        
        # Step 4: Check entity coverage
        print("🧬 STEP 4: Entity Coverage in Communities")
        print("-" * 80)
        await self._analyze_entity_coverage(query_terms)
        print()
        
        # Step 5: Direct entity search (baseline)
        print("🎯 STEP 5: Direct Entity Search (Baseline)")
        print("-" * 80)
        await self._direct_entity_search(query_terms)
        print()
        
        # Step 6: Simulate community search
        print("🔍 STEP 6: Simulate Community Search Algorithm")
        print("-" * 80)
        await self._simulate_community_search(query, query_terms)
        print()
        
        # Step 7: Recommendations
        print("💡 STEP 7: Recommendations")
        print("-" * 80)
        await self._generate_recommendations(query_terms)
        print()
    
    async def _analyze_communities(self, query_terms: List[str]):
        """Analyze all communities"""
        result = await self.conn.execute("""
            SELECT 
                cluster_key,
                level,
                title,
                nodes,
                edges,
                chunk_ids,
                occurrence,
                created_at
            FROM ag_catalog.communities
            ORDER BY occurrence DESC
        """)
        communities = await result.fetchall()
        
        print(f"Total communities: {len(communities)}")
        
        if not communities:
            print("❌ NO COMMUNITIES FOUND - This is a critical issue!")
            return
        
        print(f"\nTop 10 communities by occurrence:")
        for i, comm in enumerate(communities[:10], 1):
            nodes = json.loads(comm['nodes']) if isinstance(comm['nodes'], str) else comm['nodes']
            edges = json.loads(comm['edges']) if isinstance(comm['edges'], str) else comm['edges']
            chunk_ids = json.loads(comm['chunk_ids']) if isinstance(comm['chunk_ids'], str) else comm['chunk_ids']
            
            print(f"\n  {i}. Community {comm['cluster_key']} (Level {comm['level']})")
            print(f"     Title: {comm['title']}")
            print(f"     Occurrence: {comm['occurrence']:.4f}")
            print(f"     Nodes: {len(nodes)} nodes")
            print(f"     Edges: {len(edges)} edges")
            print(f"     Chunk IDs: {len(chunk_ids)} chunks")
            print(f"     Sample nodes: {nodes[:5]}")
            
            # Check title relevance
            title_lower = comm['title'].lower() if comm['title'] else ""
            title_matches = [term for term in query_terms if term in title_lower]
            if title_matches:
                print(f"     ✅ Title matches: {title_matches}")
            else:
                print(f"     ❌ Title doesn't match query terms")
    
    async def _analyze_age_mapping(self):
        """Analyze AGE node ID → meta_graph_id mapping"""
        
        # Get total meta_graphs
        result = await self.conn.execute("""
            SELECT COUNT(*) as total
            FROM docintel.meta_graphs
        """)
        row = await result.fetchone()
        total_meta_graphs = row['total']
        print(f"Total meta_graphs: {total_meta_graphs}")
        
        # Sample AGE node IDs from communities
        result = await self.conn.execute("""
            SELECT nodes FROM ag_catalog.communities LIMIT 5
        """)
        communities = await result.fetchall()
        
        print(f"\nTesting AGE node ID → meta_graph_id mapping:")
        sample_node_ids = set()
        for comm in communities:
            nodes = json.loads(comm['nodes']) if isinstance(comm['nodes'], str) else comm['nodes']
            sample_node_ids.update(nodes[:3])  # Take first 3 from each
        
        for node_id_str in list(sample_node_ids)[:10]:
            try:
                node_id = int(node_id_str)
                result = await self.conn.execute("""
                    SELECT meta_graph_id, nct_id, asset_ref
                    FROM docintel.meta_graphs
                    ORDER BY created_at
                    OFFSET %s LIMIT 1
                """, [node_id])
                row = await result.fetchone()
                
                if row:
                    print(f"  AGE node {node_id_str} → meta_graph {row['meta_graph_id']} (NCT: {row['nct_id']}, Asset: {row['asset_ref']})")
                    
                    # Check entities in this meta_graph
                    result2 = await self.conn.execute("""
                        SELECT COUNT(*) as entity_count
                        FROM docintel.entities
                        WHERE meta_graph_id = %s
                    """, [row['meta_graph_id']])
                    row2 = await result2.fetchone()
                    print(f"           └─ Contains {row2['entity_count']} entities")
                else:
                    print(f"  AGE node {node_id_str} → ❌ NO meta_graph found at offset {node_id}")
            except Exception as e:
                print(f"  AGE node {node_id_str} → ❌ Error: {e}")
    
    async def _analyze_entity_coverage(self, query_terms: List[str]):
        """Check if entities matching query terms exist in communities"""
        
        # Get all entities matching query terms
        term_conditions = " OR ".join(["entity_text ILIKE %s"] * len(query_terms))
        term_params = [f"%{term}%" for term in query_terms]
        
        result = await self.conn.execute(f"""
            SELECT 
                entity_text,
                entity_type,
                meta_graph_id,
                source_chunk_id,
                confidence
            FROM docintel.entities
            WHERE {term_conditions}
            LIMIT 20
        """, term_params)
        
        matching_entities = await result.fetchall()
        
        print(f"Entities matching query terms: {len(matching_entities)}")
        
        if not matching_entities:
            print("❌ NO ENTITIES found matching query terms!")
            print("   This could mean:")
            print("   1. Entity extraction didn't identify these terms")
            print("   2. Query terms are too specific/generic")
            print("   3. Documents don't contain these terms")
            return
        
        print(f"\nSample entities:")
        for ent in matching_entities[:10]:
            print(f"  - '{ent['entity_text']}' ({ent['entity_type']}, conf: {ent['confidence']:.2f})")
            print(f"    meta_graph: {ent['meta_graph_id']}, chunk: {ent['source_chunk_id']}")
        
        # Check if these meta_graphs are in communities
        meta_graph_ids = [str(ent['meta_graph_id']) for ent in matching_entities]
        
        print(f"\n🔍 Checking if these meta_graphs are in communities...")
        
        result = await self.conn.execute("""
            SELECT cluster_key, title, nodes, occurrence
            FROM ag_catalog.communities
        """)
        communities = await result.fetchall()
        
        found_in_communities = 0
        for comm in communities:
            nodes = json.loads(comm['nodes']) if isinstance(comm['nodes'], str) else comm['nodes']
            
            # Map AGE nodes to meta_graph_ids
            comm_meta_graphs = []
            for node_id_str in nodes[:50]:  # Check first 50 nodes
                try:
                    node_id = int(node_id_str)
                    result2 = await self.conn.execute("""
                        SELECT meta_graph_id
                        FROM docintel.meta_graphs
                        ORDER BY created_at
                        OFFSET %s LIMIT 1
                    """, [node_id])
                    row = await result2.fetchone()
                    if row:
                        comm_meta_graphs.append(str(row['meta_graph_id']))
                except:
                    continue
            
            # Check overlap
            overlap = set(meta_graph_ids) & set(comm_meta_graphs)
            if overlap:
                found_in_communities += 1
                print(f"  ✅ Community {comm['cluster_key']} ('{comm['title']}') contains {len(overlap)} matching meta_graphs")
                print(f"     Occurrence: {comm['occurrence']:.4f}, Nodes: {len(nodes)}")
        
        if found_in_communities == 0:
            print(f"  ❌ NO communities contain entities matching query terms!")
            print(f"     This is the ROOT CAUSE of retrieval failure!")
    
    async def _direct_entity_search(self, query_terms: List[str]):
        """Baseline: direct entity search without communities"""
        
        term_conditions = " OR ".join(["entity_text ILIKE %s"] * len(query_terms))
        term_params = [f"%{term}%" for term in query_terms]
        
        result = await self.conn.execute(f"""
            SELECT 
                entity_text,
                entity_type,
                source_chunk_id,
                confidence,
                normalized_id,
                normalized_source
            FROM docintel.entities
            WHERE {term_conditions}
            ORDER BY confidence DESC
            LIMIT 10
        """, term_params)
        
        entities = await result.fetchall()
        
        print(f"Direct entity search results: {len(entities)} entities")
        
        if entities:
            print(f"\nTop entities:")
            for ent in entities:
                print(f"  - '{ent['entity_text']}' ({ent['entity_type']})")
                print(f"    Chunk: {ent['source_chunk_id']}")
                print(f"    Normalized: {ent['normalized_id']} ({ent['normalized_source']})")
                print(f"    Confidence: {ent['confidence']:.2f}")
                print()
            
            print("✅ Direct search WORKS - entities are retrievable without communities")
        else:
            print("❌ Direct search FAILED - no entities found for query terms")
    
    async def _simulate_community_search(self, query: str, query_terms: List[str]):
        """Simulate the U-Retrieval community search algorithm"""
        
        print("Simulating U-Retrieval._find_relevant_communities()...")
        print()
        
        # Get all communities
        result = await self.conn.execute("""
            SELECT 
                cluster_key,
                level,
                title,
                nodes,
                edges,
                chunk_ids,
                occurrence
            FROM ag_catalog.communities
            ORDER BY cluster_key
        """)
        communities = await result.fetchall()
        
        relevant_communities = []
        query_lower = query.lower()
        
        for comm in communities:
            cluster_key = comm['cluster_key']
            title = comm['title']
            nodes = json.loads(comm['nodes']) if isinstance(comm['nodes'], str) else comm['nodes']
            chunk_ids = json.loads(comm['chunk_ids']) if isinstance(comm['chunk_ids'], str) else comm['chunk_ids']
            
            relevance_score = 0.0
            
            # Check title relevance
            title_match = False
            if title and any(term in title.lower() for term in query_terms):
                relevance_score += 0.3
                title_match = True
            
            # Calculate entity relevance (simulate _calculate_community_entity_relevance)
            entity_relevance = await self._calculate_entity_relevance_debug(nodes, query_lower, query_terms)
            relevance_score += entity_relevance * 0.7
            
            print(f"Community {cluster_key}: '{title}'")
            print(f"  Title match: {title_match}")
            print(f"  Entity relevance: {entity_relevance:.4f}")
            print(f"  Total relevance: {relevance_score:.4f}")
            print(f"  Nodes: {len(nodes)}, Chunks: {len(chunk_ids)}")
            
            if relevance_score > 0.01:
                relevant_communities.append({
                    'cluster_key': cluster_key,
                    'title': title,
                    'relevance_score': relevance_score
                })
                print(f"  ✅ SELECTED (relevance > 0.01)")
            else:
                print(f"  ❌ REJECTED (relevance <= 0.01)")
            print()
        
        print(f"\nFinal result: {len(relevant_communities)} relevant communities")
        
        if relevant_communities:
            print("\nTop communities:")
            for comm in sorted(relevant_communities, key=lambda x: x['relevance_score'], reverse=True)[:5]:
                print(f"  - {comm['cluster_key']}: {comm['title']} (score: {comm['relevance_score']:.4f})")
        else:
            print("❌ NO relevant communities found - this explains why retrieval failed!")
    
    async def _calculate_entity_relevance_debug(
        self, 
        age_node_ids: List[str], 
        query_lower: str,
        query_terms: List[str]
    ) -> float:
        """Debug version of _calculate_community_entity_relevance"""
        
        if not age_node_ids:
            return 0.0
        
        # Convert AGE node IDs to meta_graph_ids
        meta_graph_ids = []
        conversion_failures = 0
        
        for node_id_str in age_node_ids[:20]:  # Check first 20 nodes
            try:
                node_id = int(node_id_str)
                result = await self.conn.execute("""
                    SELECT meta_graph_id 
                    FROM docintel.meta_graphs 
                    ORDER BY created_at 
                    OFFSET %s LIMIT 1
                """, [node_id])
                row = await result.fetchone()
                if row:
                    meta_graph_ids.append(str(row['meta_graph_id']))
                else:
                    conversion_failures += 1
            except:
                conversion_failures += 1
        
        if not meta_graph_ids:
            return 0.0
        
        # Get entities
        mg_placeholders = ','.join(['%s'] * len(meta_graph_ids))
        result = await self.conn.execute(f"""
            SELECT entity_text, entity_type, normalized_id
            FROM docintel.entities 
            WHERE meta_graph_id::text IN ({mg_placeholders})
            LIMIT 100
        """, meta_graph_ids)
        
        entities = await result.fetchall()
        
        if not entities:
            return 0.0
        
        # Calculate relevance
        relevance_score = 0.0
        matches = []
        
        for entity in entities:
            entity_text = entity['entity_text']
            entity_lower = entity_text.lower()
            
            # Exact match
            for term in query_terms:
                if len(term) > 2 and term == entity_lower:
                    relevance_score += 2.0
                    matches.append((entity_text, 'exact'))
                    break
            
            # Substring match
            for term in query_terms:
                if len(term) > 2 and term in entity_lower:
                    relevance_score += 1.0
                    matches.append((entity_text, 'substring'))
                    break
            
            # Partial word match
            entity_words = entity_lower.split()
            for term in query_terms:
                if len(term) > 2 and term in entity_words:
                    relevance_score += 0.5
                    matches.append((entity_text, 'word'))
                    break
        
        # Normalize by number of entities
        if len(entities) > 0:
            relevance_score = relevance_score / len(entities)
        
        return relevance_score
    
    async def _generate_recommendations(self, query_terms: List[str]):
        """Generate recommendations to fix community search"""
        
        # Check various metrics
        result1 = await self.conn.execute("SELECT COUNT(*) as cnt FROM ag_catalog.communities")
        comm_count = (await result1.fetchone())['cnt']
        
        result2 = await self.conn.execute("SELECT COUNT(*) as cnt FROM docintel.entities")
        entity_count = (await result2.fetchone())['cnt']
        
        result3 = await self.conn.execute("SELECT COUNT(*) as cnt FROM docintel.meta_graphs")
        mg_count = (await result3.fetchone())['cnt']
        
        print(f"System metrics:")
        print(f"  - Communities: {comm_count}")
        print(f"  - Entities: {entity_count}")
        print(f"  - Meta-graphs: {mg_count}")
        print()
        
        recommendations = []
        
        if comm_count == 0:
            recommendations.append("❌ CRITICAL: No communities exist. Run community detection algorithm.")
        elif comm_count < 10:
            recommendations.append("⚠️  Very few communities. Consider running community detection with different parameters.")
        
        # Check community titles
        result4 = await self.conn.execute("""
            SELECT title FROM ag_catalog.communities WHERE title LIKE 'Clinical Community %' LIMIT 5
        """)
        generic_titles = await result4.fetchall()
        
        if len(generic_titles) >= 3:
            recommendations.append("⚠️  Community titles are too generic ('Clinical Community N'). Consider generating semantic titles based on entity content.")
        
        # Check entity coverage
        term_conditions = " OR ".join(["entity_text ILIKE %s"] * len(query_terms))
        term_params = [f"%{term}%" for term in query_terms]
        
        result5 = await self.conn.execute(f"""
            SELECT COUNT(*) as cnt FROM docintel.entities
            WHERE {term_conditions}
        """, term_params)
        matching_entities = (await result5.fetchone())['cnt']
        
        if matching_entities == 0:
            recommendations.append(f"❌ CRITICAL: No entities match query terms {query_terms}. Entity extraction may have failed or query is too specific.")
        elif matching_entities > 0:
            recommendations.append(f"✅ {matching_entities} entities match query terms - retrieval should work!")
        
        # Check AGE mapping
        result6 = await self.conn.execute("""
            SELECT nodes FROM ag_catalog.communities LIMIT 1
        """)
        row = await result6.fetchone()
        
        if row:
            nodes = json.loads(row['nodes']) if isinstance(row['nodes'], str) else row['nodes']
            if nodes:
                test_node = int(nodes[0])
                result7 = await self.conn.execute("""
                    SELECT meta_graph_id FROM docintel.meta_graphs
                    ORDER BY created_at OFFSET %s LIMIT 1
                """, [test_node])
                test_mg = await result7.fetchone()
                
                if not test_mg:
                    recommendations.append(f"❌ CRITICAL: AGE node {test_node} doesn't map to any meta_graph. Node ID offset may be wrong.")
        
        print("Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        print()
        
        print("Suggested fixes:")
        print("  1. Add fallback to direct semantic search when communities return 0 results")
        print("  2. Generate semantic community titles (e.g., 'Niraparib Safety Data')")
        print("  3. Lower relevance threshold from 0.01 to 0.001 to catch more communities")
        print("  4. Fix AGE node ID → meta_graph_id mapping if offset calculation is wrong")
        print("  5. Add fuzzy matching for entity text comparison (not just exact/substring)")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_community_search.py 'Your query here'")
        print("\nExample queries:")
        print("  python scripts/debug_community_search.py 'What is niraparib?'")
        print("  python scripts/debug_community_search.py 'What are adverse events?'")
        print("  python scripts/debug_community_search.py 'What are the primary endpoints?'")
        sys.exit(1)
    
    query = sys.argv[1]
    db_dsn = "postgresql://dbuser:dbpass123@localhost:5432/docintel"
    
    debugger = CommunitySearchDebugger(db_dsn)
    await debugger.connect()
    
    try:
        await debugger.debug_query(query)
    finally:
        await debugger.close()


if __name__ == "__main__":
    asyncio.run(main())
