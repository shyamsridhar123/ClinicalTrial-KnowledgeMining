#!/usr/bin/env python3
"""
Sync relations from docintel.relations to AGE graph edges.

This creates edges in the AGE property graph so multi-hop traversal works.

Usage:
    # Sync all relations
    pixi run -- python scripts/sync_relations_to_age.py
    
    # Dry run first
    pixi run -- python scripts/sync_relations_to_age.py --dry-run
"""

import sys
import logging
import argparse
from typing import List, Dict, Any

sys.path.insert(0, 'src')

import psycopg
from psycopg.rows import dict_row

from docintel.config import VectorDatabaseSettings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/age_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AGEGraphSyncer:
    """Sync relations from PostgreSQL to AGE graph."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        
        # Load config
        vector_db_settings = VectorDatabaseSettings()
        self.db_dsn = str(vector_db_settings.dsn)
        
        self.graph_name = 'clinical_graph'
        
        logger.info(f"Initialized AGE syncer (dry_run={dry_run})")
    
    def ensure_graph_exists(self):
        """Create AGE graph if it doesn't exist."""
        conn = psycopg.connect(self.db_dsn)
        
        with conn.cursor() as cur:
            # Load AGE extension and set search path (required for every session)
            cur.execute("LOAD 'age'")
            cur.execute("SET search_path = ag_catalog, '$user', public")
            
            # Check if graph exists
            cur.execute("""
                SELECT COUNT(*) FROM ag_catalog.ag_graph WHERE name = %s
            """, (self.graph_name,))
            
            if cur.fetchone()[0] == 0:
                logger.info(f"Creating AGE graph: {self.graph_name}")
                cur.execute(f"SELECT ag_catalog.create_graph('{self.graph_name}')")
                conn.commit()
                logger.info("✅ Graph created")
            else:
                logger.info(f"Graph {self.graph_name} already exists")
        
        conn.close()
    
    def get_entity_age_vertex_mapping(self) -> Dict[str, int]:
        """
        Map entity_id to AGE vertex ID.
        
        AGE stores vertices in repo_nodes table with vertex IDs.
        We need to find the AGE vertex ID for each entity.
        """
        conn = psycopg.connect(self.db_dsn)
        mapping = {}
        
        with conn.cursor(row_factory=dict_row) as cur:
            # Get all entities and their properties
            cur.execute("""
                SELECT entity_id, entity_text, entity_type
                FROM docintel.entities
                ORDER BY entity_id
            """)
            
            entities = [dict(e) for e in cur.fetchall()]
            
            logger.info(f"Mapping {len(entities):,} entities to AGE vertices...")
            
            # For each entity, find or create AGE vertex
            for i, entity in enumerate(entities):
                if i % 1000 == 0 and i > 0:
                    logger.info(f"  Processed {i:,}/{len(entities):,} entities")
                
                entity_id = entity['entity_id']
                
                # Query AGE to find vertex by entity_id property
                cur.execute(f"""
                    SELECT * FROM ag_catalog.cypher('{self.graph_name}', $$
                        MATCH (e:Entity {{entity_id: '{entity_id}'}})
                        RETURN id(e) as vertex_id
                    $$) as (vertex_id agtype)
                """)
                
                row = cur.fetchone()
                if row and row['vertex_id']:
                    # Extract numeric ID from agtype
                    vertex_id = int(str(row['vertex_id']))
                    mapping[entity_id] = vertex_id
        
        conn.close()
        
        logger.info(f"✅ Mapped {len(mapping):,} entities to AGE vertices")
        return mapping
    
    def create_age_vertices_for_entities(self) -> Dict[str, int]:
        """
        Create AGE vertices for all entities if they don't exist.
        Returns mapping of entity_id to AGE vertex ID.
        """
        conn = psycopg.connect(self.db_dsn)
        mapping = {}
        
        with conn.cursor(row_factory=dict_row) as cur:
            # Load AGE extension and set search path (required for every session)
            cur.execute("LOAD 'age'")
            cur.execute("SET search_path = ag_catalog, '$user', public")
            
            # Get all entities
            cur.execute("""
                SELECT entity_id, entity_text, entity_type, normalized_id, normalized_source
                FROM docintel.entities
                ORDER BY entity_id
            """)
            
            entities = [dict(e) for e in cur.fetchall()]
            
            logger.info(f"Creating AGE vertices for {len(entities):,} entities...")
            
            batch_size = 500
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i+batch_size]
                
                if self.dry_run:
                    logger.info(f"  [DRY RUN] Would create {len(batch)} vertices")
                    continue
                
                # Create vertices individually with commits to isolate failures
                created_in_batch = 0
                failed_in_batch = 0
                
                for entity in batch:
                    entity_id = entity['entity_id']
                    # Escape for Cypher: replace backslash first, then single quotes
                    entity_text = entity['entity_text'].replace('\\', '\\\\').replace("'", "\\'")
                    entity_type = (entity['entity_type'] or 'unknown').replace('\\', '\\\\').replace("'", "\\'")
                    normalized_id = (entity['normalized_id'] or '').replace('\\', '\\\\').replace("'", "\\'")
                    normalized_source = (entity['normalized_source'] or '').replace('\\', '\\\\').replace("'", "\\'")
                    
                    try:
                        # Create or match vertex (AGE doesn't support ON CREATE SET, so set all properties in MERGE)
                        cur.execute(f"""
                            SELECT * FROM ag_catalog.cypher('{self.graph_name}', $$
                                MERGE (e:Entity {{
                                    entity_id: '{entity_id}',
                                    text: '{entity_text}',
                                    type: '{entity_type}',
                                    normalized_id: '{normalized_id}',
                                    normalized_source: '{normalized_source}'
                                }})
                                RETURN id(e) as vertex_id
                            $$) as (vertex_id agtype)
                        """)
                        
                        row = cur.fetchone()
                        if row and row['vertex_id']:
                            vertex_id = int(str(row['vertex_id']))
                            mapping[entity_id] = vertex_id
                            created_in_batch += 1
                        
                        # Commit after each vertex to avoid cascading transaction aborts
                        conn.commit()
                    
                    except Exception as e:
                        # Rollback the failed transaction and continue
                        conn.rollback()
                        failed_in_batch += 1
                        if failed_in_batch <= 5:  # Log first 5 failures per batch
                            logger.warning(f"Failed to create vertex for {entity_id}: {e}")
                
                logger.info(f"  Created {created_in_batch}/{len(batch)} vertices (batch {i//batch_size+1}, failed: {failed_in_batch})")
        
        conn.close()
        
        logger.info(f"✅ Created {len(mapping):,} AGE vertices")
        return mapping
    
    def sync_relations_to_edges(self, vertex_mapping: Dict[str, int]):
        """Create AGE edges from docintel.relations table."""
        conn = psycopg.connect(self.db_dsn)
        
        with conn.cursor(row_factory=dict_row) as cur:
            # Load AGE extension and set search path (required for every session)
            cur.execute("LOAD 'age'")
            cur.execute("SET search_path = ag_catalog, '$user', public")
            
            # Get all relations
            cur.execute("""
                SELECT 
                    relation_id,
                    subject_entity_id,
                    predicate,
                    object_entity_id,
                    confidence,
                    evidence_span
                FROM docintel.relations
                ORDER BY relation_id
            """)
            
            relations = [dict(r) for r in cur.fetchall()]
            
            logger.info(f"Syncing {len(relations):,} relations to AGE edges...")
            
            # Batch edges with commits every N edges (context manager handles transactions)
            batch_size = 100
            
            created = 0
            skipped = 0
            failed = 0
            
            for i, rel in enumerate(relations):
                # Re-set search path at start of each batch (lost after commit)
                if i % batch_size == 0:
                    cur.execute("LOAD 'age'")
                    cur.execute("SET search_path = ag_catalog, '$user', public")
                
                if i % 500 == 0:
                    logger.info(f"  Progress: {i:,}/{len(relations):,} ({created:,} created, {skipped:,} skipped, {failed:,} failed)")
                
                subject_id = rel['subject_entity_id']
                object_id = rel['object_entity_id']
                predicate = rel['predicate']
                confidence = rel['confidence'] or 0.8
                
                # Proper Cypher escaping for evidence text
                evidence = (rel['evidence_span'] or '')[:200]
                evidence = evidence.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                
                # Check if both entities have vertices
                if subject_id not in vertex_mapping or object_id not in vertex_mapping:
                    skipped += 1
                    continue
                
                if self.dry_run:
                    created += 1
                    continue
                
                try:
                    # Create edge using Cypher with RELATES_TO type (generic)
                    cur.execute(f"""
                        SELECT * FROM ag_catalog.cypher('{self.graph_name}', $$
                            MATCH (subj:Entity {{entity_id: '{subject_id}'}})
                            MATCH (obj:Entity {{entity_id: '{object_id}'}})
                            CREATE (subj)-[r:RELATES_TO {{
                                predicate: '{predicate}',
                                confidence: {confidence},
                                evidence: '{evidence}'
                            }}]->(obj)
                            RETURN id(r) as edge_id
                        $$) as (edge_id agtype)
                    """)
                    
                    created += 1
                
                except Exception as e:
                    logger.warning(f"Failed to create edge {i+1} ({predicate}): {str(e)[:100]}")
                    failed += 1
                
                # Commit every batch to limit transaction size
                if (i + 1) % batch_size == 0:
                    conn.commit()
            
            # Final commit for remaining edges
            conn.commit()
        
        conn.close()
        
        logger.info(f"""
=== SYNC COMPLETE ===
Created: {created:,} edges
Skipped: {skipped:,} (missing vertices)
Failed: {failed:,}
        """)
        
        return created
    
    def verify_graph(self):
        """Verify AGE graph has edges."""
        conn = psycopg.connect(self.db_dsn)
        
        with conn.cursor() as cur:
            # Load AGE extension and set search path (required for every session)
            cur.execute("LOAD 'age'")
            cur.execute("SET search_path = ag_catalog, '$user', public")
            
            # Count vertices
            cur.execute(f"""
                SELECT * FROM ag_catalog.cypher('{self.graph_name}', $$
                    MATCH (n:Entity)
                    RETURN count(n) as node_count
                $$) as (node_count agtype)
            """)
            node_count = int(str(cur.fetchone()[0]))
            
            # Count edges
            cur.execute(f"""
                SELECT * FROM ag_catalog.cypher('{self.graph_name}', $$
                    MATCH ()-[r]->()
                    RETURN count(r) as edge_count
                $$) as (edge_count agtype)
            """)
            edge_count = int(str(cur.fetchone()[0]))
            
            logger.info(f"""
=== GRAPH STATS ===
Nodes: {node_count:,}
Edges: {edge_count:,}
            """)
        
        conn.close()
    
    def run(self):
        """Main sync process."""
        logger.info("Starting AGE graph sync...")
        
        # Step 1: Ensure graph exists
        self.ensure_graph_exists()
        
        # Step 2: Create vertices for all entities
        vertex_mapping = self.create_age_vertices_for_entities()
        
        # Step 3: Create edges from relations
        created = self.sync_relations_to_edges(vertex_mapping)
        
        # Step 4: Verify
        if not self.dry_run:
            self.verify_graph()
        
        logger.info("✅ AGE sync complete!")


def main():
    parser = argparse.ArgumentParser(description='Sync relations to AGE graph')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no changes)')
    
    args = parser.parse_args()
    
    syncer = AGEGraphSyncer(dry_run=args.dry_run)
    syncer.run()


if __name__ == '__main__':
    main()
