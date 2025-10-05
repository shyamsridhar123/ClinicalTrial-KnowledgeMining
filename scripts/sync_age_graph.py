#!/usr/bin/env python3
"""
AGE Property Graph Data Synchronization Script

Migrates entity and relation data from PostgreSQL tables to AGE property graph format.
This enables Cypher queries and graph-based operations on the clinical knowledge graph.

Priority: P1 (High Impact, Medium Effort)
Required for: Medical-Graph-RAG compliance, advanced graph queries
"""

import asyncio
import json
import logging
import os
import psycopg  # type: ignore[import-not-found]
from typing import List, Dict, Any, Optional

from docintel.knowledge_graph.age_utils import configure_age_session_async

try:
    from docintel.config import get_config as _get_config
    from docintel.config import AgeGraphSettings as _AgeGraphSettings
except ImportError:  # pragma: no cover - standalone script fallback
    _get_config = None
    _AgeGraphSettings = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _resolve_connection_string(explicit: Optional[str] = None) -> str:
    """Resolve connection string from explicit value, environment, or config."""
    if explicit:
        return explicit
    env_dsn = os.getenv("DOCINTEL_DSN") or os.getenv("DOCINTEL_VECTOR_DB_DSN")
    if env_dsn:
        return env_dsn
    if _get_config:
        return _get_config().docintel_dsn
    raise RuntimeError("No PostgreSQL connection string configured.")


class AGEGraphSynchronizer:
    """Synchronizes PostgreSQL data with AGE property graph"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = _resolve_connection_string(connection_string)
        if _get_config:
            self.age_settings = _get_config().age_graph
        elif _AgeGraphSettings:
            self.age_settings = _AgeGraphSettings()
        else:  # pragma: no cover - configuration must be provided
            raise RuntimeError("Unable to resolve AGE graph settings")
        self.graph_name = self.age_settings.graph_name
        self.conn = None
        
    async def connect(self):
        """Establish database connection"""
        self.conn = await psycopg.AsyncConnection.connect(self.connection_string)
        async with self.conn.cursor() as cur:
            await configure_age_session_async(cur, self.age_settings)
        await self.conn.commit()
        logger.info("Connected to database and configured AGE session")
        
    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            
    async def clear_graph(self):
        """Clear existing graph data"""
        try:
            # Clear all edges first (due to foreign key constraints)
            cypher_clear_edges = """
            SELECT * FROM cypher('clinical_kg', $$
                MATCH ()-[r]->()
                DELETE r
            $$) AS (result agtype);
            """
            await self.conn.execute(self._format_cypher(cypher_clear_edges))
            
            # Clear all vertices
            cypher_clear_vertices = """
            SELECT * FROM cypher('clinical_kg', $$
                MATCH (n)
                DELETE n
            $$) AS (result agtype);
            """
            await self.conn.execute(self._format_cypher(cypher_clear_vertices))
            
            logger.info("Cleared existing graph data")
            
        except Exception as e:
            logger.warning(f"Error clearing graph (might be empty): {e}")
            
    async def sync_entities(self):
        """Migrate entities from PostgreSQL to AGE vertices using batch operations"""
        logger.info("Starting entity synchronization...")
        
        # Fetch entities from PostgreSQL
        result = await self.conn.execute("""
            SELECT id, entity_text, entity_type, confidence, chunk_id, start_pos, end_pos, normalized_id, normalized_source, context_flags
            FROM ag_catalog.entities
            ORDER BY id
        """)
        
        entities = await result.fetchall()
        logger.info(f"Found {len(entities)} entities to migrate")
        
        # Build batch Cypher command for all entities
        batch_commands = []
        
        for entity in entities:
            entity_id, text, entity_type, confidence, chunk_id, start_pos, end_pos, normalized_id, normalized_source, context_flags = entity
            
            # Escape text for Cypher
            escaped_text = text.replace("'", "\\'").replace('"', '\\"')
            
            # Prepare context properties
            context_props = ""
            if context_flags:
                try:
                    context_dict = json.loads(context_flags) if isinstance(context_flags, str) else context_flags
                    for key, value in context_dict.items():
                        if isinstance(value, bool):
                            context_props += f", {key}: {str(value).lower()}"
                        elif value is not None:
                            context_props += f", {key}: '{str(value)}'"
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid context_flags for entity {entity_id}: {context_flags}")
            
            # Build CREATE command
            create_cmd = f"""
                CREATE (n{entity_id}:{entity_type.title()} {{
                    id: {entity_id},
                    text: '{escaped_text}',
                    entity_type: '{entity_type}',
                    confidence: {confidence},
                    chunk_id: {chunk_id if chunk_id else 'null'},
                    start_pos: {start_pos if start_pos else 'null'},
                    end_pos: {end_pos if end_pos else 'null'},
                    normalized_id: '{normalized_id if normalized_id else ""}',
                    normalized_source: '{normalized_source if normalized_source else ""}'
                    {context_props}
                }})"""
            
            batch_commands.append(create_cmd)
        
        # Execute all creates in a single transaction
        if batch_commands:
            batch_cypher = f"""
            SELECT * FROM cypher('clinical_kg', $$
                {' '.join(batch_commands)}
            $$) AS (result agtype);
            """
            
            try:
                async with self.conn.transaction():
                    await self.conn.execute(self._format_cypher(batch_cypher))
                logger.info(f"✅ Batch created {len(entities)} vertices successfully")
            except Exception as e:
                logger.error(f"❌ Batch entity creation failed: {e}")
                # Fallback to individual creates if batch fails
                logger.info("Falling back to individual entity creation...")
                await self._sync_entities_individually(entities)
        
        logger.info(f"Completed entity synchronization: {len(entities)} vertices processed")
        
    async def _sync_entities_individually(self, entities):
        """Fallback method for individual entity creation"""
        success_count = 0
        for entity in entities:
            entity_id, text, entity_type, confidence, chunk_id, start_pos, end_pos, normalized_id, normalized_source, context_flags = entity
            
            try:
                # Individual create with proper transaction
                escaped_text = text.replace("'", "\\'").replace('"', '\\"')
                
                context_props = ""
                if context_flags:
                    try:
                        context_dict = json.loads(context_flags) if isinstance(context_flags, str) else context_flags
                        for key, value in context_dict.items():
                            if isinstance(value, bool):
                                context_props += f", {key}: {str(value).lower()}"
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                cypher_create = f"""
                SELECT * FROM cypher('clinical_kg', $$
                    CREATE (n:{entity_type.title()} {{
                        id: {entity_id},
                        text: '{escaped_text}',
                        entity_type: '{entity_type}',
                        confidence: {confidence}
                        {context_props}
                    }})
                $$) AS (result agtype);
                """
                
                await self.conn.execute(self._format_cypher(cypher_create))
                success_count += 1
                
                if success_count % 10 == 0:
                    logger.info(f"Individual sync progress: {success_count}/{len(entities)}")
                    
            except Exception as e:
                logger.error(f"Failed to create entity {entity_id}: {e}")
        
        logger.info(f"Individual sync completed: {success_count}/{len(entities)} entities created")
        
    async def sync_relations(self):
        """Migrate relations from PostgreSQL to AGE edges using batch operations"""
        logger.info("Starting relation synchronization...")
        
        # Fetch relations with entity details
        result = await self.conn.execute("""
            SELECT 
                r.id as relation_id,
                r.predicate,
                r.confidence,
                r.evidence_span,
                r.subject_entity_id,
                r.object_entity_id,
                se.entity_text as subject_text,
                se.entity_type as subject_type,
                oe.entity_text as object_text,
                oe.entity_type as object_type
            FROM ag_catalog.relations r
            JOIN ag_catalog.entities se ON r.subject_entity_id = se.id
            JOIN ag_catalog.entities oe ON r.object_entity_id = oe.id
            ORDER BY r.id
        """)
        
        relations = await result.fetchall()
        logger.info(f"Found {len(relations)} relations to migrate")
        
        if not relations:
            logger.info("No relations to sync")
            return
        
        # Build batch Cypher commands for all relations
        batch_commands = []
        
        for relation in relations:
            (relation_id, predicate, confidence, evidence_span, 
             subject_id, object_id, subject_text, subject_type, 
             object_text, object_type) = relation
            
            # Escape evidence span
            escaped_evidence = evidence_span.replace("'", "\\'").replace('"', '\\"') if evidence_span else ""
            
            # Build MATCH and CREATE commands
            match_create_cmd = f"""
                MATCH (s{subject_id}:{subject_type.title()} {{id: {subject_id}}}),
                      (o{object_id}:{object_type.title()} {{id: {object_id}}})
                CREATE (s{subject_id})-[r{relation_id}:{predicate.upper().replace(' ', '_')} {{
                    id: {relation_id},
                    predicate: '{predicate}',
                    confidence: {confidence},
                    evidence_span: '{escaped_evidence}'
                }}]->(o{object_id})"""
            
            batch_commands.append(match_create_cmd)
        
        # Execute all relation creates in a single transaction
        if batch_commands:
            batch_cypher = f"""
            SELECT * FROM cypher('clinical_kg', $$
                {' '.join(batch_commands)}
            $$) AS (result agtype);
            """
            
            try:
                async with self.conn.transaction():
                    await self.conn.execute(self._format_cypher(batch_cypher))
                logger.info(f"✅ Batch created {len(relations)} edges successfully")
            except Exception as e:
                logger.error(f"❌ Batch relation creation failed: {e}")
                # Fallback to individual creates
                logger.info("Falling back to individual relation creation...")
                await self._sync_relations_individually(relations)
        
        logger.info(f"Completed relation synchronization: {len(relations)} edges processed")
    
    async def _sync_relations_individually(self, relations):
        """Fallback method for individual relation creation"""
        success_count = 0
        for relation in relations:
            (relation_id, predicate, confidence, evidence_span, 
             subject_id, object_id, subject_text, subject_type, 
             object_text, object_type) = relation
            
            try:
                escaped_evidence = evidence_span.replace("'", "\\'") if evidence_span else ""
                
                cypher_create = f"""
                SELECT * FROM cypher('clinical_kg', $$
                    MATCH (s:{subject_type.title()} {{id: {subject_id}}}), 
                          (o:{object_type.title()} {{id: {object_id}}})
                    CREATE (s)-[r:{predicate.upper().replace(' ', '_')} {{
                        id: {relation_id},
                        predicate: '{predicate}',
                        confidence: {confidence}
                    }}]->(o)
                $$) AS (result agtype);
                """
                
                await self.conn.execute(self._format_cypher(cypher_create))
                success_count += 1
                
                if success_count % 5 == 0:
                    logger.info(f"Individual relation sync progress: {success_count}/{len(relations)}")
                    
            except Exception as e:
                logger.error(f"Failed to create relation {relation_id}: {e}")
        
        logger.info(f"Individual relation sync completed: {success_count}/{len(relations)} relations created")
        
    async def verify_sync(self) -> Dict[str, Any]:
        """Verify the synchronization was successful"""
        logger.info("Verifying synchronization...")
        
        stats = {}
        
        try:
            # Count vertices
            result = await self.conn.execute(self._format_cypher("""
                SELECT * FROM cypher('clinical_kg', $$
                    MATCH (n) RETURN count(n) as vertex_count
                $$) AS (vertex_count agtype);
            """))
            vertex_row = await result.fetchone()
            stats['vertices'] = int(str(vertex_row[0]).strip('"'))
            
            # Count edges
            result = await self.conn.execute(self._format_cypher("""
                SELECT * FROM cypher('clinical_kg', $$
                    MATCH ()-[r]->() RETURN count(r) as edge_count
                $$) AS (edge_count agtype);
            """))
            edge_row = await result.fetchone()
            stats['edges'] = int(str(edge_row[0]).strip('"'))
            
            # Get vertex types
            result = await self.conn.execute(self._format_cypher("""
                SELECT * FROM cypher('clinical_kg', $$
                    MATCH (n) RETURN labels(n)[0] as node_type, count(*) as count
                $$) AS (node_type agtype, count agtype);
            """))
            vertex_types = {}
            async for row in result:
                node_type = str(row[0]).strip('"')
                count = int(str(row[1]).strip('"'))
                vertex_types[node_type] = count
            stats['vertex_types'] = vertex_types
            
            # Get edge types
            result = await self.conn.execute(self._format_cypher("""
                SELECT * FROM cypher('clinical_kg', $$
                    MATCH ()-[r]->() RETURN type(r) as edge_type, count(*) as count
                $$) AS (edge_type agtype, count agtype);
            """))
            edge_types = {}
            async for row in result:
                edge_type = str(row[0]).strip('"')
                count = int(str(row[1]).strip('"'))
                edge_types[edge_type] = count
            stats['edge_types'] = edge_types
            
            logger.info(f"Verification complete: {stats['vertices']} vertices, {stats['edges']} edges")
            return stats
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return {'error': str(e)}
            
    async def run_full_sync(self):
        """Run complete synchronization process"""
        logger.info("🔄 Starting AGE property graph synchronization")
        
        try:
            await self.connect()
            await self.clear_graph()
            await self.sync_entities()
            await self.sync_relations()
            stats = await self.verify_sync()
            
            logger.info("✅ AGE property graph synchronization completed successfully")
            logger.info(f"📊 Final stats: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Synchronization failed: {e}")
            raise
        finally:
            await self.close()
    
    def _format_cypher(self, statement: str) -> str:
        """Replace legacy graph name placeholders with configured values."""

        return statement.replace("clinical_kg", self.graph_name)


async def main():
    """Main execution function"""
    synchronizer = AGEGraphSynchronizer()
    
    try:
        stats = await synchronizer.run_full_sync()
        
        print("\n🎉 AGE Property Graph Synchronization Complete!")
        print("=" * 50)
        print(f"Vertices created: {stats.get('vertices', 0)}")
        print(f"Edges created: {stats.get('edges', 0)}")
        
        if 'vertex_types' in stats:
            print("\nVertex types:")
            for vtype, count in stats['vertex_types'].items():
                print(f"  {vtype}: {count}")
                
        if 'edge_types' in stats:
            print("\nEdge types:")
            for etype, count in stats['edge_types'].items():
                print(f"  {etype}: {count}")
                
        print("\n🔗 AGE property graph is now ready for Cypher queries!")
        
    except Exception as e:
        print(f"❌ Synchronization failed: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))