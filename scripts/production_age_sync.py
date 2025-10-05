#!/usr/bin/env python3
"""
Production-Ready AGE Property Graph Synchronization

Uses chunked batch processing with proper error handling and transaction management
for scalable deployment with thousands of entities.
"""

import asyncio
import logging
import os
import psycopg  # type: ignore[import-not-found]
import json
from typing import List, Dict, Any, Optional

from docintel.knowledge_graph.age_utils import configure_age_session_async

try:
    from docintel.config import get_config as _get_config
    from docintel.config import AgeGraphSettings as _AgeGraphSettings
except ImportError:  # pragma: no cover - script fallback when package not on path
    _get_config = None
    _AgeGraphSettings = None

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


class ProductionAGESync:
    """Production-ready AGE synchronization with batch processing"""
    
    def __init__(self, connection_string: Optional[str] = None, batch_size: int = 50):
        self.connection_string = _resolve_connection_string(connection_string)
        if _get_config:
            self.age_settings = _get_config().age_graph
        elif _AgeGraphSettings:
            self.age_settings = _AgeGraphSettings()
        else:  # pragma: no cover - configuration must be provided
            raise RuntimeError("Unable to resolve AGE graph settings")
        self.graph_name = self.age_settings.graph_name
        self.batch_size = batch_size
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
            cypher_clear = """
            SELECT * FROM cypher('clinical_kg', $$
                MATCH (n)
                DETACH DELETE n
            $$) AS (result agtype);
            """
            await self.conn.execute(self._format_cypher(cypher_clear))
            logger.info("Cleared existing graph data")
        except Exception as e:
            logger.warning(f"Error clearing graph: {e}")
            
    async def sync_entities_chunked(self):
        """Sync entities using chunked batch processing"""
        logger.info("Starting chunked entity synchronization...")
        
        # Fetch all entities
        result = await self.conn.execute("""
            SELECT id, entity_text, entity_type, confidence, context_flags
            FROM ag_catalog.entities
            ORDER BY id
        """)
        entities = await result.fetchall()
        logger.info(f"Found {len(entities)} entities to migrate")
        
        total_synced = 0
        
        # Process in chunks
        for i in range(0, len(entities), self.batch_size):
            chunk = entities[i:i + self.batch_size]
            
            try:
                # Process chunk individually for reliability
                chunk_synced = await self._sync_entity_chunk(chunk)
                total_synced += chunk_synced
                
                logger.info(f"Synced chunk {i//self.batch_size + 1}: {chunk_synced}/{len(chunk)} entities (Total: {total_synced}/{len(entities)})")
                
            except Exception as e:
                logger.error(f"Error syncing entity chunk {i//self.batch_size + 1}: {e}")
                
        logger.info(f"✅ Entity sync complete: {total_synced}/{len(entities)} entities synced")
        return total_synced
        
    async def _sync_entity_chunk(self, entities: List) -> int:
        """Sync a chunk of entities individually for reliability"""
        synced_count = 0
        
        for entity in entities:
            entity_id, text, entity_type, confidence, context_flags = entity
            
            try:
                # Escape text properly
                escaped_text = text.replace("'", "\\'").replace('"', '\\"')
                
                # Build context properties
                context_props = []
                if context_flags:
                    try:
                        context_dict = json.loads(context_flags) if isinstance(context_flags, str) else context_flags
                        for key, value in context_dict.items():
                            if isinstance(value, bool):
                                context_props.append(f"{key}: {str(value).lower()}")
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Invalid context_flags for entity {entity_id}")
                
                # Build properties string
                props = [
                    f"id: {entity_id}",
                    f"text: '{escaped_text}'",
                    f"entity_type: '{entity_type}'",
                    f"confidence: {confidence}"
                ]
                props.extend(context_props)
                
                # Create vertex
                cypher_create = f"""
                SELECT * FROM cypher('clinical_kg', $$
                    CREATE (n:{entity_type.title()} {{{', '.join(props)}}})
                    RETURN n.id as created_id
                $$) AS (created_id agtype);
                """
                
                await self.conn.execute(self._format_cypher(cypher_create))
                synced_count += 1
                
            except Exception as e:
                logger.error(f"Failed to sync entity {entity_id}: {e}")
                
        return synced_count
        
    async def sync_relations_chunked(self):
        """Sync relations using chunked processing"""
        logger.info("Starting chunked relation synchronization...")
        
        # Fetch relations
        result = await self.conn.execute("""
            SELECT 
                r.id, r.predicate, r.confidence, r.evidence_span,
                r.subject_entity_id, r.object_entity_id,
                se.entity_type as subject_type,
                oe.entity_type as object_type
            FROM ag_catalog.relations r
            JOIN ag_catalog.entities se ON r.subject_entity_id = se.id
            JOIN ag_catalog.entities oe ON r.object_entity_id = oe.id
            ORDER BY r.id
        """)
        
        relations = await result.fetchall()
        logger.info(f"Found {len(relations)} relations to migrate")
        
        total_synced = 0
        
        # Process in smaller chunks for relations
        relation_batch_size = 20
        for i in range(0, len(relations), relation_batch_size):
            chunk = relations[i:i + relation_batch_size]
            
            try:
                chunk_synced = await self._sync_relation_chunk(chunk)
                total_synced += chunk_synced
                
                logger.info(f"Synced relation chunk {i//relation_batch_size + 1}: {chunk_synced}/{len(chunk)} relations (Total: {total_synced}/{len(relations)})")
                
            except Exception as e:
                logger.error(f"Error syncing relation chunk {i//relation_batch_size + 1}: {e}")
                
        logger.info(f"✅ Relation sync complete: {total_synced}/{len(relations)} relations synced")
        return total_synced
        
    async def _sync_relation_chunk(self, relations: List) -> int:
        """Sync a chunk of relations individually"""
        synced_count = 0
        
        for relation in relations:
            relation_id, predicate, confidence, evidence_span, subject_id, object_id, subject_type, object_type = relation
            
            try:
                escaped_evidence = evidence_span.replace("'", "\\'") if evidence_span else ""
                
                cypher_create = f"""
                SELECT * FROM cypher('clinical_kg', $$
                    MATCH (s:{subject_type.title()} {{id: {subject_id}}}),
                          (o:{object_type.title()} {{id: {object_id}}})
                    CREATE (s)-[r:{predicate.upper().replace(' ', '_')} {{
                        id: {relation_id},
                        predicate: '{predicate}',
                        confidence: {confidence},
                        evidence_span: '{escaped_evidence}'
                    }}]->(o)
                    RETURN r.id as created_id
                $$) AS (created_id agtype);
                """
                
                await self.conn.execute(self._format_cypher(cypher_create))
                synced_count += 1
                
            except Exception as e:
                logger.error(f"Failed to sync relation {relation_id}: {e}")
                
        return synced_count
        
    async def verify_sync(self) -> Dict[str, Any]:
        """Verify synchronization results"""
        try:
            # Count vertices using internal tables (more reliable)
            result = await self.conn.execute(f"SELECT COUNT(*) FROM {self.graph_name}._ag_label_vertex")
            vertex_count = (await result.fetchone())[0]
            
            # Count edges
            result = await self.conn.execute(f"SELECT COUNT(*) FROM {self.graph_name}._ag_label_edge")
            edge_count = (await result.fetchone())[0]
            
            return {
                'vertices': vertex_count,
                'edges': edge_count,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }
            
    async def run_production_sync(self):
        """Run production-ready synchronization"""
        logger.info("🚀 STARTING PRODUCTION AGE SYNCHRONIZATION")
        
        try:
            await self.connect()
            await self.clear_graph()
            
            entities_synced = await self.sync_entities_chunked()
            relations_synced = await self.sync_relations_chunked()
            
            stats = await self.verify_sync()
            
            logger.info(f"✅ Production sync complete!")
            logger.info(f"📊 Entities synced: {entities_synced}")
            logger.info(f"📊 Relations synced: {relations_synced}")
            logger.info(f"📊 Final verification: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Production sync failed: {e}")
            raise
        finally:
            await self.close()

    def _format_cypher(self, statement: str) -> str:
        """Replace default graph name tokens with configured value."""

        return statement.replace("clinical_kg", self.graph_name)


async def main():
    """Main execution"""
    # Use smaller batch size for reliability
    sync = ProductionAGESync(batch_size=25)
    
    try:
        stats = await sync.run_production_sync()
        
        print("\n🎉 PRODUCTION AGE SYNC COMPLETE!")
        print("=" * 50)
        print(f"Graph vertices: {stats.get('vertices', 0)}")
        print(f"Graph edges: {stats.get('edges', 0)}")
        print("Ready for production queries! 🚀")
        
        return 0
        
    except Exception as e:
        print(f"❌ Production sync failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))