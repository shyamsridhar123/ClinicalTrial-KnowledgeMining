"""Database content health check."""

import psycopg
from ..base import HealthCheck, HealthStatus


class DataHealthCheck(HealthCheck):
    """Check database content statistics."""
    
    name = "Database Content"
    timeout_seconds = 10.0
    
    async def _perform_check(self):
        try:
            from docintel.config import get_vector_db_settings
            settings = get_vector_db_settings()
            
            if not settings.enabled or not settings.dsn:
                return (
                    HealthStatus.WARNING,
                    "Database disabled",
                    {"enabled": False}
                )
            
            conn = psycopg.connect(str(settings.dsn), connect_timeout=int(self.timeout_seconds))
            cursor = conn.cursor()
            
            # Use the configured schema (docintel)
            schema = settings.schema
            
            # Count embeddings (these are the chunks)
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.{settings.embeddings_table};")
            embedding_count = cursor.fetchone()[0]
            chunk_count = embedding_count  # Embeddings = chunks
            
            # Count entities
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.entities;")
            entity_count = cursor.fetchone()[0]
            
            # Count relations
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.relations;")
            relation_count = cursor.fetchone()[0]
            
            # Top entity types
            cursor.execute(f"""
                SELECT entity_type, COUNT(*) 
                FROM {schema}.entities 
                GROUP BY entity_type 
                ORDER BY COUNT(*) DESC 
                LIMIT 5;
            """)
            entity_types = dict(cursor.fetchall())
            
            cursor.close()
            conn.close()
            
            if entity_count == 0 and relation_count == 0:
                return (
                    HealthStatus.WARNING,
                    "No data - run pipeline first",
                    {"chunks": chunk_count, "entities": 0, "relations": 0}
                )
            elif entity_count > 0:
                # Have entities - that's the important part
                status_msg = f"{entity_count:,} entities, {relation_count:,} relations"
                if chunk_count == 0:
                    status_msg += " (no chunks yet)"
                
                return (
                    HealthStatus.HEALTHY,
                    status_msg,
                    {
                        "chunks": chunk_count,
                        "embeddings": embedding_count,
                        "entities": entity_count,
                        "relations": relation_count,
                        "top_entity_types": entity_types
                    }
                )
            else:
                return (
                    HealthStatus.WARNING,
                    f"{chunk_count} chunks but no entities extracted",
                    {"chunks": chunk_count, "entities": 0}
                )
        except Exception as e:
            return (
                HealthStatus.ERROR,
                f"Query failed: {str(e)[:100]}",
                {"error": str(e)[:100]}
            )
