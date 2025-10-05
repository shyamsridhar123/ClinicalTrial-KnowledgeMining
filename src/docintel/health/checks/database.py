"""Database health check."""

import psycopg
from ..base import HealthCheck, HealthStatus


class DatabaseHealthCheck(HealthCheck):
    """Check PostgreSQL + pgvector + Apache AGE."""
    
    name = "PostgreSQL Database"
    timeout_seconds = 5.0
    
    async def _perform_check(self):
        try:
            from docintel.config import get_vector_db_settings
            settings = get_vector_db_settings()
            
            if not settings.enabled or not settings.dsn:
                return (
                    HealthStatus.WARNING,
                    "Vector database disabled in config",
                    {"enabled": False}
                )
            
            conn = psycopg.connect(str(settings.dsn), connect_timeout=int(self.timeout_seconds))
            cursor = conn.cursor()
            
            # PostgreSQL version
            cursor.execute("SELECT version();")
            pg_version = cursor.fetchone()[0].split()[1]
            
            # Check extensions
            cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            has_pgvector = cursor.fetchone()[0]
            
            cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'age');")
            has_age = cursor.fetchone()[0]
            
            # Check clinical_graph
            has_graph = False
            if has_age:
                cursor.execute(
                    "SELECT EXISTS(SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'clinical_graph');"
                )
                has_graph = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            issues = []
            if not has_pgvector:
                issues.append("pgvector missing")
            if not has_age:
                issues.append("AGE missing")
            if has_age and not has_graph:
                issues.append("clinical_graph not created")
            
            if issues:
                return (
                    HealthStatus.WARNING,
                    f"Connected but: {', '.join(issues)}",
                    {
                        "version": pg_version,
                        "pgvector": has_pgvector,
                        "age": has_age,
                        "clinical_graph": has_graph
                    }
                )
            else:
                return (
                    HealthStatus.HEALTHY,
                    f"PostgreSQL {pg_version} with all extensions",
                    {
                        "version": pg_version,
                        "pgvector": True,
                        "age": True,
                        "clinical_graph": True,
                        "schema": settings.schema
                    }
                )
        except Exception as e:
            return (
                HealthStatus.ERROR,
                f"Cannot connect: {str(e)[:100]}",
                {"error": str(e)[:100]}
            )
