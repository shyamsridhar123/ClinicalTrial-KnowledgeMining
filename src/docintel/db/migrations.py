"""Migration utilities for the DocIntel pgvector schema."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from psycopg import Connection, sql

from ..config import VectorDatabaseSettings

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Migration:
    """Represents a discrete database migration."""

    version: int
    name: str
    apply: Callable[[Connection, VectorDatabaseSettings], None]


def _qualified_identifier(schema: str, name: str) -> sql.Composed:
    return sql.SQL(".").join([sql.Identifier(schema), sql.Identifier(name)])


def _ensure_migrations_table(conn: Connection, settings: VectorDatabaseSettings) -> None:
    table = _qualified_identifier(settings.schema, settings.migrations_table)
    statement = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table} (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    ).format(table=table)
    conn.execute(statement)


def _column_exists(conn: Connection, settings: VectorDatabaseSettings, column: str) -> bool:
    query = sql.SQL(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
          AND column_name = %s
        """
    )
    with conn.cursor() as cursor:
        cursor.execute(query, (settings.schema, settings.embeddings_table, column))
        return cursor.fetchone() is not None


def _get_column_data_type(conn: Connection, settings: VectorDatabaseSettings, column: str) -> Optional[str]:
    query = sql.SQL(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
          AND column_name = %s
        """
    )
    with conn.cursor() as cursor:
        cursor.execute(query, (settings.schema, settings.embeddings_table, column))
        row = cursor.fetchone()
    return None if not row else row[0]


def _get_vector_dimension(conn: Connection, settings: VectorDatabaseSettings) -> Optional[int]:
    table_regclass = f"{settings.schema}.{settings.embeddings_table}"
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = %s::regclass
              AND attname = %s
            """,
            (table_regclass, "embedding"),
        )
        row = cursor.fetchone()

    if not row or row[0] is None:
        return None

    try:
        typmod = int(row[0])
    except (TypeError, ValueError):
        return None

    if typmod < 4:
        return None

    return typmod - 4


def _apply_create_vector_store(conn: Connection, settings: VectorDatabaseSettings) -> None:
    schema_identifier = sql.Identifier(settings.schema)
    table_identifier = _qualified_identifier(settings.schema, settings.embeddings_table)
    dimension_sql = sql.SQL(str(settings.embedding_dimensions))

    conn.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(schema=schema_identifier))

    create_table = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table} (
            embedding_id BIGSERIAL PRIMARY KEY,
            nct_id TEXT NOT NULL,
            document_name TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            segment_index INTEGER NOT NULL DEFAULT 0,
            segment_count INTEGER NOT NULL DEFAULT 1,
            section TEXT,
            token_count INTEGER,
            char_count INTEGER,
            start_word_index INTEGER,
            study_phase TEXT,
            therapeutic_area TEXT,
            document_type TEXT,
            population TEXT,
            endpoint_type TEXT,
            page_reference INTEGER,
            artefact_type TEXT,
            source_path TEXT,
            parent_chunk_id TEXT,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding vector({dimension}) NOT NULL,
            embedding_model TEXT NOT NULL,
            quantization_encoding TEXT NOT NULL DEFAULT 'none',
            quantization_storage_dtype TEXT,
            quantization_scale DOUBLE PRECISION,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    ).format(table=table_identifier, dimension=dimension_sql)
    conn.execute(create_table)

    unique_index_name = sql.Identifier(f"{settings.schema}_{settings.embeddings_table}_chunk_uidx")
    conn.execute(
        sql.SQL(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
                ON {table} (nct_id, document_name, chunk_id)
            """
        ).format(index_name=unique_index_name, table=table_identifier)
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} (therapeutic_area, study_phase, document_type)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_{settings.embeddings_table}_meta_idx"),
            table=table_identifier,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} USING gin (metadata jsonb_path_ops)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_{settings.embeddings_table}_metadata_gin"),
            table=table_identifier,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} USING hnsw (embedding vector_cosine_ops)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_{settings.embeddings_table}_embedding_hnsw"),
            table=table_identifier,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE OR REPLACE FUNCTION {function_name}()
            RETURNS trigger AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        ).format(function_name=_qualified_identifier(settings.schema, "set_updated_at"))
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TRIGGER {trigger_name}
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION {function_name}()
            """
        ).format(
            trigger_name=sql.Identifier(f"{settings.embeddings_table}_set_updated_at"),
            table=table_identifier,
            function_name=_qualified_identifier(settings.schema, "set_updated_at"),
        )
    )


def _apply_extend_embedding_metadata(conn: Connection, settings: VectorDatabaseSettings) -> None:
    table_identifier = _qualified_identifier(settings.schema, settings.embeddings_table)
    dimension_sql = sql.SQL(str(settings.embedding_dimensions))

    if _column_exists(conn, settings, "page_ref"):
        conn.execute(
            sql.SQL("ALTER TABLE {table} RENAME COLUMN {old} TO {new}").format(
                table=table_identifier,
                old=sql.Identifier("page_ref"),
                new=sql.Identifier("page_reference"),
            )
        )

    if _column_exists(conn, settings, "page_reference"):
        column_type = _get_column_data_type(conn, settings, "page_reference")
        if column_type and column_type.lower() != "integer":
            conn.execute(
                sql.SQL(
                    """
                    ALTER TABLE {table}
                    ALTER COLUMN {column} TYPE INTEGER
                    USING (
                        CASE
                            WHEN NULLIF(TRIM({column}), '') IS NULL THEN NULL
                            WHEN TRIM({column}) ~ '^[0-9]+$' THEN TRIM({column})::INTEGER
                            ELSE NULL
                        END
                    )
                    """
                ).format(table=table_identifier, column=sql.Identifier("page_reference"))
            )

    for column_name, data_type in (
        ("page_reference", "INTEGER"),
        ("artefact_type", "TEXT"),
        ("source_path", "TEXT"),
        ("parent_chunk_id", "TEXT"),
    ):
        conn.execute(
            sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {type}").format(
                table=table_identifier,
                column=sql.Identifier(column_name),
                type=sql.SQL(data_type),
            )
        )

    current_dimension = _get_vector_dimension(conn, settings)
    target_dimension = int(settings.embedding_dimensions)
    if current_dimension is not None and current_dimension != target_dimension:
        _LOGGER.info(
            "db | embeddings | resetting stored vectors to change dimension | from=%s | to=%s",
            current_dimension,
            target_dimension,
        )
        conn.execute(sql.SQL("DELETE FROM {table}").format(table=table_identifier))

    conn.execute(
        sql.SQL("ALTER TABLE {table} ALTER COLUMN embedding TYPE vector({dimension})").format(
            table=table_identifier,
            dimension=dimension_sql,
        )
    )


def _apply_create_repository_graph(conn: Connection, settings: VectorDatabaseSettings) -> None:
    schema_identifier = sql.Identifier(settings.schema)
    meta_graphs_table = _qualified_identifier(settings.schema, "meta_graphs")
    meta_graph_assets_table = _qualified_identifier(settings.schema, "meta_graph_assets")
    entities_table = _qualified_identifier(settings.schema, "entities")
    relations_table = _qualified_identifier(settings.schema, "relations")
    repo_nodes_table = _qualified_identifier(settings.schema, "repo_nodes")
    repo_edges_table = _qualified_identifier(settings.schema, "repo_edges")
    repo_links_table = _qualified_identifier(settings.schema, "repo_entity_links")
    tag_summaries_table = _qualified_identifier(settings.schema, "tag_summaries")
    processing_logs_table = _qualified_identifier(settings.schema, "processing_logs")
    updated_at_function = _qualified_identifier(settings.schema, "set_updated_at")
    vector_dimension = sql.SQL(str(settings.embedding_dimensions))

    conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(schema=schema_identifier))
    conn.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"))

    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                meta_graph_id UUID PRIMARY KEY,
                chunk_id UUID NOT NULL,
                nct_id TEXT,
                document_id TEXT,
                graph_type TEXT NOT NULL,
                summary TEXT,
                entity_count INTEGER NOT NULL DEFAULT 0,
                relation_count INTEGER NOT NULL DEFAULT 0,
                processing_metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(table=meta_graphs_table)
    )

    conn.execute(
        sql.SQL(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
                ON {table} (chunk_id)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_meta_graphs_chunk_uidx"),
            table=meta_graphs_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TRIGGER {trigger_name}
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION {function_name}()
            """
        ).format(
            trigger_name=sql.Identifier("meta_graphs_set_updated_at"),
            table=meta_graphs_table,
            function_name=updated_at_function,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                asset_id UUID PRIMARY KEY,
                meta_graph_id UUID NOT NULL REFERENCES {meta_graphs}(meta_graph_id) ON DELETE CASCADE,
                chunk_id UUID,
                asset_kind TEXT NOT NULL,
                asset_ref TEXT,
                caption TEXT,
                page_number INTEGER,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(table=meta_graph_assets_table, meta_graphs=meta_graphs_table)
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} (meta_graph_id, asset_kind)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_meta_graph_assets_meta_kind_idx"),
            table=meta_graph_assets_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                repo_node_id UUID PRIMARY KEY,
                vocabulary TEXT NOT NULL,
                code TEXT NOT NULL,
                display_name TEXT,
                canonical_uri TEXT,
                description TEXT,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                source_version TEXT,
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                valid_until TIMESTAMPTZ,
                checksum TEXT,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(table=repo_nodes_table)
    )

    conn.execute(
        sql.SQL(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
                ON {table} (vocabulary, code)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_repo_nodes_vocab_code_uidx"),
            table=repo_nodes_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TRIGGER {trigger_name}
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION {function_name}()
            """
        ).format(
            trigger_name=sql.Identifier("repo_nodes_set_updated_at"),
            table=repo_nodes_table,
            function_name=updated_at_function,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                repo_edge_id UUID PRIMARY KEY,
                source_repo_node_id UUID NOT NULL REFERENCES {repo_nodes}(repo_node_id) ON DELETE CASCADE,
                target_repo_node_id UUID NOT NULL REFERENCES {repo_nodes}(repo_node_id) ON DELETE CASCADE,
                predicate TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(table=repo_edges_table, repo_nodes=repo_nodes_table)
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} (predicate)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_repo_edges_predicate_idx"),
            table=repo_edges_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TRIGGER {trigger_name}
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION {function_name}()
            """
        ).format(
            trigger_name=sql.Identifier("repo_edges_set_updated_at"),
            table=repo_edges_table,
            function_name=updated_at_function,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                entity_id UUID PRIMARY KEY,
                meta_graph_id UUID NOT NULL REFERENCES {meta_graphs}(meta_graph_id) ON DELETE CASCADE,
                chunk_id UUID NOT NULL,
                entity_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                confidence DOUBLE PRECISION,
                normalized_id TEXT,
                normalized_source TEXT,
                repository_node_id UUID REFERENCES {repo_nodes}(repo_node_id),
                asset_kind TEXT NOT NULL DEFAULT 'text',
                asset_ref TEXT,
                context_flags JSONB,
                normalization_data JSONB,
                provenance JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(table=entities_table, meta_graphs=meta_graphs_table, repo_nodes=repo_nodes_table)
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} (chunk_id, entity_type)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_entities_chunk_type_idx"),
            table=entities_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} (repository_node_id)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_entities_repo_idx"),
            table=entities_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TRIGGER {trigger_name}
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION {function_name}()
            """
        ).format(
            trigger_name=sql.Identifier("entities_set_updated_at"),
            table=entities_table,
            function_name=updated_at_function,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                relation_id UUID PRIMARY KEY,
                meta_graph_id UUID NOT NULL REFERENCES {meta_graphs}(meta_graph_id) ON DELETE CASCADE,
                chunk_id UUID NOT NULL,
                subject_entity_id UUID NOT NULL REFERENCES {entities}(entity_id) ON DELETE CASCADE,
                predicate TEXT NOT NULL,
                object_entity_id UUID NOT NULL REFERENCES {entities}(entity_id) ON DELETE CASCADE,
                confidence DOUBLE PRECISION,
                evidence_span TEXT,
                evidence_start_char INTEGER,
                evidence_end_char INTEGER,
                provenance JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(table=relations_table, meta_graphs=meta_graphs_table, entities=entities_table)
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} (meta_graph_id, predicate)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_relations_meta_predicate_idx"),
            table=relations_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TRIGGER {trigger_name}
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION {function_name}()
            """
        ).format(
            trigger_name=sql.Identifier("relations_set_updated_at"),
            table=relations_table,
            function_name=updated_at_function,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                tag_summary_id UUID PRIMARY KEY,
                meta_graph_id UUID NOT NULL REFERENCES {meta_graphs}(meta_graph_id) ON DELETE CASCADE,
                layer SMALLINT NOT NULL,
                tag_key TEXT NOT NULL,
                tag_label TEXT NOT NULL,
                confidence DOUBLE PRECISION,
                embedding vector({dimension}),
                evidence_ref JSONB,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(table=tag_summaries_table, meta_graphs=meta_graphs_table, dimension=vector_dimension)
    )

    conn.execute(
        sql.SQL(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
                ON {table} (meta_graph_id, layer, tag_key)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_tag_summaries_key_uidx"),
            table=tag_summaries_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                log_id UUID PRIMARY KEY,
                document_id UUID NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(table=processing_logs_table)
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} (document_id, stage)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_processing_logs_doc_stage_idx"),
            table=processing_logs_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                link_id UUID PRIMARY KEY,
                meta_graph_id UUID NOT NULL REFERENCES {meta_graphs}(meta_graph_id) ON DELETE CASCADE,
                chunk_id UUID NOT NULL,
                entity_id UUID NOT NULL REFERENCES {entities}(entity_id) ON DELETE CASCADE,
                repo_node_id UUID NOT NULL REFERENCES {repo_nodes}(repo_node_id) ON DELETE CASCADE,
                vocabulary TEXT,
                code TEXT,
                match_confidence DOUBLE PRECISION,
                match_method TEXT,
                normalization_metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(
            table=repo_links_table,
            meta_graphs=meta_graphs_table,
            entities=entities_table,
            repo_nodes=repo_nodes_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
                ON {table} (entity_id)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_repo_entity_links_entity_uidx"),
            table=repo_links_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} (repo_node_id)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_repo_entity_links_repo_idx"),
            table=repo_links_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
                ON {table} (chunk_id)
            """
        ).format(
            index_name=sql.Identifier(f"{settings.schema}_repo_entity_links_chunk_idx"),
            table=repo_links_table,
        )
    )

    conn.execute(
        sql.SQL(
            """
            CREATE TRIGGER {trigger_name}
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION {function_name}()
            """
        ).format(
            trigger_name=sql.Identifier("repo_entity_links_set_updated_at"),
            table=repo_links_table,
            function_name=updated_at_function,
        )
    )


MIGRATIONS: Sequence[Migration] = (
    Migration(version=1, name="create_vector_store", apply=_apply_create_vector_store),
    Migration(version=2, name="extend_embedding_metadata", apply=_apply_extend_embedding_metadata),
    Migration(version=3, name="create_repository_graph", apply=_apply_create_repository_graph),
)


def _fetch_applied_versions(conn: Connection, settings: VectorDatabaseSettings) -> List[int]:
    table = _qualified_identifier(settings.schema, settings.migrations_table)
    query = sql.SQL("SELECT version FROM {table} ORDER BY version").format(table=table)
    with conn.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
    return [int(row[0]) for row in rows]


def _record_migration(conn: Connection, settings: VectorDatabaseSettings, migration: Migration) -> None:
    table = _qualified_identifier(settings.schema, settings.migrations_table)
    insert = sql.SQL(
        "INSERT INTO {table} (version, name) VALUES (%s, %s)"
    ).format(table=table)
    with conn.cursor() as cursor:
        cursor.execute(insert, (migration.version, migration.name))


def get_pending_migrations(conn: Connection, settings: VectorDatabaseSettings) -> List[Migration]:
    _ensure_migrations_table(conn, settings)
    applied = set(_fetch_applied_versions(conn, settings))
    return [migration for migration in MIGRATIONS if migration.version not in applied]


def apply_migrations(
    conn: Connection,
    settings: VectorDatabaseSettings,
    *,
    target_version: int | None = None,
) -> List[Migration]:
    """Apply migrations up to the requested version."""

    if target_version is not None and target_version < 0:
        raise ValueError("target_version must be a positive integer")

    _ensure_migrations_table(conn, settings)

    applied_versions = set(_fetch_applied_versions(conn, settings))
    applied: List[Migration] = []

    for migration in MIGRATIONS:
        if migration.version in applied_versions:
            continue
        if target_version is not None and migration.version > target_version:
            break

        with conn.transaction():
            _LOGGER.info("db | applying migration | version=%s | name=%s", migration.version, migration.name)
            migration.apply(conn, settings)
            _record_migration(conn, settings, migration)
        applied.append(migration)

    return applied
