from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

import psycopg
from psycopg import sql
from psycopg.extras import execute_batch
from psycopg.types.json import Jsonb

from ..config import DocIntelConfig, get_config
from ..db.migrations import _qualified_identifier
from .loaders import VOCABULARY_LOADERS, VocabularyLoaderResult, VocabularySourceConfig
from .models import ReleaseMetadata, RepoEdgeRecord, RepoNodeRecord

NODE_NAMESPACE = uuid.UUID("c15a2a6d-88a7-5c17-8d6d-46947c2e675e")
EDGE_NAMESPACE = uuid.UUID("2b7bf4f0-76b3-52a0-8576-068078a6f268")


@dataclass(slots=True)
class VocabularyIngestionResult:
    """Summary of an ingestion run for a single vocabulary."""

    vocabulary: str
    nodes_written: int
    edges_written: int
    edges_skipped: int
    release: ReleaseMetadata


class RepositoryIngestor:
    """Coordinate ingestion of external vocabularies into the repository graph."""

    def __init__(self, *, config: Optional[DocIntelConfig] = None) -> None:
        self.config = config or get_config()
        if not self.config.vector_db.dsn:
            raise ValueError("Vector database DSN is not configured; cannot ingest vocabularies.")
        self.settings = self.config.repository_ingestion
        if not self.settings.enabled:
            raise RuntimeError("Repository ingestion is disabled via configuration.")

    def ingest(
        self,
        *,
        sources: Optional[Sequence[str]] = None,
        dry_run: bool = False,
    ) -> List[VocabularyIngestionResult]:
        source_configs = self._resolve_source_configs(sources)
        if not source_configs:
            raise ValueError("No vocabulary sources configured for ingestion.")

        results: List[VocabularyIngestionResult] = []
        with psycopg.connect(self.config.docintel_dsn) as conn:
            for source_config in source_configs:
                loader_cls = VOCABULARY_LOADERS.get(source_config.vocabulary)
                if loader_cls is None:
                    raise ValueError(f"Unsupported vocabulary '{source_config.vocabulary}'.")
                loader = loader_cls(source_config)
                artifacts = loader.load()
                if dry_run:
                    results.append(
                        VocabularyIngestionResult(
                            vocabulary=source_config.vocabulary,
                            nodes_written=len(artifacts.nodes),
                            edges_written=len(artifacts.edges),
                            edges_skipped=0,
                            release=artifacts.metadata,
                        )
                    )
                    continue

                node_codes = {node.code for node in artifacts.nodes}
                nodes_written = self._upsert_nodes(conn, artifacts.nodes)
                edges_written, edges_skipped = self._upsert_edges(conn, artifacts.edges, node_codes)
                self._record_processing_log(conn, artifacts.metadata, nodes_written, edges_written, edges_skipped)
                conn.commit()
                results.append(
                    VocabularyIngestionResult(
                        vocabulary=source_config.vocabulary,
                        nodes_written=nodes_written,
                        edges_written=edges_written,
                        edges_skipped=edges_skipped,
                        release=artifacts.metadata,
                    )
                )
        return results

    def _resolve_source_configs(self, sources: Optional[Sequence[str]]) -> List[VocabularySourceConfig]:
        configured = self.settings.available_sources()
        if sources:
            requested = {src.lower() for src in sources}
            missing = requested.difference(configured)
            if missing:
                raise ValueError(f"Requested vocabularies are not configured: {sorted(missing)}")
            universe = requested
        else:
            universe = configured

        configs: List[VocabularySourceConfig] = []
        for vocab in sorted(universe):
            root, version = self.settings.source_config(vocab)
            if root is None:
                raise ValueError(f"Vocabulary '{vocab}' is not configured with a release path.")
            configs.append(
                VocabularySourceConfig(
                    vocabulary=vocab,
                    root=root,
                    version=version,
                )
            )
        return configs

    def _upsert_nodes(self, conn: psycopg.Connection, nodes: Sequence[RepoNodeRecord]) -> int:
        if not nodes:
            return 0
        table = _qualified_identifier(self.config.vector_db.schema, "repo_nodes")
        insert = sql.SQL(
            """
            INSERT INTO {table} (
                repo_node_id,
                vocabulary,
                code,
                display_name,
                canonical_uri,
                description,
                metadata,
                source_version,
                ingested_at,
                valid_until,
                checksum,
                is_active
            ) VALUES (
                %(repo_node_id)s,
                %(vocabulary)s,
                %(code)s,
                %(display_name)s,
                %(canonical_uri)s,
                %(description)s,
                %(metadata)s,
                %(source_version)s,
                %(ingested_at)s,
                %(valid_until)s,
                %(checksum)s,
                %(is_active)s
            )
            ON CONFLICT (vocabulary, code)
            DO UPDATE SET
                display_name = EXCLUDED.display_name,
                canonical_uri = EXCLUDED.canonical_uri,
                description = EXCLUDED.description,
                metadata = EXCLUDED.metadata,
                source_version = EXCLUDED.source_version,
                ingested_at = EXCLUDED.ingested_at,
                valid_until = EXCLUDED.valid_until,
                checksum = EXCLUDED.checksum,
                is_active = EXCLUDED.is_active,
                updated_at = NOW()
            """
        ).format(table=table)

        payload = []
        now = datetime.now(timezone.utc)
        for record in nodes:
            payload.append(
                {
                    "repo_node_id": self._node_uuid(record.vocabulary, record.code),
                    "vocabulary": record.vocabulary,
                    "code": record.code,
                    "display_name": record.display_name,
                    "canonical_uri": record.canonical_uri,
                    "description": record.description,
                    "metadata": Jsonb(record.metadata),
                    "source_version": record.source_version,
                    "ingested_at": now,
                    "valid_until": None,
                    "checksum": record.checksum,
                    "is_active": record.is_active,
                }
            )

        with conn.cursor() as cursor:
            execute_batch(cursor, insert, payload, page_size=self.settings.batch_size)
        return len(payload)

    def _upsert_edges(
        self,
        conn: psycopg.Connection,
        edges: Sequence[RepoEdgeRecord],
        node_codes: Iterable[str],
    ) -> tuple[int, int]:
        if not edges:
            return 0, 0

        node_code_set = set(node_codes)
        table = _qualified_identifier(self.config.vector_db.schema, "repo_edges")
        insert = sql.SQL(
            """
            INSERT INTO {table} (
                repo_edge_id,
                source_repo_node_id,
                target_repo_node_id,
                predicate,
                metadata
            ) VALUES (
                %(repo_edge_id)s,
                %(source_repo_node_id)s,
                %(target_repo_node_id)s,
                %(predicate)s,
                %(metadata)s
            )
            ON CONFLICT (repo_edge_id)
            DO UPDATE SET
                predicate = EXCLUDED.predicate,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """
        ).format(table=table)

        payload = []
        skipped = 0
        for record in edges:
            if record.source_code not in node_code_set or record.target_code not in node_code_set:
                skipped += 1
                continue
            payload.append(
                {
                    "repo_edge_id": self._edge_uuid(record.vocabulary, record.predicate, record.source_code, record.target_code),
                    "source_repo_node_id": self._node_uuid(record.vocabulary, record.source_code),
                    "target_repo_node_id": self._node_uuid(record.vocabulary, record.target_code),
                    "predicate": record.predicate,
                    "metadata": Jsonb(record.metadata),
                }
            )

        if payload:
            with conn.cursor() as cursor:
                execute_batch(cursor, insert, payload, page_size=self.settings.batch_size)
        return len(payload), skipped

    def _record_processing_log(
        self,
        conn: psycopg.Connection,
        metadata: ReleaseMetadata,
        nodes_written: int,
        edges_written: int,
        edges_skipped: int,
    ) -> None:
        table = _qualified_identifier(self.config.vector_db.schema, "processing_logs")
        statement = sql.SQL(
            """
            INSERT INTO {table} (
                log_id,
                document_id,
                stage,
                status,
                message,
                metadata,
                created_at
            ) VALUES (
                %(log_id)s,
                %(document_id)s,
                %(stage)s,
                %(status)s,
                %(message)s,
                %(metadata)s,
                %(created_at)s
            )
            """
        ).format(table=table)

        log_metadata: Dict[str, object] = {
            "vocabulary": metadata.vocabulary,
            "version": metadata.version,
            "release_checksum": metadata.release_checksum,
            "file_checksums": dict(metadata.file_checksums),
            "nodes_written": nodes_written,
            "edges_written": edges_written,
            "edges_skipped": edges_skipped,
        }

        with conn.cursor() as cursor:
            cursor.execute(
                statement,
                {
                    "log_id": uuid.uuid4(),
                    "document_id": None,
                    "stage": "repository_ingestion",
                    "status": "completed",
                    "message": (
                        f"Ingested vocabulary {metadata.vocabulary} (version={metadata.version}) "
                        f"nodes={nodes_written} edges={edges_written} skipped={edges_skipped}"
                    ),
                    "metadata": Jsonb(log_metadata),
                    "created_at": datetime.now(timezone.utc),
                },
            )

    @staticmethod
    def _node_uuid(vocabulary: str, code: str) -> uuid.UUID:
        return uuid.uuid5(NODE_NAMESPACE, f"{vocabulary}:{code}")

    @staticmethod
    def _edge_uuid(vocabulary: str, predicate: str, source_code: str, target_code: str) -> uuid.UUID:
        raw = f"{vocabulary}:{predicate}:{source_code}->{target_code}"
        return uuid.uuid5(EDGE_NAMESPACE, raw)
