"""
Knowledge Graph Construction Pipeline

Persists clinical entities and relationships extracted from documents
into PostgreSQL + Apache AGE for graph-based querying and retrieval.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from datetime import datetime
from dataclasses import asdict

import psycopg  # type: ignore[import-not-found]
from psycopg import sql  # type: ignore[import-not-found]
from psycopg.rows import dict_row  # type: ignore[import-not-found]
from psycopg.types.json import Json  # type: ignore[import-not-found]

from docintel.config import get_config, get_vector_db_settings
from docintel.knowledge_graph.age_utils import configure_age_session
from docintel.knowledge_graph.triple_extraction import (
    ClinicalEntity, 
    ClinicalRelation, 
    TripleExtractionResult
)

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Builds and persists clinical knowledge graphs in PostgreSQL + AGE."""
    
    def __init__(self):
        config = get_config()
        self.db_dsn = config.docintel_dsn
        self.age_settings = config.age_graph
        vector_settings = get_vector_db_settings()
        self.schema = vector_settings.schema
        self.tables = {
            "meta_graphs": f"{self.schema}.meta_graphs",
            "meta_graph_assets": f"{self.schema}.meta_graph_assets",
            "entities": f"{self.schema}.entities",
            "relations": f"{self.schema}.relations",
            "repo_nodes": f"{self.schema}.repo_nodes",
            "repo_links": f"{self.schema}.repo_entity_links",
            "tag_summaries": f"{self.schema}.tag_summaries",
            "processing_logs": f"{self.schema}.processing_logs",
        }
    
    def _get_connection(self):
        """Get database connection."""
        conn = psycopg.connect(self.db_dsn, row_factory=dict_row)
        if self.age_settings.enabled:
            with conn.cursor() as cur:
                configure_age_session(cur, self.age_settings)
            conn.commit()
        return conn
    
    async def create_meta_graph(self, chunk_id: UUID, extraction_result: TripleExtractionResult, source_chunk_id: Optional[str] = None) -> UUID:
        """Persist the extracted triples as a meta-graph and return its identifier."""

        meta_graph_id = uuid4()
        metadata = dict(extraction_result.processing_metadata or {})
        metadata.setdefault("chunk_id", str(chunk_id))

        # Extract source_chunk_id from metadata if not provided
        if not source_chunk_id:
            source_chunk_id = metadata.get("source_chunk_id")

        nct_id = metadata.get("nct_id")
        document_id = metadata.get("document_id") or metadata.get("document_uuid")
        graph_type = metadata.get("graph_type", "clinical_extraction")
        summary = metadata.get("summary") or (
            f"Extracted {len(extraction_result.entities)} entities and "
            f"{len(extraction_result.relations)} relations from chunk {chunk_id}"
        )

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {self.tables['meta_graphs']} (
                            meta_graph_id,
                            chunk_id,
                            nct_id,
                            document_id,
                            graph_type,
                            summary,
                            entity_count,
                            relation_count,
                            processing_metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            meta_graph_id,
                            chunk_id,
                            nct_id,
                            document_id,
                            graph_type,
                            summary,
                            len(extraction_result.entities),
                            len(extraction_result.relations),
                            json.dumps(metadata),
                        ),
                    )

                    entity_lookup: Dict[str, UUID] = {}
                    repo_links: List[Tuple[object, ...]] = []
                    for entity in extraction_result.entities:
                        entity_id = self._insert_entity(cur, meta_graph_id, chunk_id, entity, source_chunk_id)
                        self._register_entity_lookup(entity_lookup, entity, entity_id)
                        self._collect_repo_link(repo_links, meta_graph_id, chunk_id, entity_id, entity)

                    if repo_links:
                        self._write_repo_links(cur, repo_links)

                    for relation in extraction_result.relations:
                        self._insert_relation(cur, meta_graph_id, chunk_id, relation, entity_lookup)

                    conn.commit()

            logger.info(
                "Created meta-graph %s | chunk=%s | entities=%d | relations=%d",
                meta_graph_id,
                chunk_id,
                len(extraction_result.entities),
                len(extraction_result.relations),
            )
            return meta_graph_id

        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error creating meta-graph for chunk %s: %s", chunk_id, exc)
            raise

    def _insert_entity(
        self,
        cursor,
        meta_graph_id: UUID,
        chunk_id: UUID,
        entity: ClinicalEntity,
        source_chunk_id: Optional[str] = None,
    ) -> UUID:
        """Insert an entity record and return its identifier."""

        entity_id = uuid4()

        context_flags_json: Optional[str] = None
        if getattr(entity, "context_flags", None):
            try:
                context_flags_json = json.dumps(asdict(entity.context_flags))
            except TypeError:
                context_flags_json = json.dumps(entity.context_flags)

        normalization_payload = getattr(entity, "normalization_data", None)
        normalization_json = json.dumps(normalization_payload) if normalization_payload else None

        provenance_payload = getattr(entity, "provenance", None)
        provenance_json = json.dumps(provenance_payload) if provenance_payload else None

        repo_node_uuid = self._safe_uuid(getattr(entity, "repository_node_id", None))
        asset_kind = (getattr(entity, "asset_kind", "text") or "text").lower()
        asset_ref = getattr(entity, "asset_ref", None)

        start_char = getattr(entity, "start_char", getattr(entity, "start_pos", None))
        end_char = getattr(entity, "end_char", getattr(entity, "end_pos", None))

        cursor.execute(
            f"""
            INSERT INTO {self.tables['entities']} (
                entity_id,
                meta_graph_id,
                chunk_id,
                entity_text,
                entity_type,
                start_char,
                end_char,
                confidence,
                normalized_id,
                normalized_source,
                repository_node_id,
                asset_kind,
                asset_ref,
                context_flags,
                normalization_data,
                provenance,
                source_chunk_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                entity_id,
                meta_graph_id,
                chunk_id,
                entity.text,
                entity.entity_type,
                start_char,
                end_char,
                entity.confidence,
                getattr(entity, "normalized_id", None),
                getattr(entity, "normalized_source", None),
                repo_node_uuid,
                asset_kind,
                asset_ref,
                context_flags_json,
                normalization_json,
                provenance_json,
                source_chunk_id,
            ),
        )
        return entity_id

    def _insert_relation(
        self,
        cursor,
        meta_graph_id: UUID,
        chunk_id: UUID,
        relation: ClinicalRelation,
        entity_lookup: Dict[str, UUID],
    ) -> None:
        """Insert a relation if both participating entities were persisted."""

        relation_id = uuid4()
        subject_id = self._resolve_entity_id(entity_lookup, relation.subject_entity)
        object_id = self._resolve_entity_id(entity_lookup, relation.object_entity)

        if not subject_id or not object_id:
            logger.warning(
                "Unable to resolve relation endpoints | predicate=%s | subject=%s | object=%s",
                relation.predicate,
                getattr(relation.subject_entity, "text", None),
                getattr(relation.object_entity, "text", None),
            )
            return

        provenance_payload = getattr(relation, "provenance", None)
        provenance_json = json.dumps(provenance_payload) if provenance_payload else None

        cursor.execute(
            f"""
            INSERT INTO {self.tables['relations']} (
                relation_id,
                meta_graph_id,
                chunk_id,
                subject_entity_id,
                predicate,
                object_entity_id,
                confidence,
                evidence_span,
                evidence_start_char,
                evidence_end_char,
                provenance
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                relation_id,
                meta_graph_id,
                chunk_id,
                subject_id,
                relation.predicate,
                object_id,
                relation.confidence,
                getattr(relation, "evidence_span", None),
                getattr(relation, "evidence_start_char", None),
                getattr(relation, "evidence_end_char", None),
                provenance_json,
            ),
        )

    def _register_entity_lookup(self, lookup: Dict[str, UUID], entity: ClinicalEntity, entity_id: UUID) -> None:
        lookup[str(id(entity))] = entity_id
        key = self._entity_lookup_key(entity)
        if key:
            lookup[key] = entity_id

    def _resolve_entity_id(self, lookup: Dict[str, UUID], entity: ClinicalEntity) -> Optional[UUID]:
        direct = lookup.get(str(id(entity)))
        if direct:
            return direct
        key = self._entity_lookup_key(entity)
        if key:
            return lookup.get(key)
        return None

    def _entity_lookup_key(self, entity: ClinicalEntity) -> Optional[str]:
        text = getattr(entity, "text", None)
        if not text:
            return None
        start = getattr(entity, "start_char", getattr(entity, "start_pos", None))
        end = getattr(entity, "end_char", getattr(entity, "end_pos", None))
        return f"{text.strip().lower()}|{start}|{end}"

    def _safe_uuid(self, value: Any) -> Optional[UUID]:
        if not value:
            return None
        try:
            return UUID(str(value))
        except (TypeError, ValueError):
            logger.debug("Skipping invalid repository node identifier: %s", value)
            return None
    
    async def query_graph(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Execute a SQL query against the knowledge-graph tables."""
        statement = (query or "").strip()
        if not statement:
            return []
        if "limit" not in statement.lower():
            statement = f"{statement.rstrip(';')}\nLIMIT {limit}"
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(statement)
                    return cur.fetchall()
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error("Error executing graph query: %s", exc)
            return []
    
    async def get_entity_neighbors(self, entity_id: UUID, hop_limit: int = 2) -> List[Dict[str, Any]]:
        """Return neighboring entities up to the requested hop limit."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        WITH RECURSIVE entity_paths AS (
                            SELECT
                                e.entity_id,
                                e.entity_text,
                                e.entity_type,
                                e.confidence,
                                0 AS hop_count,
                                ARRAY[e.entity_id] AS path
                            FROM {self.tables['entities']} e
                            WHERE e.entity_id = %s

                            UNION ALL

                            SELECT
                                CASE
                                    WHEN r.subject_entity_id = ep.entity_id THEN r.object_entity_id
                                    ELSE r.subject_entity_id
                                END AS entity_id,
                                e2.entity_text,
                                e2.entity_type,
                                e2.confidence,
                                ep.hop_count + 1,
                                ep.path || CASE
                                    WHEN r.subject_entity_id = ep.entity_id THEN r.object_entity_id
                                    ELSE r.subject_entity_id
                                END
                            FROM entity_paths ep
                            JOIN {self.tables['relations']} r
                              ON r.subject_entity_id = ep.entity_id OR r.object_entity_id = ep.entity_id
                            JOIN {self.tables['entities']} e2
                              ON e2.entity_id = CASE
                                    WHEN r.subject_entity_id = ep.entity_id THEN r.object_entity_id
                                    ELSE r.subject_entity_id
                                END
                            WHERE ep.hop_count < %s
                              AND NOT (
                                    CASE
                                        WHEN r.subject_entity_id = ep.entity_id THEN r.object_entity_id
                                        ELSE r.subject_entity_id
                                    END = ANY(ep.path)
                                )
                        )
                        SELECT DISTINCT
                            entity_id,
                            entity_text,
                            entity_type,
                            confidence,
                            hop_count
                        FROM entity_paths
                        ORDER BY hop_count, confidence DESC NULLS LAST
                        LIMIT 100
                        """,
                        (entity_id, hop_limit),
                    )
                    return cur.fetchall()
        except Exception as exc:
            logger.error("Error getting entity neighbors: %s", exc)
            return []
    
    async def get_chunk_meta_graph(self, chunk_id: UUID) -> Dict[str, Any]:
        """Return the meta-graph snapshot for a specific chunk."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT
                            entity_id,
                            entity_text,
                            entity_type,
                            start_char,
                            end_char,
                            confidence,
                            normalized_id,
                            normalized_source,
                            context_flags,
                            normalization_data,
                            asset_kind,
                            asset_ref,
                            repository_node_id,
                            provenance
                        FROM {self.tables['entities']}
                        WHERE chunk_id = %s
                        ORDER BY start_char NULLS LAST, entity_text
                        """,
                        (chunk_id,),
                    )
                    entities = cur.fetchall()

                    cur.execute(
                        f"""
                        SELECT
                            r.relation_id,
                            r.predicate,
                            r.confidence,
                            r.evidence_span,
                            r.evidence_start_char,
                            r.evidence_end_char,
                            se.entity_text AS subject_text,
                            se.entity_type AS subject_type,
                            oe.entity_text AS object_text,
                            oe.entity_type AS object_type
                        FROM {self.tables['relations']} r
                        JOIN {self.tables['entities']} se ON r.subject_entity_id = se.entity_id
                        JOIN {self.tables['entities']} oe ON r.object_entity_id = oe.entity_id
                        WHERE r.chunk_id = %s
                        ORDER BY r.confidence DESC NULLS LAST, r.predicate
                        """,
                        (chunk_id,),
                    )
                    relations = cur.fetchall()

                    return {
                        "chunk_id": str(chunk_id),
                        "entities": entities,
                        "relations": relations,
                        "entity_count": len(entities),
                        "relation_count": len(relations),
                    }
        except Exception as exc:
            logger.error("Error getting chunk meta-graph: %s", exc)
            return {}
    
    async def log_processing(self, document_id: UUID, stage: str, status: str, 
                           message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log processing status for audit trail."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        INSERT INTO {self.tables['processing_logs']} (document_id, stage, status, message, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        document_id,
                        stage,
                        status,
                        message,
                        json.dumps(metadata) if metadata else None
                    ))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error logging processing status: {e}")

    def _collect_repo_link(
        self,
        repo_links: List[Tuple[object, ...]],
        meta_graph_id: UUID,
        chunk_id: UUID,
        entity_id: UUID,
        entity: ClinicalEntity,
    ) -> None:
        """Collect repository link information for persistence."""

        repo_node_id = getattr(entity, "repository_node_id", None)
        if not repo_node_id:
            return
        vocabulary = getattr(entity, "repository_vocabulary", getattr(entity, "normalized_source", None))
        code = getattr(entity, "repository_code", getattr(entity, "normalized_id", None))
        metadata = getattr(entity, "normalization_data", None)
        repo_links.append(
            (
                str(uuid4()),
                meta_graph_id,
                chunk_id,
                entity_id,
                repo_node_id,
                vocabulary.lower() if isinstance(vocabulary, str) else vocabulary,
                code,
                getattr(entity, "confidence", None),
                "normalization",
                Json(metadata) if metadata is not None else None,
            )
        )

    def _write_repo_links(self, cursor, records: List[Tuple[object, ...]]) -> None:
        insert = sql.SQL(
            """
            INSERT INTO {table} (
                link_id,
                meta_graph_id,
                chunk_id,
                entity_id,
                repo_node_id,
                vocabulary,
                code,
                match_confidence,
                match_method,
                normalization_metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_id) DO UPDATE SET
                repo_node_id = EXCLUDED.repo_node_id,
                vocabulary = EXCLUDED.vocabulary,
                code = EXCLUDED.code,
                match_confidence = EXCLUDED.match_confidence,
                match_method = EXCLUDED.match_method,
                normalization_metadata = EXCLUDED.normalization_metadata,
                updated_at = NOW()
            """
        ).format(table=sql.SQL(self.tables["repo_links"]))
        cursor.executemany(insert, records)

class GraphQueryService:
    """Service for querying the clinical knowledge graph."""
    
    def __init__(self):
        self.graph_builder = KnowledgeGraphBuilder()
    
    async def find_related_entities(self, entity_text: str, entity_type: str = None, 
                                  limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity text.
        
        Args:
            entity_text: Text of the entity to search for
            entity_type: Optional entity type filter
            limit: Maximum number of results
            
        Returns:
            List of related entities
        """
        try:
            with self.graph_builder._get_connection() as conn:
                with conn.cursor() as cur:
                    query = f"""
                        SELECT DISTINCT
                            related.entity_text,
                            related.entity_type,
                            rel.predicate,
                            rel.confidence,
                            rel.evidence_span,
                            docs.nct_id,
                            docs.document_type
                        FROM {self.graph_builder.tables['entities']} target
                        JOIN {self.graph_builder.tables['relations']} rel
                          ON target.entity_id = rel.subject_entity_id
                          OR target.entity_id = rel.object_entity_id
                        JOIN {self.graph_builder.tables['entities']} related
                          ON related.entity_id = CASE
                                WHEN target.entity_id = rel.subject_entity_id THEN rel.object_entity_id
                                ELSE rel.subject_entity_id
                            END
                        JOIN chunks c ON target.chunk_id = c.id
                        JOIN documents docs ON c.document_id = docs.id
                        WHERE target.entity_text ILIKE %s
                    """
                    
                    params = [f"%{entity_text}%"]
                    
                    if entity_type:
                        query += " AND related.entity_type = %s"
                        params.append(entity_type)
                    
                    query += " ORDER BY rel.confidence DESC NULLS LAST LIMIT %s"
                    params.append(limit)
                    
                    cur.execute(query, params)
                    results = cur.fetchall()
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Error finding related entities: {e}")
            return []
    
    async def get_document_summary(self, nct_id: str) -> Dict[str, Any]:
        """Get a summary of entities and relations for a document."""
        try:
            with self.graph_builder._get_connection() as conn:
                with conn.cursor() as cur:
                    # Get entity counts by type
                    cur.execute(f"""
                        SELECT e.entity_type, COUNT(*) as count
                        FROM {self.graph_builder.tables['entities']} e
                        JOIN chunks c ON e.chunk_id = c.id
                        JOIN documents d ON c.document_id = d.id
                        WHERE d.nct_id = %s
                        GROUP BY e.entity_type
                        ORDER BY count DESC
                    """, (nct_id,))
                    entity_counts = cur.fetchall()
                    
                    # Get relation counts by predicate
                    cur.execute(f"""
                        SELECT r.predicate, COUNT(*) as count
                        FROM {self.graph_builder.tables['relations']} r
                        JOIN chunks c ON r.chunk_id = c.id
                        JOIN documents d ON c.document_id = d.id
                        WHERE d.nct_id = %s
                        GROUP BY r.predicate
                        ORDER BY count DESC
                    """, (nct_id,))
                    relation_counts = cur.fetchall()
                    
                    return {
                        "nct_id": nct_id,
                        "entity_counts": entity_counts,
                        "relation_counts": relation_counts
                    }
                    
        except Exception as e:
            logger.error(f"Error getting document summary: {e}")
            return {}

# Backward compatibility alias
MedGraphRAGBuilder = KnowledgeGraphBuilder

# Main processing function
async def process_chunk_to_graph(chunk_id: UUID, text: str) -> UUID:
    """
    Process a text chunk through the complete pipeline: 
    extraction -> graph construction -> persistence.
    
    Args:
        chunk_id: UUID of the chunk
        text: Text content to process
        
    Returns:
        UUID of the created Meta-Graph
    """
    from docintel.knowledge_graph.enhanced_extraction import extract_and_normalize_clinical_data

    # Extract triples with normalization so repository IDs flow into persistence
    extraction_result = await extract_and_normalize_clinical_data(text, chunk_id)
    
    # Build and persist graph
    graph_builder = KnowledgeGraphBuilder()
    meta_graph_id = await graph_builder.create_meta_graph(chunk_id, extraction_result)
    
    return meta_graph_id