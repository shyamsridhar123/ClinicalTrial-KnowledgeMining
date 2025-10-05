"""Utilities to ingest clinical vocabularies into the repository graph."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TYPE_CHECKING

import psycopg
from psycopg import sql

from docintel.repository.constants import EDGE_NAMESPACE, NODE_NAMESPACE

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from docintel.config import RepositoryIngestionSettings

logger = logging.getLogger(__name__)


@dataclass
class RepositoryNode:
    """Normalized node payload for vocabulary concepts."""

    vocabulary: str
    code: str
    display_name: str
    canonical_uri: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    source_version: Optional[str] = None
    valid_until: Optional[datetime] = None
    checksum: Optional[str] = None
    is_active: bool = True

    @property
    def repo_node_id(self) -> uuid.UUID:
        return uuid.uuid5(NODE_NAMESPACE, f"{self.vocabulary}:{self.code}")

    def model_checksum(self, algorithm: str = "sha256") -> str:
        if self.checksum:
            return self.checksum
        try:
            digest = hashlib.new(algorithm)
        except ValueError as exc:  # pragma: no cover - validated earlier
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}") from exc
        payload = {
            "display_name": self.display_name,
            "canonical_uri": self.canonical_uri,
            "description": self.description,
            "metadata": self.metadata,
            "source_version": self.source_version,
            "valid_until": self.valid_until.isoformat() if isinstance(self.valid_until, datetime) else None,
            "is_active": self.is_active,
        }
        digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        self.checksum = digest.hexdigest()
        return self.checksum


@dataclass
class RepositoryEdge:
    """Normalized edge payload describing vocabulary relationships."""

    vocabulary: str
    source_code: str
    target_code: str
    predicate: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def repo_edge_id(self) -> uuid.UUID:
        key = f"{self.vocabulary}:{self.source_code}:{self.predicate}:{self.target_code}"
        return uuid.uuid5(EDGE_NAMESPACE, key)

    @property
    def source_repo_node_id(self) -> uuid.UUID:
        return uuid.uuid5(NODE_NAMESPACE, f"{self.vocabulary}:{self.source_code}")

    @property
    def target_repo_node_id(self) -> uuid.UUID:
        return uuid.uuid5(NODE_NAMESPACE, f"{self.vocabulary}:{self.target_code}")


@dataclass
class RepositoryIngestionResult:
    """Aggregate statistics produced by an ingestion run."""

    vocabulary: str
    nodes_processed: int
    edges_processed: int
    dry_run: bool
    source_version: Optional[str]


class VocabularyLoader:
    """Base loader interface for structured vocabulary sources."""

    vocabulary: str

    def iter_nodes(self) -> Iterator[RepositoryNode]:
        raise NotImplementedError

    def iter_edges(self) -> Iterator[RepositoryEdge]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.vocabulary

    def source_version(self) -> Optional[str]:
        return None


def _open_text(path: Path) -> Iterable[List[str]]:
    handler = gzip.open if path.suffix == ".gz" else open
    with handler(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            yield line.rstrip("\n").rstrip("|").split("|")


class UMLSLoader(VocabularyLoader):
    """Parse UMLS RRF files into repository payloads."""

    vocabulary = "umls"

    def __init__(self, root: Path):
        self.root = root
        self._version_map = self._load_version_map()
        self._semantic_types = self._load_semantic_types()

    def describe(self) -> str:
        return f"UMLS ({self.root})"

    def source_version(self) -> Optional[str]:
        return self._version_map.get("UMLS")

    def iter_nodes(self) -> Iterator[RepositoryNode]:
        path = self.root / "MRCONSO.RRF"
        if not path.exists():
            raise FileNotFoundError(f"Missing MRCONSO.RRF in {self.root}")

        synonyms: Dict[str, set] = defaultdict(set)
        preferred: Dict[str, str] = {}
        source_codes: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

        for row in _open_text(path):
            if len(row) < 15:
                continue
            cui, lat, *_ = row
            if lat.upper() != "ENG":
                continue
            sab = row[11]
            tty = row[12]
            code = row[13]
            label = row[14]
            if not cui or not label:
                continue
            synonyms[cui].add(label)
            if code:
                source_codes[cui][sab].add(code)
            is_pref = row[6] == "Y"
            if cui not in preferred or is_pref or tty in {"PT", "PN", "MH"}:
                preferred[cui] = label

        for cui, label in preferred.items():
            metadata = {
                "synonyms": sorted(synonyms[cui]),
                "semantic_types": sorted(self._semantic_types.get(cui, [])),
                "source_codes": {sab: sorted(values) for sab, values in source_codes[cui].items()},
                "version_map": self._version_map,
            }
            yield RepositoryNode(
                vocabulary=self.vocabulary,
                code=cui,
                display_name=label,
                canonical_uri=f"https://uts.nlm.nih.gov/umls/concept/{cui}",
                description=None,
                metadata=metadata,
                source_version=self.source_version(),
                valid_until=None,
            )

    def iter_edges(self) -> Iterator[RepositoryEdge]:
        path = self.root / "MRREL.RRF"
        if not path.exists():
            return iter(())

        def generator() -> Iterator[RepositoryEdge]:
            for row in _open_text(path):
                if len(row) < 11:
                    continue
                cui1 = row[0]
                rel = row[3]
                cui2 = row[4]
                rela = row[7] if len(row) > 7 else ""
                sab = row[10]
                if not cui1 or not cui2:
                    continue
                predicate = rela or rel or "related_to"
                metadata = {"rel": rel, "rela": rela, "sab": sab}
                yield RepositoryEdge(
                    vocabulary=self.vocabulary,
                    source_code=cui1,
                    target_code=cui2,
                    predicate=predicate,
                    metadata=metadata,
                )

        return generator()

    def _load_semantic_types(self) -> Dict[str, set]:
        path = self.root / "MRSTY.RRF"
        semantic_types: Dict[str, set] = defaultdict(set)
        if not path.exists():
            return semantic_types
        for row in _open_text(path):
            if len(row) < 4:
                continue
            cui = row[0]
            sty = row[3]
            if cui and sty:
                semantic_types[cui].add(sty)
        return semantic_types

    def _load_version_map(self) -> Dict[str, str]:
        path = self.root / "MRSAB.RRF"
        version_map: Dict[str, str] = {}
        if not path.exists():
            return version_map
        for row in _open_text(path):
            if len(row) < 7:
                continue
            vsab = row[2]
            rsab = row[3]
            sver = row[6]
            if rsab:
                version_map[rsab.upper()] = sver or vsab
        return version_map


class RxNormLoader(VocabularyLoader):
    """Parse RxNorm RRF exports."""

    vocabulary = "rxnorm"

    def __init__(self, root: Path):
        self.root = root
        self._version = self._load_version()

    def describe(self) -> str:
        return f"RxNorm ({self.root})"

    def source_version(self) -> Optional[str]:
        return self._version

    def iter_nodes(self) -> Iterator[RepositoryNode]:
        path = self.root / "RXNCONSO.RRF"
        if not path.exists():
            raise FileNotFoundError(f"Missing RXNCONSO.RRF in {self.root}")

        synonyms: Dict[str, set] = defaultdict(set)
        preferred: Dict[str, str] = {}

        for row in _open_text(path):
            if len(row) < 15:
                continue
            rxcui = row[0]
            lat = row[1]
            if lat.upper() != "ENG":
                continue
            label = row[14]
            tty = row[12]
            if not rxcui or not label:
                continue
            synonyms[rxcui].add(label)
            is_pref = row[6] == "Y"
            if rxcui not in preferred or is_pref or tty in {"SCD", "SBD", "BPCK", "GPCK"}:
                preferred[rxcui] = label

        related_synonyms = self._load_related_synonyms(synonyms)

        for rxcui, label in preferred.items():
            aggregated_synonyms = set(synonyms[rxcui]) | set(related_synonyms.get(rxcui, set()))
            metadata = {"synonyms": sorted(aggregated_synonyms)}
            yield RepositoryNode(
                vocabulary=self.vocabulary,
                code=rxcui,
                display_name=label,
                canonical_uri=f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}",
                description=None,
                metadata=metadata,
                source_version=self._version,
                valid_until=None,
            )

    def _load_related_synonyms(self, synonyms: Dict[str, set]) -> Dict[str, set]:
        path = self.root / "RXNREL.RRF"
        related: Dict[str, set] = defaultdict(set)
        if not path.exists():
            return related
        for row in _open_text(path):
            if len(row) < 5:
                continue
            source = row[0]
            rel = row[3]
            target = row[4]
            if not source or not target:
                continue
            relation = rel.upper()
            if relation in {"PAR", "CHD"}:
                if target in synonyms:
                    related[source].update(synonyms[target])
                if source in synonyms:
                    related[target].update(synonyms[source])
        return related

    def iter_edges(self) -> Iterator[RepositoryEdge]:
        path = self.root / "RXNREL.RRF"
        if not path.exists():
            return iter(())

        def generator() -> Iterator[RepositoryEdge]:
            for row in _open_text(path):
                if len(row) < 11:
                    continue
                source = row[0]
                rel = row[3]
                target = row[4]
                rela = row[7] if len(row) > 7 else ""
                if not source or not target:
                    continue
                predicate = rela or rel or "related_to"
                metadata = {"rel": rel, "rela": rela, "sab": row[10]}
                yield RepositoryEdge(
                    vocabulary=self.vocabulary,
                    source_code=source,
                    target_code=target,
                    predicate=predicate,
                    metadata=metadata,
                )

        return generator()

    def _load_version(self) -> Optional[str]:
        path = self.root / "RXNSAB.RRF"
        if not path.exists():
            return None
        for row in _open_text(path):
            if len(row) < 7:
                continue
            rsab = row[3]
            sver = row[6]
            if rsab and rsab.upper() == "RXNORM":
                return sver or row[2]
        return None


class SnomedLoader(VocabularyLoader):
    """Parse SNOMED CT RF2 release files."""

    vocabulary = "snomed"

    def __init__(self, root: Path):
        self.root = root
        self._concepts_file = self._find_latest("sct2_Concept_", "Concept")
        self._description_file = self._find_latest("sct2_Description_", "Description")
        self._relationship_file = self._find_latest("sct2_Relationship_", "Relationship")
        self._effective_date = None

    def describe(self) -> str:
        return f"SNOMED ({self.root})"

    def source_version(self) -> Optional[str]:
        return self._effective_date

    def iter_nodes(self) -> Iterator[RepositoryNode]:
        concept_file = self._concepts_file
        description_file = self._description_file
        if concept_file is None or description_file is None:
            raise FileNotFoundError("SNOMED concept and description files are required")

        active_concepts: Dict[str, Dict[str, object]] = {}
        with open(concept_file, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            headers = next(reader)
            idx = {name: position for position, name in enumerate(headers)}
            for row in reader:
                concept_id = row[idx["id"]]
                active = row[idx["active"]] == "1"
                effective_time = row[idx["effectiveTime"]]
                if self._effective_date is None and effective_time:
                    self._effective_date = effective_time
                if not concept_id:
                    continue
                active_concepts[concept_id] = {
                    "active": active,
                    "moduleId": row[idx.get("moduleId", 0)],
                }

        descriptions: Dict[str, Dict[str, str]] = defaultdict(dict)
        with open(description_file, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            headers = next(reader)
            idx = {name: position for position, name in enumerate(headers)}
            for row in reader:
                concept_id = row[idx["conceptId"]]
                term = row[idx["term"]]
                type_id = row[idx["typeId"]]
                if not concept_id or concept_id not in active_concepts:
                    continue
                descriptions[concept_id].setdefault(type_id, term)

        fsn_type = "900000000000003001"
        synonym_type = "900000000000013009"

        for concept_id, concept_meta in active_concepts.items():
            if not concept_meta.get("active", False):
                continue
            term = descriptions[concept_id].get(fsn_type) or descriptions[concept_id].get(synonym_type)
            if not term:
                continue
            synonyms = [value for key, value in descriptions[concept_id].items() if key != fsn_type]
            metadata = {
                "synonyms": sorted(set(synonyms)),
                "module_id": concept_meta.get("moduleId"),
            }
            yield RepositoryNode(
                vocabulary=self.vocabulary,
                code=concept_id,
                display_name=term,
                canonical_uri=f"https://snomedbrowser.com/Codes/Details/{concept_id}",
                description=None,
                metadata=metadata,
                source_version=self._effective_date,
                valid_until=None,
                is_active=concept_meta.get("active", False),
            )

    def iter_edges(self) -> Iterator[RepositoryEdge]:
        relationship_file = self._relationship_file
        if relationship_file is None:
            return iter(())

        def generator() -> Iterator[RepositoryEdge]:
            with open(relationship_file, "r", encoding="utf-8") as handle:
                reader = csv.reader(handle, delimiter="\t")
                headers = next(reader)
                idx = {name: position for position, name in enumerate(headers)}
                for row in reader:
                    active = row[idx["active"]] == "1"
                    if not active:
                        continue
                    source_id = row[idx["sourceId"]]
                    target_id = row[idx["destinationId"]]
                    type_id = row[idx["typeId"]]
                    if not source_id or not target_id:
                        continue
                    metadata = {
                        "characteristicTypeId": row[idx.get("characteristicTypeId", 0)],
                        "modifierId": row[idx.get("modifierId", 0)],
                    }
                    yield RepositoryEdge(
                        vocabulary=self.vocabulary,
                        source_code=source_id,
                        target_code=target_id,
                        predicate=type_id,
                        metadata=metadata,
                    )

        return generator()

    def _find_latest(self, prefix: str, label: str) -> Optional[Path]:
        candidates = sorted(self.root.glob(f"{prefix}*"))
        for path in candidates:
            if path.is_file():
                return path
        logger.warning("SNOMED %s file not found under %s", label, self.root)
        return None


class RepositoryIngestionOrchestrator:
    """Coordinate loading vocabularies into the repository tables."""

    def __init__(
        self,
        conn: Optional[psycopg.Connection],
        schema: str,
        *,
        batch_size: int = 2000,
        edge_batch_size: int = 4000,
        checksum_algorithm: str = "sha256",
        processing_stage: str = "repository_ingestion",
    ) -> None:
        self.conn = conn
        self.schema = schema
        self.batch_size = batch_size
        self.edge_batch_size = edge_batch_size
        self.checksum_algorithm = checksum_algorithm
        self.processing_stage = processing_stage
        self.repo_nodes_table = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier("repo_nodes"))
        self.repo_edges_table = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier("repo_edges"))
        self.processing_logs_table = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier("processing_logs"))

    def ingest(
        self,
        loader: VocabularyLoader,
        *,
        dry_run: bool = False,
        node_limit: Optional[int] = None,
        edge_limit: Optional[int] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> RepositoryIngestionResult:
        nodes_iter = loader.iter_nodes()
        edges_iter = loader.iter_edges()

        if dry_run:
            nodes = self._take(nodes_iter, node_limit)
            edges = self._take(edges_iter, edge_limit)
            logger.info(
                "Dry-run: %s produced %d nodes and %d edges",
                loader.describe(),
                len(nodes),
                len(edges),
            )
            if self.conn is not None:
                payload = {
                    "dry_run": True,
                    "nodes": len(nodes),
                    "edges": len(edges),
                    "source_version": loader.source_version(),
                }
                if metadata:
                    payload.update(metadata)
                self._record_processing_log(
                    loader.vocabulary,
                    status="completed",
                    message=f"Repository dry-run completed for {loader.describe()}",
                    metadata=payload,
                )
            return RepositoryIngestionResult(
                vocabulary=loader.vocabulary,
                nodes_processed=len(nodes),
                edges_processed=len(edges),
                dry_run=True,
                source_version=loader.source_version(),
            )

        if self.conn is None:
            raise RuntimeError("Database connection is required when dry_run is False")

        start_metadata = {
            "dry_run": False,
            "source_version": loader.source_version(),
        }
        if metadata:
            start_metadata.update(metadata)
        self._record_processing_log(
            loader.vocabulary,
            status="started",
            message=f"Repository ingestion started for {loader.describe()}",
            metadata=start_metadata,
        )

        node_count = self._write_nodes(nodes_iter)
        edge_count = self._write_edges(edges_iter)
        completion_metadata = {
            "nodes": node_count,
            "edges": edge_count,
            "source_version": loader.source_version(),
        }
        if metadata:
            completion_metadata.update(metadata)
        self._record_processing_log(
            loader.vocabulary,
            status="completed",
            message=f"Repository ingestion completed for {loader.describe()}",
            metadata=completion_metadata,
        )
        logger.info(
            "Ingested %d nodes and %d edges for %s",
            node_count,
            edge_count,
            loader.describe(),
        )
        return RepositoryIngestionResult(
            vocabulary=loader.vocabulary,
            nodes_processed=node_count,
            edges_processed=edge_count,
            dry_run=False,
            source_version=loader.source_version(),
        )

    def _write_nodes(self, iterator: Iterator[RepositoryNode]) -> int:
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
                valid_until,
                checksum,
                is_active
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
            ON CONFLICT (vocabulary, code) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                canonical_uri = EXCLUDED.canonical_uri,
                description = EXCLUDED.description,
                metadata = EXCLUDED.metadata,
                source_version = EXCLUDED.source_version,
                valid_until = EXCLUDED.valid_until,
                checksum = EXCLUDED.checksum,
                is_active = EXCLUDED.is_active,
                updated_at = NOW()
            """
        ).format(table=self.repo_nodes_table)

        total = 0
        with self.conn.cursor() as cursor:
            batch: List[Tuple[object, ...]] = []
            for node in iterator:
                total += 1
                batch.append(
                    (
                        str(node.repo_node_id),
                        node.vocabulary,
                        node.code,
                        node.display_name,
                        node.canonical_uri,
                        node.description,
                        json.dumps(node.metadata, sort_keys=True),
                        node.source_version,
                        node.valid_until,
                        node.model_checksum(self.checksum_algorithm),
                        node.is_active,
                    )
                )
                if len(batch) >= self.batch_size:
                    cursor.executemany(insert, batch)
                    batch.clear()
            if batch:
                cursor.executemany(insert, batch)
        self.conn.commit()
        return total

    def _write_edges(self, iterator: Iterator[RepositoryEdge]) -> int:
        insert = sql.SQL(
            """
            INSERT INTO {table} (
                repo_edge_id,
                source_repo_node_id,
                target_repo_node_id,
                predicate,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (repo_edge_id) DO UPDATE SET
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """
        ).format(table=self.repo_edges_table)

        total = 0
        with self.conn.cursor() as cursor:
            batch: List[Tuple[object, ...]] = []
            for edge in iterator:
                total += 1
                batch.append(
                    (
                        str(edge.repo_edge_id),
                        str(edge.source_repo_node_id),
                        str(edge.target_repo_node_id),
                        edge.predicate,
                        json.dumps(edge.metadata, sort_keys=True),
                    )
                )
                if len(batch) >= self.edge_batch_size:
                    cursor.executemany(insert, batch)
                    batch.clear()
            if batch:
                cursor.executemany(insert, batch)
        self.conn.commit()
        return total

    def _record_processing_log(
        self,
        vocabulary: str,
        *,
        status: str,
        message: str,
        metadata: Dict[str, object],
    ) -> None:
        # Skip logging for vocabulary ingestion - processing_logs is for document processing
        # Vocabulary ingestion uses repo_nodes/repo_edges tables instead
        logger.debug(f"Vocabulary {vocabulary} status: {status} - {message}")
        return

    @staticmethod
    def _take(iterator: Iterator, limit: Optional[int]) -> List:
        if limit is None:
            return list(iterator)
        return list(islice(iterator, limit))


def build_loaders(settings: "RepositoryIngestionSettings") -> List[VocabularyLoader]:
    loaders: List[VocabularyLoader] = []
    
    # Check if medspaCy vocabularies should be used instead of licensed ones
    if settings.use_medspacy_vocabularies:
        try:
            from .medspacy_loaders import build_medspacy_loaders
            medspacy_loaders = build_medspacy_loaders(settings.resolve_path(settings.medspacy_quickumls_path))
            loaders.extend(medspacy_loaders)
            logger.info(f"Using medspaCy-based vocabularies: {len(medspacy_loaders)} loaders")
            return loaders
        except ImportError as e:
            logger.warning(f"medspaCy loaders not available: {e}. Falling back to licensed vocabularies.")
    
    # Fallback to licensed vocabularies if available
    if settings.umls_source_root:
        umls_path = settings.resolve_path(settings.umls_source_root)
        if umls_path:
            loaders.append(UMLSLoader(umls_path))
    if settings.rxnorm_source_root:
        rxnorm_path = settings.resolve_path(settings.rxnorm_source_root)
        if rxnorm_path:
            loaders.append(RxNormLoader(rxnorm_path))
    if settings.snomed_source_root:
        snomed_path = settings.resolve_path(settings.snomed_source_root)
        if snomed_path:
            loaders.append(SnomedLoader(snomed_path))
    
    # If no loaders available, try mock loaders for development
    if not loaders:
        logger.warning("No vocabulary sources available. Consider using mock loaders for development.")
        try:
            from .mock_loaders import MockUMLSLoader, MockRxNormLoader, MockSnomedLoader
            loaders = [MockUMLSLoader(), MockRxNormLoader(), MockSnomedLoader()]
            logger.info("Using mock vocabulary loaders for development")
        except ImportError:
            logger.error("No vocabulary loaders available")
    
    return loaders
