from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .models import ReleaseMetadata, RepoEdgeRecord, RepoNodeRecord

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VocabularySourceConfig:
    """Configuration needed to load a vocabulary release from disk."""

    vocabulary: str
    root: Path
    version: Optional[str]


@dataclass(slots=True)
class VocabularyLoaderResult:
    """Container for normalized records extracted from a vocabulary release."""

    nodes: List[RepoNodeRecord]
    edges: List[RepoEdgeRecord]
    metadata: ReleaseMetadata


def _open_text_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Expected file is missing: {path}")
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", errors="ignore")
    return path.open(mode="r", encoding="utf-8", errors="ignore")


def _compute_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_checksum(file_checksums: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(file_checksums):
        digest.update(name.encode())
        digest.update(b"=")
        digest.update(file_checksums[name].encode())
    return digest.hexdigest()


class UMLSLoader:
    """Extract concepts and relationships from a UMLS RRF release."""

    REQUIRED_FILES = ("MRCONSO.RRF", "MRREL.RRF")

    def __init__(self, config: VocabularySourceConfig) -> None:
        self.config = config

    def load(self) -> VocabularyLoaderResult:
        root = self.config.root
        for filename in self.REQUIRED_FILES:
            candidate = root / filename
            if not candidate.exists():
                gz_candidate = candidate.with_suffix(candidate.suffix + ".gz")
                if gz_candidate.exists():
                    continue
                raise FileNotFoundError(f"UMLS release missing required file: {candidate}")

        term_path = self._available_path(root / "MRCONSO.RRF")
        rel_path = self._available_path(root / "MRREL.RRF")
        sty_path = self._available_path(root / "MRSTY.RRF", optional=True)
        def_path = self._available_path(root / "MRDEF.RRF", optional=True)

        file_checksums: Dict[str, str] = {
            term_path.name: _compute_checksum(term_path),
            rel_path.name: _compute_checksum(rel_path),
        }
        if sty_path:
            file_checksums[sty_path.name] = _compute_checksum(sty_path)
        if def_path:
            file_checksums[def_path.name] = _compute_checksum(def_path)

        preferred: Dict[str, str] = {}
        synonyms: Dict[str, set[str]] = defaultdict(set)
        sab_map: Dict[str, set[str]] = defaultdict(set)

        with _open_text_file(term_path) as handle:
            for line in handle:
                parts = line.rstrip("|\n").split("|")
                if len(parts) < 15:
                    continue
                cui = parts[0]
                lat = parts[1]
                term_status = parts[2]
                is_pref = parts[6]
                sab = parts[11]
                tty = parts[12]
                string = parts[14].strip()
                if lat != "ENG" or not string:
                    continue
                synonyms[cui].add(string)
                sab_map[cui].add(sab)
                if is_pref == "Y" or tty in {"PT", "PN", "PF"} or term_status == "P":
                    if cui not in preferred:
                        preferred[cui] = string

        semantic_types: Dict[str, List[str]] = defaultdict(list)
        if sty_path:
            with _open_text_file(sty_path) as handle:
                for line in handle:
                    parts = line.rstrip("|\n").split("|")
                    if len(parts) < 3:
                        continue
                    cui = parts[0]
                    sty = parts[2]
                    if sty:
                        semantic_types[cui].append(sty)

        definitions: Dict[str, str] = {}
        if def_path:
            with _open_text_file(def_path) as handle:
                for line in handle:
                    parts = line.rstrip("|\n").split("|")
                    if len(parts) < 6:
                        continue
                    cui = parts[0]
                    definition = parts[5].strip()
                    if cui not in definitions and definition:
                        definitions[cui] = definition

        edges: List[RepoEdgeRecord] = []
        with _open_text_file(rel_path) as handle:
            for line in handle:
                parts = line.rstrip("|\n").split("|")
                if len(parts) < 16:
                    continue
                cui1 = parts[0]
                cui2 = parts[4]
                rel = parts[3]
                rela = parts[7]
                sab = parts[10]
                if not cui1 or not cui2:
                    continue
                predicate = (rela or rel or "associated_with").lower()
                metadata = {
                    "rel": rel or None,
                    "rela": rela or None,
                    "sab": sab or None,
                    "stype1": parts[2] or None,
                    "stype2": parts[6] or None,
                }
                metadata = {k: v for k, v in metadata.items() if v}
                edges.append(
                    RepoEdgeRecord(
                        vocabulary="umls",
                        predicate=predicate,
                        source_code=cui1,
                        target_code=cui2,
                        metadata=metadata,
                    )
                )

        nodes: List[RepoNodeRecord] = []
        for cui, terms in synonyms.items():
            preferred_term = preferred.get(cui)
            if not preferred_term:
                preferred_term = sorted(terms)[0]
            node_metadata = {
                "synonyms": sorted(t for t in terms if t != preferred_term),
                "semantic_types": sorted(set(semantic_types.get(cui, []))),
                "sources": sorted(sab_map.get(cui, [])),
            }
            description = definitions.get(cui)
            checksum = hashlib.sha256(
                f"umls|{cui}|{preferred_term}|{description or ''}|{json.dumps(node_metadata, sort_keys=True)}".encode()
            ).hexdigest()
            nodes.append(
                RepoNodeRecord(
                    vocabulary="umls",
                    code=cui,
                    display_name=preferred_term,
                    canonical_uri=None,
                    description=description,
                    metadata=node_metadata,
                    source_version=self.config.version or "unknown",
                    checksum=checksum,
                )
            )

        metadata = ReleaseMetadata.create(
            vocabulary="umls",
            version=self.config.version or "unknown",
            release_checksum=_release_checksum(file_checksums),
            file_checksums=file_checksums,
            total_concepts=len(nodes),
            total_relationships=len(edges),
        )
        return VocabularyLoaderResult(nodes=nodes, edges=edges, metadata=metadata)

    @staticmethod
    def _available_path(path: Path, *, optional: bool = False) -> Optional[Path]:
        if path.exists():
            return path
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            return gz_path
        if optional:
            return None
        raise FileNotFoundError(f"Required UMLS artefact not found: {path}")


class RxNormLoader:
    """Extract concepts and relationships from a RxNorm RRF release."""

    REQUIRED_FILES = ("RXNCONSO.RRF",)

    def __init__(self, config: VocabularySourceConfig) -> None:
        self.config = config

    def load(self) -> VocabularyLoaderResult:
        root = self.config.root
        term_path = self._available_path(root / "RXNCONSO.RRF")
        rel_path = self._available_path(root / "RXNREL.RRF", optional=True)

        file_checksums: Dict[str, str] = {term_path.name: _compute_checksum(term_path)}
        if rel_path:
            file_checksums[rel_path.name] = _compute_checksum(rel_path)

        preferred: Dict[str, str] = {}
        synonyms: Dict[str, set[str]] = defaultdict(set)
        tty_map: Dict[str, set[str]] = defaultdict(set)
        ingredients: Dict[str, set[str]] = defaultdict(set)

        with _open_text_file(term_path) as handle:
            for line in handle:
                parts = line.rstrip("|\n").split("|")
                if len(parts) < 15:
                    continue
                rxcui = parts[0]
                lat = parts[1]
                term_type = parts[12]
                string = parts[14].strip()
                if lat != "ENG" or not string:
                    continue
                synonyms[rxcui].add(string)
                tty_map[rxcui].add(term_type)
                if term_type in {"PIN", "SCD", "SBD", "GPCK", "BPCK"}:
                    preferred.setdefault(rxcui, string)
                if term_type == "IN" and string:
                    ingredients[rxcui].add(string)

        nodes: List[RepoNodeRecord] = []
        for rxcui, terms in synonyms.items():
            preferred_term = preferred.get(rxcui, sorted(terms)[0])
            metadata = {
                "synonyms": sorted(t for t in terms if t != preferred_term),
                "term_types": sorted(tty_map.get(rxcui, [])),
            }
            checksum = hashlib.sha256(
                f"rxnorm|{rxcui}|{preferred_term}|{json.dumps(metadata, sort_keys=True)}".encode()
            ).hexdigest()
            nodes.append(
                RepoNodeRecord(
                    vocabulary="rxnorm",
                    code=rxcui,
                    display_name=preferred_term,
                    canonical_uri=f"https://www.nlm.nih.gov/research/umls/rxnorm/docs/termtypes.html#{rxcui}",
                    description=None,
                    metadata=metadata,
                    source_version=self.config.version or "unknown",
                    checksum=checksum,
                )
            )

        edges: List[RepoEdgeRecord] = []
        if rel_path:
            with _open_text_file(rel_path) as handle:
                for line in handle:
                    parts = line.rstrip("|\n").split("|")
                    if len(parts) < 16:
                        continue
                    cui1 = parts[0]
                    cui2 = parts[4]
                    rel = parts[3]
                    rela = parts[7]
                    if not cui1 or not cui2:
                        continue
                    predicate = (rela or rel or "related_to").lower()
                    metadata = {
                        "rel": rel or None,
                        "rela": rela or None,
                        "sab": parts[10] or None,
                    }
                    metadata = {k: v for k, v in metadata.items() if v}
                    edges.append(
                        RepoEdgeRecord(
                            vocabulary="rxnorm",
                            predicate=predicate,
                            source_code=cui1,
                            target_code=cui2,
                            metadata=metadata,
                        )
                    )

        metadata = ReleaseMetadata.create(
            vocabulary="rxnorm",
            version=self.config.version or "unknown",
            release_checksum=_release_checksum(file_checksums),
            file_checksums=file_checksums,
            total_concepts=len(nodes),
            total_relationships=len(edges),
        )
        return VocabularyLoaderResult(nodes=nodes, edges=edges, metadata=metadata)

    @staticmethod
    def _available_path(path: Path, *, optional: bool = False) -> Optional[Path]:
        if path.exists():
            return path
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            return gz_path
        if optional:
            return None
        raise FileNotFoundError(f"Required RxNorm artefact not found: {path}")


class SnomedLoader:
    """Extract concepts and relationships from a SNOMED CT RF2 release."""

    def __init__(self, config: VocabularySourceConfig) -> None:
        self.config = config

    def load(self) -> VocabularyLoaderResult:
        root = self.config.root
        concept_file = self._prefer_file(root, "sct2_Concept", suffix=".txt")
        description_file = self._prefer_file(root, "sct2_Description", suffix=".txt")
        relationship_file = self._prefer_file(root, "sct2_Relationship", suffix=".txt", optional=True)

        file_checksums: Dict[str, str] = {}
        for path in filter(None, [concept_file, description_file, relationship_file]):
            file_checksums[path.name] = _compute_checksum(path)

        active_concepts: Dict[str, str] = {}
        if concept_file:
            with _open_text_file(concept_file) as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader, None)
                if not header:
                    raise ValueError("SNOMED concept file missing header")
                id_idx = header.index("id")
                active_idx = header.index("active")
                module_idx = header.index("moduleId") if "moduleId" in header else None
                for row in reader:
                    concept_id = row[id_idx]
                    is_active = row[active_idx] == "1"
                    if not concept_id or not is_active:
                        continue
                    if module_idx is not None:
                        active_concepts[concept_id] = row[module_idx]
                    else:
                        active_concepts[concept_id] = "core"

        preferred_terms: Dict[str, str] = {}
        synonyms: Dict[str, set[str]] = defaultdict(set)
        descriptions: Dict[str, str] = {}
        if description_file:
            with _open_text_file(description_file) as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader, None)
                if not header:
                    raise ValueError("SNOMED description file missing header")
                concept_idx = header.index("conceptId")
                type_idx = header.index("typeId")
                term_idx = header.index("term")
                active_idx = header.index("active")
                lang_idx = header.index("languageCode") if "languageCode" in header else None
                for row in reader:
                    if row[active_idx] != "1":
                        continue
                    if lang_idx is not None and row[lang_idx] != "en":
                        continue
                    concept_id = row[concept_idx]
                    term = row[term_idx].strip()
                    if not term or concept_id not in active_concepts:
                        continue
                    synonyms[concept_id].add(term)
                    if row[type_idx] in {"900000000000003001", "900000000000013009"}:  # Fully specified name or preferred term
                        preferred_terms.setdefault(concept_id, term)
                    elif concept_id not in preferred_terms:
                        preferred_terms[concept_id] = term
                    descriptions.setdefault(concept_id, term)

        nodes: List[RepoNodeRecord] = []
        for concept_id, module in active_concepts.items():
            terms = synonyms.get(concept_id, set())
            if not terms:
                continue
            preferred_term = preferred_terms.get(concept_id, sorted(terms)[0])
            metadata = {
                "synonyms": sorted(t for t in terms if t != preferred_term),
                "module": module,
            }
            checksum = hashlib.sha256(
                f"snomed|{concept_id}|{preferred_term}|{json.dumps(metadata, sort_keys=True)}".encode()
            ).hexdigest()
            nodes.append(
                RepoNodeRecord(
                    vocabulary="snomed",
                    code=concept_id,
                    display_name=preferred_term,
                    canonical_uri=f"http://snomed.info/id/{concept_id}",
                    description=descriptions.get(concept_id),
                    metadata=metadata,
                    source_version=self.config.version or "unknown",
                    checksum=checksum,
                )
            )

        edges: List[RepoEdgeRecord] = []
        if relationship_file:
            with _open_text_file(relationship_file) as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader, None)
                if header:
                    source_idx = header.index("sourceId")
                    destination_idx = header.index("destinationId")
                    type_idx = header.index("typeId")
                    active_idx = header.index("active")
                    for row in reader:
                        if row[active_idx] != "1":
                            continue
                        source_id = row[source_idx]
                        target_id = row[destination_idx]
                        type_id = row[type_idx]
                        if source_id not in active_concepts or target_id not in active_concepts:
                            continue
                        predicate = self._relationship_predicate(type_id)
                        metadata = {"typeId": type_id}
                        edges.append(
                            RepoEdgeRecord(
                                vocabulary="snomed",
                                predicate=predicate,
                                source_code=source_id,
                                target_code=target_id,
                                metadata=metadata,
                            )
                        )

        metadata = ReleaseMetadata.create(
            vocabulary="snomed",
            version=self.config.version or "unknown",
            release_checksum=_release_checksum(file_checksums),
            file_checksums=file_checksums,
            total_concepts=len(nodes),
            total_relationships=len(edges),
        )
        return VocabularyLoaderResult(nodes=nodes, edges=edges, metadata=metadata)

    @staticmethod
    def _prefer_file(root: Path, stem: str, *, suffix: str, optional: bool = False) -> Optional[Path]:
        candidates = sorted(root.glob(f"**/{stem}*{suffix}"))
        if candidates:
            return candidates[0]
        if optional:
            return None
        raise FileNotFoundError(f"SNOMED release missing required artefact matching {stem}*{suffix} in {root}")

    @staticmethod
    def _relationship_predicate(type_id: str) -> str:
        mapping = {
            "116680003": "is_a",
            "363705008": "has_finding_site",
            "246075003": "causative_agent",
            "363699004": "has_focus",
        }
        return mapping.get(type_id, "related_to")


VOCABULARY_LOADERS = {
    "umls": UMLSLoader,
    "rxnorm": RxNormLoader,
    "snomed": SnomedLoader,
}
