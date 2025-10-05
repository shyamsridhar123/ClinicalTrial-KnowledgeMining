"""Clinical entity normalization utilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import importlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Optional dependency bootstrapping -------------------------------------------------

SPACY_AVAILABLE = False
spacy: Optional[Any] = None
SpanType: Optional[Any] = None

try:
    spacy = importlib.import_module("spacy")
    SpanType = getattr(importlib.import_module("spacy.tokens"), "Span")
    SPACY_AVAILABLE = True
except Exception:  # pragma: no cover - spaCy optional at runtime
    spacy = None
    SpanType = None
    SPACY_AVAILABLE = False

SCISPACY_AVAILABLE = False
EntityLinkerCls: Optional[Any] = None
_SCISPACY_IMPORT_ERROR: Optional[Exception] = None

try:
    scispacy_linking = importlib.import_module("scispacy.linking")
    EntityLinkerCls = getattr(scispacy_linking, "EntityLinker")
    SCISPACY_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    _SCISPACY_IMPORT_ERROR = exc
    EntityLinkerCls = None

FUZZY_AVAILABLE = False
fuzz: Optional[Any] = None
process: Optional[Any] = None

try:
    fuzzy_module = importlib.import_module("fuzzywuzzy")
    fuzz = getattr(fuzzy_module, "fuzz")
    process = getattr(fuzzy_module, "process")
    FUZZY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    fuzz = None
    process = None
    FUZZY_AVAILABLE = False


class ClinicalVocabulary(Enum):
    """Supported clinical vocabularies."""

    UMLS = "umls"
    SNOMED = "snomed"
    RXNORM = "rxnorm"
    ICD10 = "icd10"
    LOINC = "loinc"


@dataclass
class NormalizedEntity:
    """Normalized clinical entity enriched with vocabulary metadata."""

    original_text: str
    normalized_text: str
    vocabulary: ClinicalVocabulary
    concept_id: str
    concept_name: str
    semantic_type: Optional[str] = None
    confidence_score: float = 0.0
    alternative_ids: List[str] = None
    definition: Optional[str] = None
    synonyms: List[str] = None

    def __post_init__(self) -> None:  # noqa: D401 - dataclass hook
        if self.alternative_ids is None:
            self.alternative_ids = []
        if self.synonyms is None:
            self.synonyms = []


@dataclass
class EntityNormalizationResult:
    """Normalization response for a single extracted entity."""

    original_entity: str
    entity_type: str
    normalizations: List[NormalizedEntity]
    best_match: Optional[NormalizedEntity] = None
    processing_metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:  # noqa: D401 - dataclass hook
        if self.processing_metadata is None:
            self.processing_metadata = {}
        if self.normalizations and not self.best_match:
            self.best_match = max(self.normalizations, key=lambda item: item.confidence_score)


class ClinicalVocabularyCache:
    """SQLite-backed cache for vocabulary lookups."""

    def __init__(self, cache_dir: str = "./data/vocabulary_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "vocabulary_cache.db"
        self._init_cache_db()

    def _init_cache_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vocabulary_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE NOT NULL,
                    original_text TEXT NOT NULL,
                    vocabulary TEXT NOT NULL,
                    concept_id TEXT,
                    concept_name TEXT,
                    semantic_type TEXT,
                    confidence_score REAL,
                    definition TEXT,
                    synonyms TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vocabulary_cache_query
                ON vocabulary_cache(query_hash)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vocabulary_cache_text
                ON vocabulary_cache(original_text, vocabulary)
                """
            )

    def _get_query_hash(self, text: str, vocabulary: ClinicalVocabulary, entity_type: str = "") -> str:
        query_string = f"{text.lower()}:{vocabulary.value}:{entity_type.lower()}"
        return hashlib.md5(query_string.encode()).hexdigest()

    def get_cached_result(
        self,
        text: str,
        vocabulary: ClinicalVocabulary,
        entity_type: str = "",
    ) -> Optional[NormalizedEntity]:
        query_hash = self._get_query_hash(text, vocabulary, entity_type)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM vocabulary_cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
            if not row:
                return None
            synonyms = json.loads(row["synonyms"]) if row["synonyms"] else []
            return NormalizedEntity(
                original_text=row["original_text"],
                normalized_text=row["concept_name"] or text,
                vocabulary=ClinicalVocabulary(row["vocabulary"]),
                concept_id=row["concept_id"] or "",
                concept_name=row["concept_name"] or "",
                semantic_type=row["semantic_type"],
                confidence_score=row["confidence_score"] or 0.0,
                definition=row["definition"],
                synonyms=synonyms,
            )

    def cache_result(
        self,
        text: str,
        vocabulary: ClinicalVocabulary,
        entity_type: str,
        result: Optional[NormalizedEntity],
    ) -> None:
        query_hash = self._get_query_hash(text, vocabulary, entity_type)
        with sqlite3.connect(self.db_path) as conn:
            if result:
                synonyms = json.dumps(result.synonyms[:20]) if result.synonyms else None
                conn.execute(
                    """
                    INSERT OR REPLACE INTO vocabulary_cache (
                        query_hash,
                        original_text,
                        vocabulary,
                        concept_id,
                        concept_name,
                        semantic_type,
                        confidence_score,
                        definition,
                        synonyms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_hash,
                        text,
                        vocabulary.value,
                        result.concept_id,
                        result.concept_name,
                        result.semantic_type,
                        result.confidence_score,
                        result.definition,
                        synonyms,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO vocabulary_cache (
                        query_hash,
                        original_text,
                        vocabulary,
                        concept_id
                    ) VALUES (?, ?, ?, NULL)
                    """,
                    (query_hash, text, vocabulary.value),
                )


# Minimal fallback vocabulary so tests can run without large knowledge bases
FALLBACK_VOCABULARIES: Dict[ClinicalVocabulary, Dict[str, Dict[str, Any]]] = {
    ClinicalVocabulary.RXNORM: {
        "metformin": {
            "concept_id": "RXNORM:6809",
            "concept_name": "Metformin",
            "semantic_type": "Pharmacologic Substance",
            "definition": "An oral antihyperglycemic agent used for type 2 diabetes.",
            "synonyms": ["Glucophage", "metformin hydrochloride"],
        },
        "aspirin": {
            "concept_id": "RXNORM:1191",
            "concept_name": "Aspirin",
            "semantic_type": "Pharmacologic Substance",
            "definition": "An NSAID used for pain relief and cardiovascular prophylaxis.",
            "synonyms": ["acetylsalicylic acid", "ASA"],
        },
    },
    ClinicalVocabulary.SNOMED: {
        "myocardial infarction": {
            "concept_id": "SNOMEDCT:22298006",
            "concept_name": "Myocardial infarction",
            "semantic_type": "Disease",
            "definition": "Necrosis of myocardial tissue due to ischemia.",
            "synonyms": ["heart attack"],
        },
        "hypertension": {
            "concept_id": "SNOMEDCT:38341003",
            "concept_name": "Hypertensive disorder",
            "semantic_type": "Disease",
            "definition": "Persistently elevated arterial blood pressure.",
            "synonyms": ["high blood pressure"],
        },
    },
    ClinicalVocabulary.ICD10: {
        "type 2 diabetes": {
            "concept_id": "ICD10:E11.9",
            "concept_name": "Type 2 diabetes mellitus without complications",
            "semantic_type": "Disease",
            "definition": "Type 2 diabetes without specified complications.",
            "synonyms": ["adult onset diabetes"],
        },
        "acute myocardial infarction": {
            "concept_id": "ICD10:I21.9",
            "concept_name": "Acute myocardial infarction, unspecified",
            "semantic_type": "Disease",
            "definition": "Acute myocardial infarction without specified site.",
            "synonyms": ["ami"],
        },
    },
    ClinicalVocabulary.LOINC: {
        "hba1c": {
            "concept_id": "LOINC:4548-4",
            "concept_name": "Hemoglobin A1c/Hemoglobin.total in Blood",
            "semantic_type": "Laboratory Test",
            "definition": "Glycated hemoglobin percentage measurement.",
            "synonyms": ["a1c", "glycated hemoglobin"],
        },
        "systolic blood pressure": {
            "concept_id": "LOINC:8480-6",
            "concept_name": "Systolic blood pressure",
            "semantic_type": "Laboratory Test",
            "definition": "Measurement of systolic arterial pressure.",
            "synonyms": ["sbp"],
        },
    },
    ClinicalVocabulary.UMLS: {
        "type 2 diabetes": {
            "concept_id": "UMLS:C0011860",
            "concept_name": "Diabetes Mellitus, Non-Insulin-Dependent",
            "semantic_type": "Disease",
            "definition": "A form of diabetes characterized by insulin resistance.",
            "synonyms": ["type II diabetes", "niddm"],
        },
        "nausea": {
            "concept_id": "UMLS:C0027497",
            "concept_name": "Nausea",
            "semantic_type": "Sign or Symptom",
            "definition": "An unpleasant sensation often preceding vomiting.",
            "synonyms": ["queasiness"],
        },
    },
}


class ClinicalEntityNormalizer:
    """Normalize extracted entities to clinical vocabularies."""

    def __init__(
        self,
        cache_dir: str = "./data/vocabulary_cache",
        *,
        enable_scispacy: bool = True,
        max_candidates: int = 5,
        db_dsn: Optional[str] = None,
    ) -> None:
        self.cache = ClinicalVocabularyCache(cache_dir)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.max_candidates = max(1, max_candidates)
        self.builtin_vocabularies: Dict[ClinicalVocabulary, Dict[str, Dict[str, Any]]] = {}
        self._load_builtin_vocabularies()
        
        # Database connection for repo_nodes query
        self.db_dsn = db_dsn
        self._db_conn = None

        self.scispacy_enabled = bool(enable_scispacy and SPACY_AVAILABLE and SCISPACY_AVAILABLE)
        self._spacy_model: Optional[Any] = None
        self.umls_linker = None
        self.rxnorm_linker = None

        if self.scispacy_enabled:
            self.scispacy_enabled = self._initialize_scispacy_linkers()
            if not self.scispacy_enabled and _SCISPACY_IMPORT_ERROR:
                logger.warning("scispaCy disabled: %s", _SCISPACY_IMPORT_ERROR)

        if not FUZZY_AVAILABLE:
            logger.debug("Fuzzy matching unavailable; fallback normalization will rely on exact matches.")

    def _initialize_scispacy_linkers(self) -> bool:
        if not SPACY_AVAILABLE or not SCISPACY_AVAILABLE:
            return False

        model_candidates = ["en_core_sci_sm", "en_core_web_sm"]
        for model_name in model_candidates:
            try:
                self._spacy_model = spacy.load(model_name)  # type: ignore[arg-type]
                break
            except Exception as exc:  # pragma: no cover - depends on local models
                logger.debug("spaCy model '%s' unavailable: %s", model_name, exc)
                self._spacy_model = None
        if self._spacy_model is None or EntityLinkerCls is None:
            logger.warning("spaCy biomedical model not available; falling back to built-in vocabularies.")
            return False

        try:
            umls_linker = EntityLinkerCls(
                resolve_abbreviations=True,
                threshold=0.75,
                max_entities_per_mention=self.max_candidates,
                linker_name="umls",
            )
        except Exception as exc:  # pragma: no cover - heavy dependency
            logger.warning("Unable to initialize UMLS linker: %s", exc)
            return False

        try:
            rxnorm_linker = EntityLinkerCls(
                resolve_abbreviations=False,
                filter_for_definitions=False,
                threshold=0.75,
                max_entities_per_mention=self.max_candidates,
                linker_name="rxnorm",
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("RxNorm linker unavailable: %s", exc)
            rxnorm_linker = None

        self.umls_linker = umls_linker
        self.rxnorm_linker = rxnorm_linker
        return True

    async def normalize_entity(self, entity_text: str, entity_type: str = "") -> EntityNormalizationResult:
        logger.info("Normalizing entity '%s' (type=%s)", entity_text, entity_type)

        normalizations: List[NormalizedEntity] = []
        vocabularies = self._get_relevant_vocabularies(entity_type)
        cache_hits = 0

        loop = asyncio.get_running_loop()

        if self.scispacy_enabled and entity_text.strip():
            scispacy_results = await loop.run_in_executor(
                self.executor,
                self._normalize_with_scispacy,
                entity_text,
                entity_type,
            )
            for result in scispacy_results:
                self.cache.cache_result(entity_text, result.vocabulary, entity_type, result)
            normalizations.extend(scispacy_results)

        # Batch query database for all vocabularies at once (much faster)
        uncached_vocabs = []
        for vocabulary in vocabularies:
            cached = self.cache.get_cached_result(entity_text, vocabulary, entity_type)
            if cached:
                cache_hits += 1
                normalizations.append(cached)
            else:
                uncached_vocabs.append(vocabulary)
        
        # Query database for all uncached vocabularies in one go
        if uncached_vocabs and self.db_dsn:
            db_results = await self._normalize_with_database_batch(entity_text, uncached_vocabs)
            for vocab, db_result in db_results.items():
                self.cache.cache_result(entity_text, vocab, entity_type, db_result)
                if db_result:
                    normalizations.append(db_result)
        
        # Fallback to builtin for any still missing
        for vocabulary in uncached_vocabs:
            if not any(n.vocabulary == vocabulary for n in normalizations):
                builtin_result = await loop.run_in_executor(
                    self.executor,
                    self._normalize_with_builtin,
                    entity_text,
                    vocabulary,
                )
                self.cache.cache_result(entity_text, vocabulary, entity_type, builtin_result)
                if builtin_result:
                    normalizations.append(builtin_result)

        deduped: Dict[Tuple[str, str], NormalizedEntity] = {}
        for norm in normalizations:
            key = (norm.vocabulary.value, norm.concept_id)
            existing = deduped.get(key)
            if existing is None or norm.confidence_score > existing.confidence_score:
                deduped[key] = norm

        ordered = sorted(deduped.values(), key=lambda item: item.confidence_score, reverse=True)

        result = EntityNormalizationResult(
            original_entity=entity_text,
            entity_type=entity_type,
            normalizations=ordered,
            processing_metadata={
                "vocabularies_searched": [v.value for v in vocabularies],
                "scispacy_used": self.scispacy_enabled,
                "cache_hits": cache_hits,
                "search_method": "scispacy+fallback" if self.scispacy_enabled else ("fuzzy_match" if FUZZY_AVAILABLE else "exact_match"),
                "sources_returned": sorted({norm.vocabulary.value for norm in ordered}),
            },
        )

        if result.best_match:
            logger.debug(
                "Best normalization for '%s': %s (%s) via %s",
                entity_text,
                result.best_match.concept_name,
                result.best_match.concept_id,
                result.best_match.vocabulary.value,
            )
        else:
            logger.debug("No normalization candidates for '%s'", entity_text)

        return result

    async def normalize_entities_batch(self, entities: List[Tuple[str, str]]) -> List[EntityNormalizationResult]:
        tasks = [self.normalize_entity(text, entity_type) for text, entity_type in entities]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        filtered: List[EntityNormalizationResult] = []
        for entity, result in zip(entities, results):
            if isinstance(result, Exception):
                logger.error("Normalization failed for '%s': %s", entity[0], result)
                continue
            filtered.append(result)
        return filtered

    def _normalize_with_scispacy(self, entity_text: str, entity_type: str) -> List[NormalizedEntity]:
        if not self.scispacy_enabled or not entity_text.strip() or self._spacy_model is None:
            return []

        label = (entity_type or "entity").upper()
        normalizations: List[NormalizedEntity] = []

        umls_doc = self._create_doc_for_linking(entity_text, label)
        if umls_doc is not None and self.umls_linker is not None:
            umls_doc = self.umls_linker(umls_doc)
            normalizations.extend(
                self._extract_from_linker(umls_doc, self.umls_linker, ClinicalVocabulary.UMLS)
            )

        if self.rxnorm_linker is not None and self._should_use_rxnorm(entity_type):
            rx_doc = self._create_doc_for_linking(entity_text, label)
            if rx_doc is not None:
                rx_doc = self.rxnorm_linker(rx_doc)
                normalizations.extend(
                    self._extract_from_linker(rx_doc, self.rxnorm_linker, ClinicalVocabulary.RXNORM)
                )

        return normalizations

    def _extract_from_linker(
        self,
        doc,
        linker: Any,
        vocabulary: ClinicalVocabulary,
    ) -> List[NormalizedEntity]:
        if not doc.ents:
            return []

        span = doc.ents[0]
        results: List[NormalizedEntity] = []
        for concept_id, score in span._.kb_ents[: self.max_candidates]:
            entity = linker.kb.cui_to_entity.get(concept_id)
            if not entity:
                continue
            synonyms = list(entity.aliases[:10]) if getattr(entity, "aliases", None) else []
            semantic_type = ";".join(entity.types) if getattr(entity, "types", None) else None
            results.append(
                NormalizedEntity(
                    original_text=span.text,
                    normalized_text=entity.canonical_name,
                    vocabulary=vocabulary,
                    concept_id=concept_id,
                    concept_name=entity.canonical_name,
                    semantic_type=semantic_type,
                    confidence_score=float(score),
                    definition=entity.definition,
                    synonyms=synonyms,
                )
            )
        return results

    def _create_doc_for_linking(self, text: str, label: str):
        if self._spacy_model is None or SpanType is None:
            return None

        doc = self._spacy_model.make_doc(text)
        if len(doc) == 0:
            return None
        span = SpanType(doc, 0, len(doc), label=label)
        doc.ents = [span]
        return doc

    async def _normalize_with_database_batch(
        self,
        entity_text: str,
        vocabularies: List[ClinicalVocabulary],
    ) -> Dict[ClinicalVocabulary, Optional[NormalizedEntity]]:
        """Query repo_nodes for multiple vocabularies using optimized indexes."""
        results: Dict[ClinicalVocabulary, Optional[NormalizedEntity]] = {}
        
        if not self.db_dsn or not vocabularies:
            return results
        
        try:
            import psycopg  # type: ignore[import-not-found]
            from psycopg.rows import dict_row  # type: ignore[import-not-found]
            
            # Create connection if needed
            if self._db_conn is None:
                self._db_conn = await psycopg.AsyncConnection.connect(
                    self.db_dsn,
                    row_factory=dict_row,
                )
            
            key = entity_text.lower().strip()
            if not key or len(key) < 2:
                return {v: None for v in vocabularies}
            
            # Map vocabulary enums to database names
            vocab_map = {
                ClinicalVocabulary.UMLS: 'umls',
                ClinicalVocabulary.RXNORM: 'rxnorm',
                ClinicalVocabulary.SNOMED: 'snomed',
                ClinicalVocabulary.ICD10: 'icd10',
                ClinicalVocabulary.LOINC: 'loinc',
            }
            vocab_names = [vocab_map[v] for v in vocabularies if v in vocab_map]
            if not vocab_names:
                return results
            
            async with self._db_conn.cursor() as cur:
                # OPTIMIZED: Use indexed columns for exact match (hits idx_repo_nodes_display_name_lower)
                placeholders = ','.join(['%s'] * len(vocab_names))
                await cur.execute(
                    f"""
                    SELECT vocabulary, code, display_name, description, metadata,
                           1.0 as similarity_score
                    FROM docintel.repo_nodes
                    WHERE vocabulary IN ({placeholders})
                      AND (LOWER(display_name) = %s OR LOWER(code) = %s)
                      AND is_active = true
                    LIMIT {len(vocab_names)}
                    """,
                    (*vocab_names, key, key)
                )
                rows = await cur.fetchall()
                
                # Process exact matches
                for row in rows:
                    vocab_enum = next((v for v, n in vocab_map.items() if n == row['vocabulary']), None)
                    if vocab_enum and vocab_enum not in results:
                        metadata = row.get('metadata') or {}
                        results[vocab_enum] = NormalizedEntity(
                            original_text=entity_text,
                            normalized_text=row['display_name'] or row['code'],
                            vocabulary=vocab_enum,
                            concept_id=f"{row['vocabulary'].upper()}:{row['code']}",
                            concept_name=row['display_name'] or row['code'],
                            semantic_type=metadata.get('semantic_type'),
                            confidence_score=1.0,
                            definition=row.get('description'),
                            synonyms=metadata.get('synonyms', [])[:10],
                        )
                
                # OPTIMIZED: Use indexed ILIKE for fuzzy matching (much faster than full table scan)
                unmatched_vocabs = [v for v in vocabularies if v not in results]
                if unmatched_vocabs and len(key) >= 3:
                    unmatched_names = [vocab_map[v] for v in unmatched_vocabs if v in vocab_map]
                    if unmatched_names:
                        placeholders = ','.join(['%s'] * len(unmatched_names))
                        # Use indexed ILIKE with wildcards (hits idx_repo_nodes_vocab_display)
                        await cur.execute(
                            f"""
                            SELECT vocabulary, code, display_name, description, metadata
                            FROM docintel.repo_nodes
                            WHERE vocabulary IN ({placeholders})
                              AND is_active = true
                              AND (
                                  display_name ILIKE %s
                                  OR display_name ILIKE %s
                                  OR code ILIKE %s
                              )
                            LIMIT 20
                            """,
                            (*unmatched_names, f'%{key}%', f'{key}%', f'%{key}%')
                        )
                        candidates = await cur.fetchall()
                        
                        # Use Python fuzzy matching on limited candidates
                        candidates_by_vocab: Dict[str, tuple] = {}
                        for candidate in candidates:
                            vocab_name = candidate['vocabulary']
                            display_name = (candidate.get('display_name') or '').lower()
                            code = (candidate.get('code') or '').lower()
                            
                            # Calculate fuzzy match score
                            if FUZZY_AVAILABLE:
                                name_score = fuzz.token_sort_ratio(key, display_name) if display_name else 0  # type: ignore[misc]
                                code_score = fuzz.ratio(key, code) if code else 0  # type: ignore[misc]
                                sim_score = max(name_score, code_score) / 100.0
                            else:
                                # Fallback to simple substring match
                                sim_score = 0.5 if key in display_name or key in code else 0.0
                            
                            # Only accept if similarity >= 0.7 (70% match threshold)
                            if sim_score < 0.7:
                                continue
                            
                            # Keep best match per vocabulary
                            if vocab_name not in candidates_by_vocab or sim_score > candidates_by_vocab[vocab_name][1]:
                                candidates_by_vocab[vocab_name] = (candidate, sim_score)
                        
                        # Create NormalizedEntity for each matched vocabulary
                        for vocab_name, (candidate, sim_score) in candidates_by_vocab.items():
                            vocab_enum = next((v for v, n in vocab_map.items() if n == vocab_name), None)
                            if vocab_enum and vocab_enum not in results:
                                metadata = candidate.get('metadata') or {}
                                results[vocab_enum] = NormalizedEntity(
                                    original_text=entity_text,
                                    normalized_text=candidate['display_name'] or candidate['code'],
                                    vocabulary=vocab_enum,
                                    concept_id=f"{vocab_name.upper()}:{candidate['code']}",
                                    concept_name=candidate['display_name'] or candidate['code'],
                                    semantic_type=metadata.get('semantic_type'),
                                    confidence_score=float(sim_score),
                                    definition=candidate.get('description'),
                                    synonyms=metadata.get('synonyms', [])[:10],
                                )
            
            # Mark unmatched vocabularies as None
            for vocab in vocabularies:
                if vocab not in results:
                    results[vocab] = None
            
            return results
            
        except Exception as exc:
            logger.debug(f"Batch database normalization failed for '{entity_text}': {exc}")
            return {v: None for v in vocabularies}

    async def _normalize_with_database(
        self,
        entity_text: str,
        vocabulary: ClinicalVocabulary,
    ) -> Optional[NormalizedEntity]:
        """Query repo_nodes table for vocabulary matches."""
        if not self.db_dsn:
            return None
        
        try:
            import psycopg  # type: ignore[import-not-found]
            from psycopg.rows import dict_row  # type: ignore[import-not-found]
            
            # Create connection if needed
            if self._db_conn is None:
                self._db_conn = await psycopg.AsyncConnection.connect(
                    self.db_dsn,
                    row_factory=dict_row,
                )
            
            key = entity_text.lower().strip()
            if not key:
                return None
            
            # Map vocabulary enum to database vocabulary names
            vocab_map = {
                ClinicalVocabulary.UMLS: 'umls',
                ClinicalVocabulary.RXNORM: 'rxnorm',
                ClinicalVocabulary.SNOMED: 'snomed',
                ClinicalVocabulary.ICD10: 'icd10',
                ClinicalVocabulary.LOINC: 'loinc',
            }
            vocab_name = vocab_map.get(vocabulary)
            if not vocab_name:
                return None
            
            async with self._db_conn.cursor() as cur:
                # Try exact match first (case-insensitive)
                await cur.execute(
                    """
                    SELECT code, display_name, description, metadata
                    FROM docintel.repo_nodes
                    WHERE vocabulary = %s 
                      AND (LOWER(display_name) = %s OR LOWER(code) = %s)
                      AND is_active = true
                    LIMIT 1
                    """,
                    (vocab_name, key, key)
                )
                row = await cur.fetchone()
                
                if row:
                    metadata = row.get('metadata') or {}
                    return NormalizedEntity(
                        original_text=entity_text,
                        normalized_text=row['display_name'] or row['code'],
                        vocabulary=vocabulary,
                        concept_id=f"{vocab_name.upper()}:{row['code']}",
                        concept_name=row['display_name'] or row['code'],
                        semantic_type=metadata.get('semantic_type'),
                        confidence_score=1.0,
                        definition=row.get('description'),
                        synonyms=metadata.get('synonyms', [])[:10],
                    )
                
                # Try partial match with similarity
                if FUZZY_AVAILABLE:
                    await cur.execute(
                        """
                        SELECT code, display_name, description, metadata
                        FROM docintel.repo_nodes
                        WHERE vocabulary = %s 
                          AND (display_name ILIKE %s OR code ILIKE %s)
                          AND is_active = true
                        LIMIT 20
                        """,
                        (vocab_name, f'%{key}%', f'%{key}%')
                    )
                    candidates = await cur.fetchall()
                    
                    if candidates:
                        # Use fuzzy matching to find best match
                        best_score = 0
                        best_candidate = None
                        
                        for candidate in candidates:
                            display_name = (candidate.get('display_name') or '').lower()
                            code = (candidate.get('code') or '').lower()
                            
                            # Calculate similarity scores
                            name_score = fuzz.token_sort_ratio(key, display_name) if display_name else 0  # type: ignore[misc]
                            code_score = fuzz.ratio(key, code) if code else 0  # type: ignore[misc]
                            score = max(name_score, code_score)
                            
                            if score > best_score and score >= 70:  # Minimum 70% match
                                best_score = score
                                best_candidate = candidate
                        
                        if best_candidate:
                            metadata = best_candidate.get('metadata') or {}
                            return NormalizedEntity(
                                original_text=entity_text,
                                normalized_text=best_candidate['display_name'] or best_candidate['code'],
                                vocabulary=vocabulary,
                                concept_id=f"{vocab_name.upper()}:{best_candidate['code']}",
                                concept_name=best_candidate['display_name'] or best_candidate['code'],
                                semantic_type=metadata.get('semantic_type'),
                                confidence_score=best_score / 100.0,
                                definition=best_candidate.get('description'),
                                synonyms=metadata.get('synonyms', [])[:10],
                            )
            
            return None
            
        except Exception as exc:
            logger.debug(f"Database normalization failed for '{entity_text}': {exc}")
            return None

    def _normalize_with_builtin(
        self,
        entity_text: str,
        vocabulary: ClinicalVocabulary,
    ) -> Optional[NormalizedEntity]:
        vocab = self.builtin_vocabularies.get(vocabulary, {})
        if not vocab:
            return None

        key = entity_text.lower().strip()
        if not key:
            return None

        if key in vocab:
            return self._create_normalized_entity(entity_text, vocab[key], vocabulary, 1.0)

        if FUZZY_AVAILABLE:
            best_match = process.extractOne(key, list(vocab.keys()), scorer=fuzz.token_sort_ratio)  # type: ignore[misc]
            if best_match and best_match[1] >= 80:
                return self._create_normalized_entity(
                    entity_text,
                    vocab[best_match[0]],
                    vocabulary,
                    best_match[1] / 100.0,
                )

        for concept in vocab.values():
            synonyms = concept.get("synonyms", [])
            if any(key == synonym.lower() for synonym in synonyms):
                return self._create_normalized_entity(entity_text, concept, vocabulary, 0.9)

        return None

    def _should_use_rxnorm(self, entity_type: str) -> bool:
        entity_type_lower = (entity_type or "").lower()
        return entity_type_lower in {
            "medication",
            "drug",
            "dosage",
            "treatment",
            "therapy",
            "intervention",
        }

    def _create_normalized_entity(
        self,
        original_text: str,
        concept: Dict[str, Any],
        vocabulary: ClinicalVocabulary,
        confidence: float,
    ) -> NormalizedEntity:
        synonyms = concept.get("synonyms", [])
        semantic_type = concept.get("semantic_type")
        definition = concept.get("definition")
        return NormalizedEntity(
            original_text=original_text,
            normalized_text=concept["concept_name"],
            vocabulary=vocabulary,
            concept_id=concept["concept_id"],
            concept_name=concept["concept_name"],
            semantic_type=semantic_type,
            confidence_score=confidence,
            definition=definition,
            synonyms=list(synonyms),
        )

    def _load_builtin_vocabularies(self) -> None:
        self.builtin_vocabularies = {
            vocab: {term: dict(values) for term, values in data.items()}
            for vocab, data in FALLBACK_VOCABULARIES.items()
        }

    def _get_relevant_vocabularies(self, entity_type: str) -> List[ClinicalVocabulary]:
        entity_type_lower = (entity_type or "").lower()
        mapping: Dict[str, List[ClinicalVocabulary]] = {
            "drug": [ClinicalVocabulary.RXNORM, ClinicalVocabulary.UMLS],
            "medication": [ClinicalVocabulary.RXNORM, ClinicalVocabulary.UMLS],
            "treatment": [ClinicalVocabulary.RXNORM, ClinicalVocabulary.UMLS],
            "therapy": [ClinicalVocabulary.RXNORM, ClinicalVocabulary.UMLS],
            "dosage": [ClinicalVocabulary.RXNORM, ClinicalVocabulary.UMLS],
            "condition": [ClinicalVocabulary.SNOMED, ClinicalVocabulary.ICD10, ClinicalVocabulary.UMLS],
            "disease": [ClinicalVocabulary.SNOMED, ClinicalVocabulary.ICD10, ClinicalVocabulary.UMLS],
            "disorder": [ClinicalVocabulary.SNOMED, ClinicalVocabulary.ICD10, ClinicalVocabulary.UMLS],
            "adverse_event": [ClinicalVocabulary.SNOMED, ClinicalVocabulary.UMLS],
            "symptom": [ClinicalVocabulary.SNOMED, ClinicalVocabulary.UMLS],
            "sign": [ClinicalVocabulary.SNOMED, ClinicalVocabulary.UMLS],
            "procedure": [ClinicalVocabulary.SNOMED, ClinicalVocabulary.UMLS],
            "measurement": [ClinicalVocabulary.LOINC, ClinicalVocabulary.UMLS],
            "lab_test": [ClinicalVocabulary.LOINC, ClinicalVocabulary.UMLS],
            "endpoint": [ClinicalVocabulary.UMLS],
            "population": [ClinicalVocabulary.UMLS],
            "timepoint": [ClinicalVocabulary.UMLS],
        }

        vocabularies = mapping.get(entity_type_lower, [ClinicalVocabulary.UMLS])
        if ClinicalVocabulary.UMLS not in vocabularies:
            vocabularies.append(ClinicalVocabulary.UMLS)
        return vocabularies

    async def get_normalization_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.cache.db_path) as conn:
            total_cached = conn.execute("SELECT COUNT(*) FROM vocabulary_cache").fetchone()[0]
            by_vocab = conn.execute(
                "SELECT vocabulary, COUNT(*) FROM vocabulary_cache GROUP BY vocabulary"
            ).fetchall()
            by_status = conn.execute(
                """
                SELECT
                    CASE WHEN concept_id IS NOT NULL THEN 'successful' ELSE 'failed' END,
                    COUNT(*)
                FROM vocabulary_cache
                GROUP BY 1
                """
            ).fetchall()

        return {
            "cache_stats": {
                "total_cached": total_cached,
                "by_vocabulary": dict(by_vocab),
                "by_status": dict(by_status),
            },
            "vocabularies_supported": [v.value for v in ClinicalVocabulary],
            "scispacy_enabled": self.scispacy_enabled,
            "fuzzy_matching_available": FUZZY_AVAILABLE,
        }


async def normalize_clinical_entities(
    entities: List[Tuple[str, str]],
    cache_dir: str = "./data/vocabulary_cache",
    *,
    enable_scispacy: bool = True,
) -> List[EntityNormalizationResult]:
    normalizer = ClinicalEntityNormalizer(cache_dir, enable_scispacy=enable_scispacy)
    return await normalizer.normalize_entities_batch(entities)
