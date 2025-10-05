"""Utilities for writing embedding artefacts to disk and pgvector."""

from __future__ import annotations

import json
import struct
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg  # type: ignore[import-not-found]
from pgvector.psycopg import register_vector  # type: ignore[import-not-found]
from psycopg import sql  # type: ignore[import-not-found]
from psycopg.types.json import Json  # type: ignore[import-not-found]

from ..config import VectorDatabaseSettings
from ..storage import EmbeddingLayout


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingRecord:
    """Represents a chunk embedding with associated metadata."""

    chunk_id: str
    embedding: List[float]
    metadata: dict

    def as_json(self) -> str:
        return json.dumps(
            {
                "chunk_id": self.chunk_id,
                "embedding": self.embedding,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
        )


class EmbeddingWriter:
    """Persist embedding vectors into the embedding storage layout."""

    def __init__(
        self,
        layout: EmbeddingLayout,
        *,
        quantization_encoding: str = "none",
        store_float32: bool = True,
        vector_db_settings: Optional[VectorDatabaseSettings] = None,
    ) -> None:
        self._layout = layout
        self._quantization_encoding = (quantization_encoding or "none").strip().lower()
        if self._quantization_encoding not in {"none", "bfloat16", "int8"}:
            raise ValueError(
                "EmbeddingWriter quantization_encoding must be one of {'none', 'bfloat16', 'int8'}"
            )
        self._store_float32 = bool(store_float32)
        self._pgvector_sink: Optional[_PgvectorSink] = None
        if vector_db_settings and vector_db_settings.enabled:
            if not vector_db_settings.dsn:
                raise ValueError("vector_db_settings.dsn must be set when vector database is enabled")
            self._pgvector_sink = _PgvectorSink(vector_db_settings)

    def _relative_stub(self, nct_id: str, document_name: str) -> Path:
        stem = Path(document_name).stem or "document"
        return Path(nct_id) / stem

    def _target_path(self, nct_id: str, document_name: str) -> Path:
        stub = self._relative_stub(nct_id, document_name)
        return (self._layout.vectors / stub).with_suffix(".jsonl")

    def exists(self, nct_id: str, document_name: str) -> bool:
        return self._target_path(nct_id, document_name).exists()

    def write(self, nct_id: str, document_name: str, records: Iterable[EmbeddingRecord]) -> Path:
        path = self._target_path(nct_id, document_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payloads_for_db: List[Tuple[EmbeddingRecord, Dict[str, object]]] = []
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                payload = self._serialise_record(record)
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")
                if self._pgvector_sink is not None:
                    payloads_for_db.append((record, payload))
        if self._pgvector_sink is not None:
            try:
                self._pgvector_sink.write(nct_id, document_name, payloads_for_db)
            except Exception:
                _LOGGER.exception(
                    "embeddings | failed to persist embeddings to pgvector | nct_id=%s | document=%s",
                    nct_id,
                    document_name,
                )
                raise
        return path

    def _serialise_record(self, record: EmbeddingRecord) -> Dict[str, object]:
        metadata = dict(record.metadata)
        encoding = self._quantization_encoding
        payload: Dict[str, object] = {
            "chunk_id": record.chunk_id,
            "metadata": metadata,
        }

        if encoding == "none":
            payload["embedding"] = list(record.embedding)
            metadata.setdefault("quantization_encoding", "none")
            return payload

        if encoding == "bfloat16":
            quantized = _quantize_bfloat16(record.embedding)
            payload["embedding_quantized"] = {
                "encoding": "bfloat16",
                "values": quantized,
            }
            metadata["quantization_encoding"] = "bfloat16"
            metadata["quantization_storage_dtype"] = "uint16"
            metadata.setdefault("quantization_precision_bits", 16)
            if self._store_float32:
                payload["embedding"] = list(record.embedding)
            return payload

        if encoding == "int8":
            quantized, scale = _quantize_int8(record.embedding)
            payload["embedding_quantized"] = {
                "encoding": "int8",
                "values": quantized,
                "scale": scale,
                "zero_point": 0,
            }
            metadata["quantization_encoding"] = "int8"
            metadata["quantization_storage_dtype"] = "int8"
            metadata["quantization_scale"] = scale
            metadata.setdefault("quantization_precision_bits", 8)
            if self._store_float32:
                payload["embedding"] = list(record.embedding)
            return payload

        raise ValueError(f"Unsupported quantization encoding: {encoding}")


def _quantize_bfloat16(values: Sequence[float]) -> List[int]:
    quantized: List[int] = []
    for value in values:
        float_value = float(value)
        # bfloat16 retains the sign bit and 8 exponent bits from IEEE 754 float32.
        int_bits = struct.unpack(">I", struct.pack(">f", float_value))[0]
        quantized.append(int_bits >> 16)
    return quantized


def _quantize_int8(values: Sequence[float]) -> tuple[List[int], float]:
    if not values:
        return [], 0.0

    max_abs = max(abs(float(v)) for v in values)
    scale = max_abs / 127.0 if max_abs else 1.0
    if scale == 0.0:
        scale = 1.0

    quantized: List[int] = []
    for value in values:
        scaled = int(round(float(value) / scale))
        quantized.append(int(max(-127, min(127, scaled))))

    return quantized, scale


__all__ = [
    "EmbeddingRecord",
    "EmbeddingWriter",
]


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _PgvectorSink:
    """Helper that upserts embeddings into PostgreSQL using pgvector."""

    def __init__(self, settings: VectorDatabaseSettings) -> None:
        self._settings = settings
        self._table_identifier = sql.SQL(".").join(
            [sql.Identifier(settings.schema), sql.Identifier(settings.embeddings_table)]
        )
        self._delete_sql = sql.SQL(
            "DELETE FROM {table} WHERE nct_id = %s AND document_name = %s"
        ).format(table=self._table_identifier)
        self._insert_sql = sql.SQL(
            """
            INSERT INTO {table} (
                nct_id,
                document_name,
                chunk_id,
                segment_index,
                segment_count,
                section,
                token_count,
                char_count,
                start_word_index,
                study_phase,
                therapeutic_area,
                document_type,
                population,
                endpoint_type,
                page_reference,
                artefact_type,
                source_path,
                parent_chunk_id,
                metadata,
                embedding,
                embedding_model,
                quantization_encoding,
                quantization_storage_dtype,
                quantization_scale
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
             )
            ON CONFLICT (nct_id, document_name, chunk_id)
            DO UPDATE SET
                segment_index = EXCLUDED.segment_index,
                segment_count = EXCLUDED.segment_count,
                section = EXCLUDED.section,
                token_count = EXCLUDED.token_count,
                char_count = EXCLUDED.char_count,
                start_word_index = EXCLUDED.start_word_index,
                study_phase = EXCLUDED.study_phase,
                therapeutic_area = EXCLUDED.therapeutic_area,
                document_type = EXCLUDED.document_type,
                population = EXCLUDED.population,
                endpoint_type = EXCLUDED.endpoint_type,
                page_reference = EXCLUDED.page_reference,
                artefact_type = EXCLUDED.artefact_type,
                source_path = EXCLUDED.source_path,
                parent_chunk_id = EXCLUDED.parent_chunk_id,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                quantization_encoding = EXCLUDED.quantization_encoding,
                quantization_storage_dtype = EXCLUDED.quantization_storage_dtype,
                quantization_scale = EXCLUDED.quantization_scale,
                updated_at = NOW()
            """
        ).format(table=self._table_identifier)

    def write(
        self,
        nct_id: str,
        document_name: str,
        payloads: Sequence[Tuple[EmbeddingRecord, Dict[str, object]]],
    ) -> None:
        if not self._settings.dsn:
            raise ValueError("Vector database DSN is required to persist embeddings")

        dsn = str(self._settings.dsn)
        with psycopg.connect(dsn) as conn:
            register_vector(conn)
            timeout_seconds = float(self._settings.statement_timeout_seconds)
            if timeout_seconds > 0:
                timeout_ms = max(1, int(timeout_seconds * 1000))
                conn.execute(f"SET statement_timeout TO {timeout_ms}")

            with conn.transaction():
                with conn.cursor() as cursor:
                    cursor.execute(self._delete_sql, (nct_id, document_name))
                    if not payloads:
                        return

                    rows: List[Tuple[object, ...]] = []
                    for record, payload in payloads:
                        embedding = [float(value) for value in record.embedding]
                        if len(embedding) != int(self._settings.embedding_dimensions):
                            _LOGGER.warning(
                                "db | skipping embedding due to dimension mismatch | expected=%s | actual=%s | chunk=%s",
                                self._settings.embedding_dimensions,
                                len(embedding),
                                payload.get("chunk_id"),
                            )
                            continue

                        metadata = dict(payload.get("metadata", {}))
                        segment_index = _safe_int(metadata.get("segment_index")) or 0
                        segment_count = _safe_int(metadata.get("segment_count")) or 1
                        token_count = _safe_int(metadata.get("token_count"))
                        char_count = _safe_int(metadata.get("char_count"))
                        start_word_index = _safe_int(metadata.get("start_word_index"))

                        quant_encoding = metadata.get("quantization_encoding", "none")
                        quant_storage = metadata.get("quantization_storage_dtype")
                        quant_scale = metadata.get("quantization_scale")
                        quant_scale_value = None if quant_scale is None else float(quant_scale)

                        embedding_model = metadata.get("model") or metadata.get("embedding_model") or "unknown"
                        artefact_type = metadata.get("artefact_type")
                        source_path = metadata.get("source_path")
                        parent_chunk_id = metadata.get("parent_chunk_id")
                        page_reference = _safe_int(metadata.get("page_reference"))

                        rows.append(
                            (
                                nct_id,
                                document_name,
                                payload.get("chunk_id"),
                                segment_index,
                                segment_count,
                                metadata.get("section"),
                                token_count,
                                char_count,
                                start_word_index,
                                metadata.get("study_phase"),
                                metadata.get("therapeutic_area"),
                                metadata.get("document_type"),
                                metadata.get("population"),
                                metadata.get("endpoint_type"),
                                page_reference,
                                artefact_type,
                                source_path,
                                parent_chunk_id,
                                Json(metadata),
                                embedding,
                                embedding_model,
                                quant_encoding,
                                quant_storage,
                                quant_scale_value,
                            )
                        )

                    if not rows:
                        return

                    cursor.executemany(self._insert_sql, rows)
