"""Pipeline phase that generates embeddings for parsed document chunks."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from ..config import EmbeddingSettings, get_embedding_settings, get_vector_db_settings
from ..pipeline import PhaseResult, PipelineContext, PipelinePhase
from ..storage import EmbeddingLayout, ProcessingLayout, build_embedding_layout, ensure_embedding_layout
from .client import EmbeddingClient, EmbeddingClientError, EmbeddingResponse
from .writer import EmbeddingRecord, EmbeddingWriter

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from transformers import PreTrainedTokenizerBase  # type: ignore[import-not-found]

_LOGGER = logging.getLogger(__name__)

_TOKENIZER_FALLBACKS = {
    "microsoft/biomedclip-pubmedbert_256-vit_base_patch16_224": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
}


def _normalise_model_identifier(model_name: str) -> str:
    if not model_name:
        return model_name
    normalised = model_name.strip()
    if normalised.startswith("hf-hub:"):
        normalised = normalised.split(":", 1)[1]
    return normalised


def _default_writer_factory(layout: EmbeddingLayout, settings: EmbeddingSettings) -> EmbeddingWriter:
    vector_settings = get_vector_db_settings()
    return EmbeddingWriter(
        layout,
        quantization_encoding=settings.embedding_quantization_encoding,
        store_float32=settings.embedding_quantization_store_float32,
        vector_db_settings=vector_settings,
    )


@lru_cache(maxsize=8)
def _load_tokenizer_by_name(model_name: str) -> Optional["PreTrainedTokenizerBase"]:
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        _LOGGER.warning(
            "embeddings | transformers not available, falling back to heuristic segmentation | model=%s",
            model_name,
        )
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[operator]
            model_name,
            use_fast=True,
            local_files_only=True,
        )
    except Exception as exc:  # pragma: no cover - import failure path
        _LOGGER.warning(
            "embeddings | tokenizer not cached locally | model=%s | error=%s | attempting remote fetch",
            model_name,
            exc,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[operator]
                model_name,
                use_fast=True,
            )
        except Exception as remote_exc:  # pragma: no cover - import failure path
            _LOGGER.warning(
                "embeddings | unable to load tokenizer | model=%s | error=%s | using heuristic segmentation",
                model_name,
                remote_exc,
            )
            return None

    return tokenizer


def _strip_hf_repo_prefix(model_name: str) -> str:
    cleaned = (model_name or "").strip()
    for prefix in ("hf-hub:", "hf_hub:", "hfhub:", "hub:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned


def _tokenizer_name_candidates(model_name: str) -> List[str]:
    sanitized = _strip_hf_repo_prefix(model_name)
    candidates: List[str] = []
    fallback = _TOKENIZER_FALLBACKS.get(sanitized.lower())
    if fallback:
        candidates.append(fallback)
    if sanitized:
        candidates.append(sanitized)
    return list(dict.fromkeys(filter(None, candidates)))


def _load_embedding_tokenizer(settings: EmbeddingSettings) -> Optional["PreTrainedTokenizerBase"]:
    """Skip tokenizer loading for sentence-transformers models that handle context internally."""
    model_name = getattr(settings, "embedding_model_name", None)
    if not model_name:
        return None

    # Skip tokenizer loading for sentence-transformers models
    # These models handle tokenization and context window internally
    if "sentence-transformers" in model_name:
        _LOGGER.info("embeddings | skipping separate tokenizer for sentence-transformers model | model=%s", model_name)
        return None

    for candidate in _tokenizer_name_candidates(model_name):
        if candidate != model_name:
            _LOGGER.debug(
                "embeddings | attempting alternate tokenizer name | model=%s | candidate=%s",
                model_name,
                candidate,
            )
        tokenizer = _load_tokenizer_by_name(candidate)
        if tokenizer is not None:
            if candidate != model_name:
                _LOGGER.info(
                    "embeddings | using tokenizer '%s' for embedding model '%s'",
                    candidate,
                    model_name,
                )
            return tokenizer

    return None


@dataclass(slots=True)
class _PreparedSegment:
    text: str
    metadata: Dict[str, object]


@dataclass(slots=True)
class _PreparedImage:
    path: Path
    metadata: Dict[str, object]


@dataclass(slots=True)
class EmbeddingPhase(PipelinePhase):
    """Generate embeddings for all chunk files emitted by the parsing stage."""

    force_reembed: bool = False
    batch_size_override: Optional[int] = None
    name: str = "embeddings"
    client_factory: Callable[[EmbeddingSettings], EmbeddingClient] = field(
        default=EmbeddingClient,
        repr=False,
    )
    writer_factory: Callable[[EmbeddingLayout, EmbeddingSettings], EmbeddingWriter] = field(
        default=_default_writer_factory,
        repr=False,
    )
    tokenizer_factory: Callable[[EmbeddingSettings], Optional["PreTrainedTokenizerBase"]] = field(
        default=_load_embedding_tokenizer,
        repr=False,
    )

    async def run(self, context: PipelineContext) -> PhaseResult:
        processing_layout = context.processing_layout
        if processing_layout is None:
            raise ValueError("processing_layout missing from pipeline context")

        embedding_settings = context.embedding_settings or get_embedding_settings()
        context.embedding_settings = embedding_settings

        embedding_layout = context.embedding_layout or build_embedding_layout(embedding_settings)
        ensure_embedding_layout(embedding_layout)
        context.embedding_layout = embedding_layout

        chunk_files = sorted(processing_layout.chunks.glob("**/*.json"))
        if not chunk_files:
            report = {
                "statistics": {
                    "documents_total": 0,
                    "documents_processed": 0,
                    "documents_skipped": 0,
                    "documents_failed": 0,
                    "chunks_embedded": 0,
                },
                "documents": [],
            }
            context.extra.setdefault("embeddings", {})["report"] = report
            return PhaseResult(name=self.name, succeeded=True, details={"report": report})

        client = self.client_factory(embedding_settings)
        writer = self.writer_factory(embedding_layout, embedding_settings)

        stats = {
            "documents_total": len(chunk_files),
            "documents_processed": 0,
            "documents_skipped": 0,
            "documents_failed": 0,
            "chunks_embedded": 0,
        }
        documents_report: List[Dict[str, object]] = []
        batch_size = self.batch_size_override or int(embedding_settings.embedding_batch_size)
        max_tokens = int(embedding_settings.embedding_max_tokens)

        try:
            chunks_root = processing_layout.chunks
            tokenizer = None
            if max_tokens > 0:
                tokenizer = self.tokenizer_factory(embedding_settings)
                if tokenizer is not None:
                    tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
                    if isinstance(tokenizer_max_len, int) and tokenizer_max_len > 0:
                        if max_tokens > tokenizer_max_len:
                            _LOGGER.debug(
                                "embeddings | clamping max_tokens to tokenizer limit | original=%s | tokenizer_limit=%s",
                                max_tokens,
                                tokenizer_max_len,
                            )
                            max_tokens = tokenizer_max_len
            for chunk_file in chunk_files:
                doc_report = await self._embed_document(
                    chunk_file,
                    chunks_root,
                    processing_layout,
                    writer,
                    client,
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                    tokenizer=tokenizer,
                    force=self.force_reembed,
                )
                documents_report.append(doc_report)
                status = doc_report.get("status")
                if status == "processed":
                    stats["documents_processed"] += 1
                    stats["chunks_embedded"] += int(doc_report.get("chunk_count", 0))
                elif status == "skipped":
                    stats["documents_skipped"] += 1
                else:
                    stats["documents_failed"] += 1
        finally:
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()

        succeeded = stats["documents_failed"] == 0
        report = {"statistics": stats, "documents": documents_report}
        context.extra.setdefault("embeddings", {})["report"] = report
        return PhaseResult(name=self.name, succeeded=succeeded, details={"report": report})

    async def _embed_document(
        self,
        chunk_file: Path,
        chunks_root: Path,
        processing_layout: ProcessingLayout,
        writer: EmbeddingWriter,
        client: EmbeddingClient,
        *,
        batch_size: int,
        max_tokens: int,
        tokenizer: Optional["PreTrainedTokenizerBase"],
        force: bool,
    ) -> Dict[str, object]:
        relative = chunk_file.relative_to(chunks_root)
        parts = relative.parts
        if not parts:
            raise ValueError(f"Chunk path {chunk_file} is not under {chunks_root}")
        nct_id = parts[0]
        document_path = Path(*parts[1:]) if len(parts) > 1 else Path(relative.name)
        document_name = document_path.as_posix()
        document_stem = document_path.stem or document_name

        if not force and writer.exists(nct_id, document_name):
            return {
                "status": "skipped",
                "document": str(chunk_file),
                "nct_id": nct_id,
                "document_name": document_name,
                "reason": "already_embedded",
            }

        try:
            payload = json.loads(chunk_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _LOGGER.error("embeddings | invalid chunk payload | path=%s | error=%s", chunk_file, exc)
            return {
                "status": "failed",
                "document": str(chunk_file),
                "nct_id": nct_id,
                "document_name": document_name,
                "error": str(exc),
            }

        if isinstance(payload, dict):
            chunk_list = payload.get("chunks", [])
        elif isinstance(payload, list):
            chunk_list = payload
        else:
            chunk_list = []

        processing_root = processing_layout.root
        chunk_source_path = Path("chunks") / relative
        table_source_path = Path("tables") / relative
        figure_source_path = Path("figures") / relative

        chunk_segments: List[_PreparedSegment] = []
        if chunk_list:
            chunk_segments = _expand_chunk_items(
                chunk_list,
                max_tokens=max_tokens,
                tokenizer=tokenizer,
            )
            for segment in chunk_segments:
                metadata = segment.metadata
                metadata.setdefault("artefact_type", "chunk")
                metadata.setdefault("parent_chunk_id", metadata.get("parent_chunk_id") or metadata.get("chunk_id"))
                metadata.setdefault("source_path", str(chunk_source_path))
                page_value = metadata.get("page_reference")
                if page_value is None:
                    page_value = metadata.pop("page", None)
                if page_value is None:
                    page_value = metadata.pop("page_ref", None)
                metadata["page_reference"] = _coerce_int(page_value)

        table_segments = _load_table_segments(
            processing_layout.tables,
            relative,
            document_stem=document_stem,
            source_path=str(table_source_path),
        )

        figure_segments, figure_images = _load_figure_segments(
            processing_layout.figures,
            relative,
            document_stem=document_stem,
            processing_root=processing_root,
            source_path=str(figure_source_path),
        )

        text_segments: List[_PreparedSegment] = [
            *chunk_segments,
            *table_segments,
            *figure_segments,
        ]

        texts: List[str] = [segment.text for segment in text_segments]
        metadata_entries: List[Dict[str, object]] = [dict(segment.metadata) for segment in text_segments]
        artefact_meta: List[Tuple[str, Optional[str], Optional[int]]] = [
            (
                str(metadata.get("artefact_type", "chunk")),
                metadata.get("source_path"),
                _coerce_int(metadata.get("page_reference")),
            )
            for metadata in metadata_entries
        ]

        if not texts and not figure_images:
            writer.write(nct_id, document_name, [])
            return {
                "status": "processed",
                "document": str(chunk_file),
                "nct_id": nct_id,
                "document_name": document_name,
                "chunk_count": 0,
                "text_embedding_count": 0,
                "image_embedding_count": 0,
            }

        records: List[EmbeddingRecord] = []
        text_record_count = 0
        image_record_count = 0

        try:
            for batch_start in range(0, len(texts), batch_size):
                batch_texts = texts[batch_start : batch_start + batch_size]
                if not batch_texts:
                    continue
                embeddings = await client.embed_texts(batch_texts)
                for idx, embedding in enumerate(embeddings):
                    absolute_index = batch_start + idx
                    if absolute_index >= len(metadata_entries):
                        continue
                    base_metadata = dict(metadata_entries[absolute_index])
                    artefact_type, source_path, page_reference = artefact_meta[absolute_index]
                    base_metadata = _prepare_embedding_metadata(
                        base_metadata,
                        nct_id=nct_id,
                        document_name=document_name,
                        embedding_model=embedding.model,
                        artefact_type=artefact_type,
                        source_path=source_path,
                        page_reference=page_reference,
                    )
                    chunk_id = str(base_metadata.get("chunk_id") or f"chunk-{absolute_index:04d}")
                    records.append(
                        EmbeddingRecord(
                            chunk_id=chunk_id,
                            embedding=embedding.embedding,
                            metadata=base_metadata,
                        )
                    )
                    text_record_count += 1
        except EmbeddingClientError as exc:
            _LOGGER.error("embeddings | embedding request failed | path=%s | error=%s", chunk_file, exc)
            return {
                "status": "failed",
                "document": str(chunk_file),
                "nct_id": nct_id,
                "document_name": document_name,
                "error": str(exc),
            }

        if figure_images:
            image_paths = [prepared.path for prepared in figure_images]
            try:
                image_embeddings = await client.embed_images(image_paths)
            except EmbeddingClientError as exc:
                _LOGGER.error(
                    "embeddings | image embedding request failed | path=%s | error=%s",
                    chunk_file,
                    exc,
                )
                return {
                    "status": "failed",
                    "document": str(chunk_file),
                    "nct_id": nct_id,
                    "document_name": document_name,
                    "error": str(exc),
                }
            for idx, response in enumerate(image_embeddings):
                if not response:
                    continue
                
                # Get figure metadata from the prepared image
                figure_metadata = figure_images[response.index].metadata if response.index < len(figure_images) else {}
                figure_id = figure_metadata.get("figure_id", f"figure-{idx:04d}")
                parent_chunk_id = figure_metadata.get("parent_chunk_id", f"figure-{figure_id}")
                
                base_metadata = {
                    "chunk_id": f"{parent_chunk_id}-image",
                    "artefact_type": "figure_image",
                    "figure_id": figure_id,
                    "parent_chunk_id": parent_chunk_id,
                    "source_path": str(figure_source_path),
                    "page_reference": figure_metadata.get("page_reference"),
                    "image_path": figure_metadata.get("image_path"),
                }
                
                base_metadata = _prepare_embedding_metadata(
                    base_metadata,
                    nct_id=nct_id,
                    document_name=document_name,
                    embedding_model=response.model,
                    artefact_type="figure_image",
                    source_path=str(figure_source_path),
                    page_reference=figure_metadata.get("page_reference"),
                )
                
                chunk_id = str(base_metadata.get("chunk_id") or f"figure-image-{response.index:04d}")
                records.append(
                    EmbeddingRecord(
                        chunk_id=chunk_id,
                        embedding=response.embedding,
                        metadata=base_metadata,
                    )
                )
                image_record_count += 1

        writer.write(nct_id, document_name, records)
        total_embeddings = len(records)
        return {
            "status": "processed",
            "document": str(chunk_file),
            "nct_id": nct_id,
            "document_name": document_name,
            "chunk_count": total_embeddings,
            "text_embedding_count": text_record_count,
            "image_embedding_count": image_record_count,
        }


def _expand_chunk_items(
    chunk_list: Iterable[Dict[str, object]],
    *,
    max_tokens: int,
    tokenizer: Optional["PreTrainedTokenizerBase"],
) -> List[_PreparedSegment]:
    segments: List[_PreparedSegment] = []
    safe_max_tokens = max(0, max_tokens or 0)
    for index, item in enumerate(chunk_list):
        if not isinstance(item, dict):
            continue
        segments.extend(
            _split_chunk_item(
                item,
                index=index,
                max_tokens=safe_max_tokens,
                tokenizer=tokenizer,
            )
        )
    return segments


def _split_chunk_item(
    item: Dict[str, object],
    *,
    index: int,
    max_tokens: int,
    tokenizer: Optional["PreTrainedTokenizerBase"],
) -> List[_PreparedSegment]:
    text = str(item.get("text", ""))
    section = item.get("section")
    token_count = _coerce_int(item.get("token_count"))
    start_word_index = _coerce_int(item.get("start_word_index")) or 0
    char_count = _coerce_int(item.get("char_count")) or len(text)
    base_chunk_id = str(item.get("id") or f"chunk-{index:04d}")

    metadata_base = {
        "parent_chunk_id": base_chunk_id,
        "section": section,
        "token_count": token_count,
        "start_word_index": start_word_index,
        "char_count": char_count,
    }

    token_ids: Optional[List[int]] = None
    if tokenizer is not None and max_tokens > 0 and text:
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:  # pragma: no cover - tokenizer failure path
            _LOGGER.warning(
                "embeddings | tokenizer encode failed | chunk=%s | error=%s | falling back to heuristic split",
                base_chunk_id,
                exc,
            )
            token_ids = None

    token_ids_length = len(token_ids) if token_ids is not None else None
    if token_ids_length is not None:
        metadata_base["token_count"] = token_ids_length

    effective_token_count = token_count or token_ids_length or 0
    needs_split = max_tokens > 0 and effective_token_count > max_tokens

    # If the tokenizer truncated the sequence to its model_max_length, the length we
    # observe may be equal to max_tokens even though the original chunk is longer.
    if (
        needs_split
        and token_ids is not None
        and token_ids_length is not None
        and token_ids_length <= max_tokens
        and token_count
        and token_count > token_ids_length
    ):
        _LOGGER.debug(
            "embeddings | tokenizer truncated chunk below requested max | chunk=%s | token_count=%s | tokenizer_len=%s",
            base_chunk_id,
            token_count,
            token_ids_length,
        )
        token_ids = None
        token_ids_length = None

    if token_ids is not None and token_ids_length is not None and token_ids_length > max_tokens:
        segments = _segment_with_tokenizer(
            tokenizer=tokenizer,
            token_ids=token_ids,
            max_tokens=max_tokens,
            base_metadata=metadata_base,
            base_chunk_id=base_chunk_id,
            start_word_index=start_word_index,
        )
        if segments:
            return segments

    if not needs_split:
        metadata = {**metadata_base, "chunk_id": base_chunk_id, "segment_index": 0, "segment_count": 1}
        return [_PreparedSegment(text=text, metadata=metadata)]

    words = text.split()
    if not words:
        metadata = {**metadata_base, "chunk_id": base_chunk_id, "segment_index": 0, "segment_count": 1}
        return [_PreparedSegment(text=text, metadata=metadata)]

    desired_segments = max(1, math.ceil(token_count / max_tokens))
    words_per_segment = max(1, math.ceil(len(words) / desired_segments))
    approx_tokens_per_word = token_count / len(words)

    segments = []
    for segment_index, start in enumerate(range(0, len(words), words_per_segment)):
        segment_words = words[start : start + words_per_segment]
        if not segment_words:
            continue
        segment_text = " ".join(segment_words)
        if not segment_text:
            continue
        segment_chunk_id = base_chunk_id if segment_index == 0 else f"{base_chunk_id}-part{segment_index:02d}"
        estimated_tokens = int(round(len(segment_words) * approx_tokens_per_word))
        if estimated_tokens <= 0:
            estimated_tokens = len(segment_words)
        segment_token_count = min(max_tokens, estimated_tokens)
        segment_metadata = {
            **metadata_base,
            "chunk_id": segment_chunk_id,
            "token_count": segment_token_count,
            "start_word_index": start_word_index + start,
            "char_count": len(segment_text),
            "segment_index": segment_index,
            "segment_count": 0,
        }
        segments.append(_PreparedSegment(text=segment_text, metadata=segment_metadata))

    if not segments:
        metadata = {**metadata_base, "chunk_id": base_chunk_id, "segment_index": 0, "segment_count": 1}
        return [_PreparedSegment(text=text, metadata=metadata)]

    segment_total = len(segments)
    for segment in segments:
        segment.metadata["segment_count"] = segment_total

    return segments


def _segment_with_tokenizer(
    *,
    tokenizer: Optional["PreTrainedTokenizerBase"],
    token_ids: List[int],
    max_tokens: int,
    base_metadata: Dict[str, object],
    base_chunk_id: str,
    start_word_index: int,
) -> List[_PreparedSegment]:
    if tokenizer is None or not token_ids:
        return []

    segments: List[_PreparedSegment] = []
    word_offset = start_word_index
    for segment_index, start_token in enumerate(range(0, len(token_ids), max_tokens)):
        token_slice = token_ids[start_token : start_token + max_tokens]
        if not token_slice:
            continue
        try:
            segment_text = tokenizer.decode(  # type: ignore[union-attr]
                token_slice,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:  # pragma: no cover - tokenizer failure path
            _LOGGER.warning(
                "embeddings | tokenizer decode failed | chunk=%s | error=%s | skipping slice",
                base_chunk_id,
                exc,
            )
            continue

        segment_text = segment_text.strip()
        if not segment_text:
            continue

        segment_words = segment_text.split()
        segment_chunk_id = base_chunk_id if segment_index == 0 else f"{base_chunk_id}-part{segment_index:02d}"
        segment_word_count = len(segment_words)
        segment_metadata = {
            **base_metadata,
            "chunk_id": segment_chunk_id,
            "token_count": len(token_slice),
            "start_word_index": word_offset,
            "char_count": len(segment_text),
            "segment_index": segment_index,
            "segment_count": 0,
        }
        segments.append(_PreparedSegment(text=segment_text, metadata=segment_metadata))
        if segment_word_count:
            word_offset += segment_word_count

    if not segments:
        return []

    segment_total = len(segments)
    for segment in segments:
        segment.metadata["segment_count"] = segment_total

    return segments


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _load_table_segments(
    tables_root: Path,
    relative: Path,
    *,
    document_stem: str,
    source_path: str,
) -> List[_PreparedSegment]:
    path = tables_root / relative
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - table load failure
        _LOGGER.warning(
            "embeddings | failed to load table artefacts | path=%s | error=%s",
            path,
            exc,
        )
        return []

    if isinstance(payload, dict):
        items = payload.get("tables") or payload.get("data") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    segments: List[_PreparedSegment] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        caption = _clean_text(item.get("caption"))
        markdown = _clean_text(item.get("markdown"))
        html = _clean_text(item.get("html"))
        text_blocks = [block for block in (caption, markdown) if block]
        if not text_blocks and html:
            text_blocks.append(html)
        text = "\n\n".join(block.strip() for block in text_blocks if block).strip()
        if not text:
            continue
        table_id = item.get("id")
        base_identifier = _normalise_identifier(table_id, f"{document_stem}-table-{index:04d}")
        chunk_id = f"table-{base_identifier}"
        metadata = {
            "chunk_id": chunk_id,
            "segment_index": 0,
            "segment_count": 1,
            "artefact_type": "table",
            "table_id": table_id or base_identifier,
            "parent_chunk_id": chunk_id,
            "source_path": source_path,
            "page_reference": None,
        }
        if caption:
            metadata.setdefault("caption", caption)
        segments.append(_PreparedSegment(text=text, metadata=metadata))
    return segments


def _load_figure_segments(
    figures_root: Path,
    relative: Path,
    *,
    document_stem: str,
    processing_root: Path,
    source_path: str,
) -> Tuple[List[_PreparedSegment], List[_PreparedImage]]:
    path = figures_root / relative
    if not path.exists():
        return [], []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - figure load failure
        _LOGGER.warning(
            "embeddings | failed to load figure artefacts | path=%s | error=%s",
            path,
            exc,
        )
        return [], []

    if not isinstance(payload, list):
        return [], []

    text_segments: List[_PreparedSegment] = []
    image_segments: List[_PreparedImage] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        figure_id = item.get("id")
        base_identifier = _normalise_identifier(figure_id, f"{document_stem}-figure-{index:04d}")
        parent_chunk_id = f"figure-{base_identifier}"
        page_ref = _extract_page_reference(item.get("provenance"))

        caption = _clean_text(item.get("caption"))
        if caption:
            caption_chunk_id = f"{parent_chunk_id}-caption"
            metadata = {
                "chunk_id": caption_chunk_id,
                "segment_index": 0,
                "segment_count": 1,
                "artefact_type": "figure_caption",
                "figure_id": figure_id or base_identifier,
                "parent_chunk_id": parent_chunk_id,
                "source_path": source_path,
                "page_reference": page_ref,
            }
            text_segments.append(_PreparedSegment(text=caption, metadata=metadata))

        image_path_value = item.get("image_path") or item.get("image_uri")
        if image_path_value:
            image_path = (processing_root / image_path_value).resolve()
            metadata = {
                "chunk_id": f"{parent_chunk_id}-image",
                "artefact_type": "figure_image",
                "figure_id": figure_id or base_identifier,
                "parent_chunk_id": parent_chunk_id,
                "source_path": source_path,
                "page_reference": page_ref,
                "image_path": str(image_path_value),
            }
            image_segments.append(_PreparedImage(path=image_path, metadata=metadata))

    return text_segments, image_segments


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.strip()


def _normalise_identifier(raw_id: object, fallback: str) -> str:
    candidate = str(raw_id or "").strip()
    candidate = candidate.lstrip("#/")
    candidate = candidate.replace("/", "-")
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate)
    candidate = candidate.strip("- ")
    return candidate or fallback


def _extract_page_reference(provenance: object) -> Optional[int]:
    if isinstance(provenance, list):
        for entry in provenance:
            if not isinstance(entry, dict):
                continue
            for key in ("page_no", "page", "page_number"):
                if key in entry and entry[key] is not None:
                    try:
                        return int(entry[key])
                    except (TypeError, ValueError):  # pragma: no cover - malformed page
                        continue
    return None


def _prepare_embedding_metadata(
    metadata: Dict[str, object],
    *,
    nct_id: str,
    document_name: str,
    embedding_model: str,
    artefact_type: Optional[str] = None,
    source_path: Optional[str] = None,
    page_reference: Optional[int] = None,
) -> Dict[str, object]:
    normalised = dict(metadata)
    chunk_id = normalised.get("chunk_id")
    if chunk_id is None:
        chunk_id = f"chunk-{hash(frozenset(normalised.items())) & 0xFFFFFFFF:08x}"
        normalised["chunk_id"] = chunk_id
    normalised["nct_id"] = nct_id
    normalised["document_name"] = document_name
    normalised["model"] = embedding_model
    if artefact_type is not None:
        normalised["artefact_type"] = artefact_type
    else:
        normalised.setdefault("artefact_type", "chunk")
    if source_path is not None:
        normalised["source_path"] = source_path
    else:
        normalised.setdefault("source_path", None)
    if page_reference is not None:
        normalised["page_reference"] = page_reference
    else:
        if "page_reference" not in normalised and "page_ref" in normalised:
            normalised["page_reference"] = normalised.pop("page_ref")
        if "page_reference" not in normalised and "page" in normalised:
            normalised["page_reference"] = normalised.pop("page")
    return normalised


__all__ = ["EmbeddingPhase"]
