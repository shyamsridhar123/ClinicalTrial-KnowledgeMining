from __future__ import annotations

import json
import struct
from collections.abc import Generator
from pathlib import Path
from typing import Dict, List

import pytest  # type: ignore[import-not-found]

from docintel import config
from docintel.embeddings.client import EmbeddingResponse
from docintel.embeddings.phase import EmbeddingPhase
from docintel.embeddings.writer import EmbeddingRecord, EmbeddingWriter
from docintel.pipeline import PipelineContext
from docintel.storage import (
    build_embedding_layout,
    build_processing_layout,
    build_storage_layout,
    ensure_embedding_layout,
    ensure_processing_layout,
    ensure_storage_layout,
)

from docintel.embeddings import phase as embeddings_phase_module


class _StubEmbeddingClient:
    def __init__(self, *_, **__) -> None:
        self.calls: List[List[str]] = []
        self.closed = False

    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResponse]:
        self.calls.append(list(texts))
        return [
            EmbeddingResponse(embedding=[float(len(text)), float(index)], index=index, model="stub-model")
            for index, text in enumerate(texts)
        ]

    async def aclose(self) -> None:
        self.closed = True


class _StubTokenizer:
    model_max_length = 512

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._reverse: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> List[int]:  # noqa: ARG002 - signature compatibility
        tokens = text.split()
        ids: List[int] = []
        for token in tokens:
            if token not in self._vocab:
                token_id = len(self._vocab) + 1
                self._vocab[token] = token_id
                self._reverse[token_id] = token
            ids.append(self._vocab[token])
        return ids

    def decode(  # noqa: D401 - simple analog of HF API
        self,
        token_ids: List[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:  # noqa: ARG002 - keep parity with HF API
        return " ".join(self._reverse[token_id] for token_id in token_ids if token_id in self._reverse)


class _TruncatingTokenizer(_StubTokenizer):
    model_max_length = 128

    def encode(self, text: str, *, add_special_tokens: bool = False) -> List[int]:  # noqa: ARG002
        full = super().encode(text, add_special_tokens=add_special_tokens)
        return full[: self.model_max_length]


@pytest.fixture(autouse=True)
def clear_embedding_settings_cache() -> Generator[None, None, None]:
    config.get_embedding_settings.cache_clear()
    yield
    config.get_embedding_settings.cache_clear()


def test_embedding_writer_invokes_pgvector_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from docintel.embeddings import writer as writer_module

    spy_instances: List[object] = []
    spy_calls: List[tuple[str, str, List[tuple[EmbeddingRecord, Dict[str, object]]]]] = []

    class _SpySink:
        def __init__(self, settings: config.VectorDatabaseSettings) -> None:
            self.settings = settings
            spy_instances.append(self)

        def write(
            self,
            nct_id: str,
            document_name: str,
            payloads: List[tuple[EmbeddingRecord, Dict[str, object]]],
        ) -> None:
            spy_calls.append((nct_id, document_name, list(payloads)))

    monkeypatch.setattr(writer_module, "_PgvectorSink", _SpySink)

    embedding_settings = config.EmbeddingSettings(embedding_storage_root=tmp_path / "embeddings")
    embedding_layout = build_embedding_layout(embedding_settings)
    ensure_embedding_layout(embedding_layout)

    vector_settings = config.VectorDatabaseSettings(
        enabled=True,
        dsn="postgresql://postgres:postgres@localhost:5432/docintel",
        embedding_dimensions=2,
    )

    writer = EmbeddingWriter(
        embedding_layout,
        quantization_encoding="none",
        store_float32=True,
        vector_db_settings=vector_settings,
    )

    record = EmbeddingRecord(
        chunk_id="chunk-0",
        embedding=[0.1, 0.2],
        metadata={
            "chunk_id": "chunk-0",
            "model": "stub-model",
            "segment_index": 0,
            "segment_count": 1,
        },
    )

    output_path = writer.write("NCT00000001", "protocol.json", [record])

    assert output_path.exists()
    assert spy_instances, "pgvector sink should be initialised when enabled"
    assert spy_calls, "pgvector sink should capture write invocations"
    call_nct, call_doc, payloads = spy_calls[0]
    assert call_nct == "NCT00000001"
    assert call_doc == "protocol.json"
    assert len(payloads) == 1
    db_record, serialised = payloads[0]
    assert isinstance(db_record, EmbeddingRecord)
    assert serialised["chunk_id"] == "chunk-0"
    assert serialised["metadata"]["quantization_encoding"] == "none"


@pytest.mark.asyncio
async def test_embedding_phase_generates_vectors(tmp_path: Path) -> None:
    ingestion_settings = config.DataCollectionSettings(storage_root=tmp_path / "ingestion")
    storage_layout = build_storage_layout(ingestion_settings)
    ensure_storage_layout(storage_layout)

    parsing_settings = config.ParsingSettings(processed_storage_root=tmp_path / "processing", ocr_enabled=False)
    processing_layout = build_processing_layout(parsing_settings)
    ensure_processing_layout(processing_layout)

    embedding_settings = config.EmbeddingSettings(embedding_storage_root=tmp_path / "embeddings", embedding_batch_size=4)
    embedding_layout = build_embedding_layout(embedding_settings)
    ensure_embedding_layout(embedding_layout)

    chunk_dir = processing_layout.chunks / "NCT00000001"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = chunk_dir / "protocol.json"
    chunk_file.write_text(
        json.dumps(
            [
                {"id": "chunk-0", "text": "protocol chunk one", "token_count": 3},
                {"id": "chunk-1", "text": "protocol chunk two", "token_count": 4},
            ]
        ),
        encoding="utf-8",
    )

    created_clients: List[_StubEmbeddingClient] = []

    def _client_factory(*args, **kwargs):
        client = _StubEmbeddingClient()
        created_clients.append(client)
        return client

    phase = EmbeddingPhase(
        force_reembed=True,
        client_factory=_client_factory,
        tokenizer_factory=lambda *_: None,
    )
    context = PipelineContext(
        ingestion_settings=ingestion_settings,
        parsing_settings=parsing_settings,
        embedding_settings=embedding_settings,
        storage_layout=storage_layout,
        processing_layout=processing_layout,
        embedding_layout=embedding_layout,
    )

    result = await phase.run(context)

    assert result.succeeded
    report = result.details["report"]
    assert report["statistics"]["documents_processed"] == 1
    assert report["statistics"]["chunks_embedded"] == 2

    output_file = embedding_layout.vectors / "NCT00000001/protocol.jsonl"
    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first_record = json.loads(lines[0])
    assert first_record["chunk_id"] == "chunk-0"
    assert first_record["metadata"]["model"] == "stub-model"
    assert pytest.approx(first_record["embedding"][0]) == len("protocol chunk one")

    assert created_clients and created_clients[0].closed
    assert created_clients[0].calls == [["protocol chunk one", "protocol chunk two"]]


@pytest.mark.asyncio
async def test_embedding_phase_skips_existing_vectors(tmp_path: Path) -> None:
    ingestion_settings = config.DataCollectionSettings(storage_root=tmp_path / "ingestion")
    storage_layout = build_storage_layout(ingestion_settings)
    ensure_storage_layout(storage_layout)

    parsing_settings = config.ParsingSettings(processed_storage_root=tmp_path / "processing")
    processing_layout = build_processing_layout(parsing_settings)
    ensure_processing_layout(processing_layout)

    embedding_settings = config.EmbeddingSettings(embedding_storage_root=tmp_path / "embeddings")
    embedding_layout = build_embedding_layout(embedding_settings)
    ensure_embedding_layout(embedding_layout)

    chunk_dir = processing_layout.chunks / "NCT99999999"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = chunk_dir / "sap.json"
    chunk_file.write_text(json.dumps([]), encoding="utf-8")

    writer = EmbeddingWriter(embedding_layout)
    writer.write("NCT99999999", "sap.json", [])

    client = _StubEmbeddingClient()
    phase = EmbeddingPhase(client_factory=lambda *_: client, tokenizer_factory=lambda *_: None)

    context = PipelineContext(
        ingestion_settings=ingestion_settings,
        parsing_settings=parsing_settings,
        embedding_settings=embedding_settings,
        storage_layout=storage_layout,
        processing_layout=processing_layout,
        embedding_layout=embedding_layout,
    )

    result = await phase.run(context)

    assert result.succeeded
    report = result.details["report"]
    assert report["statistics"]["documents_skipped"] == 1
    assert client.calls == []


@pytest.mark.asyncio
async def test_embedding_phase_splits_long_chunks(tmp_path: Path) -> None:
    ingestion_settings = config.DataCollectionSettings(storage_root=tmp_path / "ingestion")
    storage_layout = build_storage_layout(ingestion_settings)
    ensure_storage_layout(storage_layout)

    parsing_settings = config.ParsingSettings(processed_storage_root=tmp_path / "processing", ocr_enabled=False)
    processing_layout = build_processing_layout(parsing_settings)
    ensure_processing_layout(processing_layout)

    embedding_settings = config.EmbeddingSettings(
        embedding_storage_root=tmp_path / "embeddings",
        embedding_batch_size=2,
        embedding_max_tokens=128,
    )
    embedding_layout = build_embedding_layout(embedding_settings)
    ensure_embedding_layout(embedding_layout)

    chunk_dir = processing_layout.chunks / "NCT12345678"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = chunk_dir / "long.json"
    words = [f"word{i}" for i in range(600)]
    chunk_file.write_text(
        json.dumps(
            [
                {
                    "id": "chunk-large",
                    "text": " ".join(words),
                    "token_count": 600,
                    "start_word_index": 0,
                }
            ]
        ),
        encoding="utf-8",
    )

    created_clients: List[_StubEmbeddingClient] = []
    tokenizer = _StubTokenizer()

    def _client_factory(*args, **kwargs):
        client = _StubEmbeddingClient()
        created_clients.append(client)
        return client

    phase = EmbeddingPhase(
        force_reembed=True,
        client_factory=_client_factory,
        tokenizer_factory=lambda *_: tokenizer,
    )

    context = PipelineContext(
        ingestion_settings=ingestion_settings,
        parsing_settings=parsing_settings,
        embedding_settings=embedding_settings,
        storage_layout=storage_layout,
        processing_layout=processing_layout,
        embedding_layout=embedding_layout,
    )

    result = await phase.run(context)

    assert result.succeeded
    report = result.details["report"]
    assert report["statistics"]["documents_processed"] == 1
    assert report["statistics"]["chunks_embedded"] == 5

    output_file = embedding_layout.vectors / "NCT12345678/long.jsonl"
    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").strip().splitlines()]
    chunk_ids = [record["chunk_id"] for record in records]
    assert chunk_ids == [
        "chunk-large",
        "chunk-large-part01",
        "chunk-large-part02",
        "chunk-large-part03",
        "chunk-large-part04",
    ]
    for index, record in enumerate(records):
        metadata = record["metadata"]
        assert metadata["segment_index"] == index
        assert metadata["segment_count"] == 5
        assert metadata["parent_chunk_id"] == "chunk-large"
        assert metadata["chunk_id"] == chunk_ids[index]

    assert created_clients and created_clients[0].closed
    # Expect three batches due to embedding_batch_size=2 for five segments.
    assert created_clients[0].calls[0] == [" ".join(words[:128]), " ".join(words[128:256])]
    assert len(created_clients[0].calls) == 3
    # Final batch should contain the remaining single segment with <=128 words.
    assert all(len(call_text.split()) <= 128 for batch in created_clients[0].calls for call_text in batch)


@pytest.mark.asyncio
async def test_embedding_phase_handles_tokenizer_truncation(tmp_path: Path) -> None:
    ingestion_settings = config.DataCollectionSettings(storage_root=tmp_path / "ingestion")
    storage_layout = build_storage_layout(ingestion_settings)
    ensure_storage_layout(storage_layout)

    parsing_settings = config.ParsingSettings(processed_storage_root=tmp_path / "processing", ocr_enabled=False)
    processing_layout = build_processing_layout(parsing_settings)
    ensure_processing_layout(processing_layout)

    embedding_settings = config.EmbeddingSettings(
        embedding_storage_root=tmp_path / "embeddings",
        embedding_batch_size=2,
        embedding_max_tokens=128,
    )
    embedding_layout = build_embedding_layout(embedding_settings)
    ensure_embedding_layout(embedding_layout)

    chunk_dir = processing_layout.chunks / "NCT87654321"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = chunk_dir / "truncated.json"
    words = [f"tok{i}" for i in range(600)]
    chunk_file.write_text(
        json.dumps([
            {
                "id": "chunk-trunc",
                "text": " ".join(words),
                "token_count": 600,
                "start_word_index": 0,
            }
        ]),
        encoding="utf-8",
    )

    tokenizer = _TruncatingTokenizer()

    created_clients: List[_StubEmbeddingClient] = []

    def _client_factory(*args, **kwargs):
        client = _StubEmbeddingClient()
        created_clients.append(client)
        return client

    phase = EmbeddingPhase(
        force_reembed=True,
        client_factory=_client_factory,
        tokenizer_factory=lambda *_: tokenizer,
    )

    context = PipelineContext(
        ingestion_settings=ingestion_settings,
        parsing_settings=parsing_settings,
        embedding_settings=embedding_settings,
        storage_layout=storage_layout,
        processing_layout=processing_layout,
        embedding_layout=embedding_layout,
    )

    result = await phase.run(context)

    assert result.succeeded
    report = result.details["report"]
    assert report["statistics"]["documents_processed"] == 1
    assert report["statistics"]["chunks_embedded"] == 5

    output_file = embedding_layout.vectors / "NCT87654321/truncated.jsonl"
    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(records) == 5
    assert [record["metadata"]["segment_index"] for record in records] == list(range(5))

    assert created_clients and created_clients[0].closed
    first_batch = created_clients[0].calls[0]
    assert len(first_batch) == 2
    assert all(len(call_text.split()) <= 128 for call_text in first_batch)
    # Ensure later batches continue the sequence and maintain <=128 words per segment.
    for batch in created_clients[0].calls:
        for call_text in batch:
            assert len(call_text.split()) <= 128
    assert created_clients[0].calls[-1][-1].startswith("tok480")


@pytest.mark.asyncio
async def test_embedding_phase_supports_bfloat16_quantization(tmp_path: Path) -> None:
    ingestion_settings = config.DataCollectionSettings(storage_root=tmp_path / "ingestion")
    storage_layout = build_storage_layout(ingestion_settings)
    ensure_storage_layout(storage_layout)

    parsing_settings = config.ParsingSettings(processed_storage_root=tmp_path / "processing", ocr_enabled=False)
    processing_layout = build_processing_layout(parsing_settings)
    ensure_processing_layout(processing_layout)

    embedding_settings = config.EmbeddingSettings(
        embedding_storage_root=tmp_path / "embeddings",
        embedding_quantization_encoding="bfloat16",
        embedding_quantization_store_float32=False,
        embedding_batch_size=4,
    )
    embedding_layout = build_embedding_layout(embedding_settings)
    ensure_embedding_layout(embedding_layout)

    chunk_dir = processing_layout.chunks / "NCT13579135"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = chunk_dir / "protocol.json"
    chunk_file.write_text(
        json.dumps(
            [
                {"id": "chunk-0", "text": "quantization demo chunk", "token_count": 4},
            ]
        ),
        encoding="utf-8",
    )

    async def _stub_embed_texts(texts: List[str]) -> List[EmbeddingResponse]:
        return [EmbeddingResponse(embedding=[0.125, -0.5, 0.75], index=0, model="stub-model")]

    class _QuantStubClient(_StubEmbeddingClient):
        async def embed_texts(self, texts: List[str]) -> List[EmbeddingResponse]:  # type: ignore[override]
            return await _stub_embed_texts(texts)

    client = _QuantStubClient()

    phase = EmbeddingPhase(
        force_reembed=True,
        client_factory=lambda *_: client,
        tokenizer_factory=lambda *_: None,
    )

    context = PipelineContext(
        ingestion_settings=ingestion_settings,
        parsing_settings=parsing_settings,
        embedding_settings=embedding_settings,
        storage_layout=storage_layout,
        processing_layout=processing_layout,
        embedding_layout=embedding_layout,
    )

    result = await phase.run(context)

    assert result.succeeded
    output_file = embedding_layout.vectors / "NCT13579135/protocol.jsonl"
    payload = json.loads(output_file.read_text(encoding="utf-8").strip())
    assert "embedding" not in payload
    assert payload["metadata"]["quantization_encoding"] == "bfloat16"
    quantised = payload["embedding_quantized"]
    assert quantised["encoding"] == "bfloat16"
    assert len(quantised["values"]) == 3

    def _bfloat16_to_float(value: int) -> float:
        return struct.unpack(">f", struct.pack(">I", int(value) << 16))[0]

    reconstructed = [_bfloat16_to_float(val) for val in quantised["values"]]
    assert reconstructed[0] == pytest.approx(0.125, rel=1e-2, abs=1e-2)
    assert reconstructed[1] == pytest.approx(-0.5, rel=1e-2, abs=1e-2)
    assert reconstructed[2] == pytest.approx(0.75, rel=1e-2, abs=1e-2)


def test_tokenizer_candidates_biomedclip_mapping() -> None:
    candidates = embeddings_phase_module._tokenizer_name_candidates(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    assert candidates[0] == "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    assert "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" in candidates


def test_tokenizer_candidates_strip_prefix() -> None:
    candidates = embeddings_phase_module._tokenizer_name_candidates("hf_hub:foo/bar-baz")
    assert candidates[-1] == "foo/bar-baz"

