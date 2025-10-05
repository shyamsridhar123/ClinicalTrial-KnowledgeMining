from __future__ import annotations

import json
from pathlib import Path

import pytest

from docintel.config import DataCollectionSettings, ParsingSettings
from docintel.parsing import DoclingParseResult, ParsingOrchestrator
from docintel.parsing.artifacts import ArtifactWriter
from docintel.storage import (
    build_processing_layout,
    build_storage_layout,
    ensure_processing_layout,
    ensure_storage_layout,
)


class StubDoclingClient:
    def __init__(self, results: dict[str, DoclingParseResult]) -> None:
        self._results = results
        self.calls: list[tuple[Path, str | None]] = []

    async def parse_document(self, *, document_path: Path, ocr_text: str | None = None) -> DoclingParseResult:
        key = document_path.name if ocr_text is None else f"{document_path.name}:ocr"
        self.calls.append((document_path, ocr_text))
        return self._results[key]


class StubOcrEngine:
    def __init__(self, ocr_text: str | None) -> None:
        self._ocr_text = ocr_text
        self.calls = 0

    async def extract_text(self, document_path: Path) -> str | None:
        self.calls += 1
        return self._ocr_text


@pytest.mark.asyncio
async def test_parsing_orchestrator_processes_documents(tmp_path: Path) -> None:
    ingestion_root = tmp_path / "ingestion"
    processing_root = tmp_path / "processing"

    ingestion_settings = DataCollectionSettings(storage_root=ingestion_root)
    storage_layout = build_storage_layout(ingestion_settings)
    ensure_storage_layout(storage_layout)

    parsing_settings = ParsingSettings(processed_storage_root=processing_root, ocr_enabled=False)
    processing_layout = build_processing_layout(parsing_settings)
    ensure_processing_layout(processing_layout)

    nct_dir = storage_layout.documents / "NCT00000001"
    nct_dir.mkdir(parents=True, exist_ok=True)
    doc_path = nct_dir / "protocol.pdf"
    doc_path.write_bytes(b"PDFDATA")
    # Companion metadata generated during ingestion should be ignored by the parser.
    (nct_dir / "protocol.pdf.json").write_text("{}", encoding="utf-8")

    result = DoclingParseResult(
        document={"title": "Protocol"},
        markdown="# Protocol",
        html="<h1>Protocol</h1>",
        plain_text="This is a protocol document",
        tables=[],
        figures=[],
        chunks=[{"id": "chunk-0", "text": "This is a protocol document", "token_count": 5}],
    metadata={"model": "ibm-granite/granite-docling-258M"},
    )

    docling_client = StubDoclingClient({"protocol.pdf": result})
    ocr_engine = StubOcrEngine(None)
    artifact_writer = ArtifactWriter(processing_layout)

    orchestrator = ParsingOrchestrator(
        parsing_settings,
        storage_layout,
        processing_layout,
        docling_client,  # type: ignore[arg-type]
        ocr_engine,  # type: ignore[arg-type]
        artifact_writer,
    )

    report = await orchestrator.run()

    assert report["statistics"]["processed"] == 1
    assert report["statistics"]["failed"] == 0
    assert report["statistics"]["skipped_existing"] == 0

    structured = processing_layout.structured / "NCT00000001/protocol.json"
    assert structured.exists()
    chunks_file = processing_layout.chunks / "NCT00000001/protocol.json"
    assert chunks_file.exists()

    # Running again without force should skip the existing artefact
    second_report = await orchestrator.run()
    assert second_report["statistics"]["skipped_existing"] == 1

    # Ensure non-supported files were not scheduled for parsing.
    assert len(docling_client.calls) == 1
    assert docling_client.calls[0][0] == doc_path


@pytest.mark.asyncio
async def test_parsing_orchestrator_applies_ocr_when_plain_text_blank(tmp_path: Path) -> None:
    ingestion_root = tmp_path / "ingestion"
    processing_root = tmp_path / "processing"

    ingestion_settings = DataCollectionSettings(storage_root=ingestion_root)
    storage_layout = build_storage_layout(ingestion_settings)
    ensure_storage_layout(storage_layout)

    parsing_settings = ParsingSettings(processed_storage_root=processing_root, ocr_enabled=True)
    processing_layout = build_processing_layout(parsing_settings)
    ensure_processing_layout(processing_layout)

    nct_dir = storage_layout.documents / "NCT12345678"
    nct_dir.mkdir(parents=True, exist_ok=True)
    doc_path = nct_dir / "icf.pdf"
    doc_path.write_bytes(b"PDFDATA")

    blank_result = DoclingParseResult(
        document={},
        markdown="",
        html="",
        plain_text=" ",
        tables=[],
        figures=[],
        chunks=[],
        metadata={},
    )
    ocr_result = DoclingParseResult(
        document={"title": "ICF"},
        markdown="# Consent",
        html="<h1>Consent</h1>",
        plain_text="Consent text",
        tables=[],
        figures=[],
        chunks=[],
        metadata={},
    )

    docling_client = StubDoclingClient(
        {
            "icf.pdf": blank_result,
            "icf.pdf:ocr": ocr_result,
        }
    )
    ocr_engine = StubOcrEngine("Extracted OCR text")
    artifact_writer = ArtifactWriter(processing_layout)

    orchestrator = ParsingOrchestrator(
        parsing_settings,
        storage_layout,
        processing_layout,
        docling_client,  # type: ignore[arg-type]
        ocr_engine,  # type: ignore[arg-type]
        artifact_writer,
    )

    report = await orchestrator.run()

    assert report["statistics"]["ocr_used"] == 1
    chunks_file = processing_layout.chunks / "NCT12345678/icf.json"
    assert chunks_file.exists()
    # Chunks were empty, so fallback chunking should run
    chunks = chunks_file.read_text(encoding="utf-8")
    assert "chunk-" in chunks


@pytest.mark.asyncio
async def test_parsing_orchestrator_force_reparse(tmp_path: Path) -> None:
    ingestion_settings = DataCollectionSettings(storage_root=tmp_path / "ingestion")
    storage_layout = build_storage_layout(ingestion_settings)
    ensure_storage_layout(storage_layout)

    parsing_settings = ParsingSettings(processed_storage_root=tmp_path / "processing", ocr_enabled=False)
    processing_layout = build_processing_layout(parsing_settings)
    ensure_processing_layout(processing_layout)

    nct_dir = storage_layout.documents / "NCT88888888"
    nct_dir.mkdir(parents=True, exist_ok=True)
    doc_path = nct_dir / "sap.pdf"
    doc_path.write_bytes(b"PDFDATA")

    result_one = DoclingParseResult(
        document={"version": 1},
        markdown="Version 1",
        html="<p>Version 1</p>",
        plain_text="Version 1",
        tables=[],
        figures=[],
        chunks=[{"id": "chunk-0", "text": "Version 1", "token_count": 2}],
        metadata={},
    )
    result_two = DoclingParseResult(
        document={"version": 2},
        markdown="Version 2",
        html="<p>Version 2</p>",
        plain_text="Version 2",
        tables=[],
        figures=[],
        chunks=[{"id": "chunk-0", "text": "Version 2", "token_count": 2}],
        metadata={},
    )

    client = StubDoclingClient({"sap.pdf": result_one, "sap.pdf:ocr": result_two})
    artifact_writer = ArtifactWriter(processing_layout)
    orchestrator = ParsingOrchestrator(
        parsing_settings,
        storage_layout,
        processing_layout,
        client,  # type: ignore[arg-type]
        StubOcrEngine(None),  # type: ignore[arg-type]
        artifact_writer,
    )

    await orchestrator.run()
    structured_path = processing_layout.structured / "NCT88888888/sap.json"
    first_structured = json.loads(structured_path.read_text(encoding="utf-8"))
    assert first_structured["version"] == 1

    # Update stub to return different content when reprocessing
    client._results["sap.pdf"] = result_two
    await orchestrator.run(force_reparse=True)
    second_structured = json.loads(structured_path.read_text(encoding="utf-8"))
    assert second_structured["version"] == 2