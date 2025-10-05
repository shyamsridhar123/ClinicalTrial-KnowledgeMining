import json
from pathlib import Path
from typing import List, Tuple
from uuid import UUID

import pytest

from docintel.extract import ExtractionJob


class _DummyEntity:
    def __init__(self, text: str = "Entity", entity_type: str = "condition") -> None:
        self.text = text
        self.entity_type = entity_type
        self.start_char = 0
        self.end_char = len(text)
        self.confidence = 0.95
        self.normalized_id = None
        self.normalized_source = None
        self.context_flags = None


class _DummyRelation:
    def __init__(self) -> None:
        self.subject_entity = _DummyEntity("Subject", "condition")
        self.object_entity = _DummyEntity("Object", "medication")
        self.predicate = "treats"
        self.confidence = 0.85
        self.evidence_span = "Subject treats Object"


class _DummyResult:
    def __init__(self, chunk_uuid: UUID) -> None:
        self.entities = [_DummyEntity()]
        self.relations = [_DummyRelation()]
        self.processing_metadata = {"chunk_id": str(chunk_uuid)}
        self.normalization_stats = {"normalized_entities": 0}
        self.normalized_entities: List[_DummyEntity] = []


class _RecordingExtractor:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, UUID]] = []

    async def extract_and_normalize_triples(self, text: str, chunk_uuid: UUID) -> _DummyResult:
        self.calls.append((text, chunk_uuid))
        return _DummyResult(chunk_uuid)


def _write_chunk_payload(path: Path, payload: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_discover_chunks_uses_parsed_artifacts(tmp_path: Path) -> None:
    chunks_root = tmp_path / "chunks"
    sample_payload = [
        {
            "id": "NCT12345-chunk-0000",
            "text": "This is a sufficiently long chunk of clinical text that should be processed.",
            "token_count": 128,
            "contains_tables": False,
            "section_header": "Introduction",
        },
        {
            "id": "NCT12345-chunk-0001",
            "text": "Another chunk of text that belongs to the same document.",
            "token_count": 64,
            "contains_tables": True,
            "section_header": "Methods",
        },
    ]

    _write_chunk_payload(chunks_root / "NCT12345" / "Prot.json", sample_payload)

    job = ExtractionJob(extractor=_RecordingExtractor(), chunk_root=chunks_root, max_concurrency=1)

    discovered = job._discover_chunks(nct_id=None, limit=None)

    assert len(discovered) == 2
    assert discovered[0]["chunk_id"] == "NCT12345-chunk-0000"
    assert discovered[0]["document_id"] == "Prot"
    assert discovered[0]["metadata"]["token_count"] == 128
    assert isinstance(discovered[0]["chunk_uuid"], UUID)
    assert discovered[1]["metadata"]["section_header"] == "Methods"
    assert discovered[1]["sequence_index"] == 1

    limited = job._discover_chunks(nct_id=None, limit=1)
    assert len(limited) == 1
    assert limited[0]["chunk_id"] == "NCT12345-chunk-0000"


@pytest.mark.asyncio
async def test_extract_from_chunks_reuses_metadata(tmp_path: Path) -> None:
    chunks_root = tmp_path / "chunks"
    text_one = "This chunk has more than fifty characters to ensure it passes the filter." * 2
    text_two = "This second chunk also exceeds the minimum character requirement." * 2

    first_payload = {
        "id": "NCT54321-chunk-0000",
        "text": text_one,
        "token_count": 256,
        "contains_tables": False,
        "section_header": "Overview",
    }
    second_payload = {
        "id": "NCT54321-chunk-0001",
        "text": text_two,
        "token_count": 144,
        "contains_tables": True,
        "section_header": "Details",
    }

    _write_chunk_payload(chunks_root / "NCT54321" / "Study.json", [first_payload, second_payload])

    extractor = _RecordingExtractor()
    job = ExtractionJob(extractor=extractor, chunk_root=chunks_root, max_concurrency=2)

    report = await job.extract_from_chunks()

    assert report["chunks_discovered"] == 2
    assert report["chunks_processed"] == 2
    assert report["chunks_skipped"] == 0
    assert report["chunks_failed"] == 0
    assert len(extractor.calls) == 2
    assert all(isinstance(call[1], UUID) for call in extractor.calls)

    result_chunk_ids = {entry["chunk_id"] for entry in job.results}
    assert result_chunk_ids == {"NCT54321-chunk-0000", "NCT54321-chunk-0001"}

    result_chunk_uuids = {entry["chunk_uuid"] for entry in job.results}
    call_chunk_uuids = {str(call[1]) for call in extractor.calls}
    assert result_chunk_uuids == call_chunk_uuids

    first_result = min(job.results, key=lambda item: item["chunk_metadata"]["sequence_index"])
    assert first_result["chunk_metadata"]["section_header"] == "Overview"
    assert first_result["metadata"]["source_chunk_id"] == "NCT54321-chunk-0000"
    assert first_result["relations"][0]["predicate"] == "treats"