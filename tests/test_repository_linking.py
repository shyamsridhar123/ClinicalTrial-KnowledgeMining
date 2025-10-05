from __future__ import annotations

import json
import uuid
from psycopg.types.json import Json  # type: ignore[import-not-found]
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

from docintel.knowledge_graph.enhanced_extraction import (
    EnhancedClinicalEntity,
    _derive_repository_link,
)
from docintel.knowledge_graph.entity_normalization import (
    ClinicalVocabulary,
    EntityNormalizationResult,
    NormalizedEntity,
)
from docintel.knowledge_graph.graph_construction import (
    KnowledgeGraphBuilder,
    process_chunk_to_graph,
)

from docintel.knowledge_graph.triple_extraction import ClinicalEntity, TripleExtractionResult
from docintel.repository.constants import NODE_NAMESPACE


def _build_normalization(concept_id: str, vocabulary: ClinicalVocabulary) -> EntityNormalizationResult:
    normalized = NormalizedEntity(
        original_text="Acetaminophen",
        normalized_text="Acetaminophen",
        vocabulary=vocabulary,
        concept_id=concept_id,
        concept_name="Acetaminophen",
        semantic_type="Pharmacologic Substance",
        confidence_score=0.93,
        synonyms=["Paracetamol"],
    )
    return EntityNormalizationResult(
        original_entity="Acetaminophen",
        entity_type="medication",
        normalizations=[normalized],
        best_match=normalized,
        processing_metadata={"source": "test"},
    )


def test_enhanced_entity_assigns_repository_node_id() -> None:
    entity = ClinicalEntity(
        text="Acetaminophen",
        entity_type="medication",
        start_char=0,
        end_char=13,
        confidence=0.95,
    )
    normalization_result = _build_normalization("RXNORM:12345", ClinicalVocabulary.RXNORM)

    enhanced = EnhancedClinicalEntity.from_clinical_entity(entity, normalization_result)

    expected_uuid = uuid.uuid5(NODE_NAMESPACE, "rxnorm:12345")
    assert enhanced.repository_node_id == str(expected_uuid)
    assert entity.repository_node_id == str(expected_uuid)
    assert entity.repository_vocabulary == "rxnorm"
    assert entity.repository_code == "12345"
    assert entity.normalization_data is not None


def test_derive_repository_link_handles_prefixed_and_plain_codes() -> None:
    prefixed_id = "UMLS:C0027497"
    plain_id = "C0027497"
    prefixed_uuid, prefixed_code = _derive_repository_link("umls", prefixed_id)
    plain_uuid, plain_code = _derive_repository_link("umls", plain_id)

    expected_uuid = uuid.uuid5(NODE_NAMESPACE, "umls:C0027497")
    assert prefixed_uuid == str(expected_uuid)
    assert plain_uuid == str(expected_uuid)
    assert prefixed_code == "C0027497"
    assert plain_code == "C0027497"


def test_insert_entity_persists_repository_node_id():
    builder = KnowledgeGraphBuilder()
    cursor = MagicMock()
    chunk_id = uuid.uuid4()
    meta_graph_id = uuid.uuid4()
    repo_uuid = uuid.uuid5(NODE_NAMESPACE, "rxnorm:12345")

    entity = ClinicalEntity(
        text="Acetaminophen",
        entity_type="medication",
        start_char=0,
        end_char=13,
        confidence=0.9,
    )
    entity.repository_node_id = str(repo_uuid)
    entity.repository_vocabulary = "rxnorm"
    entity.repository_code = "12345"
    entity.normalization_data = {
        "repository": {
            "repository_node_id": str(repo_uuid),
            "vocabulary": "rxnorm",
            "code": "12345",
        }
    }

    builder._insert_entity(cursor, meta_graph_id, chunk_id, entity)

    assert cursor.execute.called
    inserted_values = cursor.execute.call_args[0][1]
    assert inserted_values[10] == repo_uuid
    normalization_json = inserted_values[14]
    assert normalization_json is not None
    payload = json.loads(normalization_json)
    assert payload["repository"]["repository_node_id"] == str(repo_uuid)
    assert payload["repository"]["code"] == "12345"


def test_collect_repo_link_records_payload(monkeypatch):
    builder = KnowledgeGraphBuilder()
    repo_links: List[Tuple[object, ...]] = []
    meta_graph_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    repo_uuid = uuid.uuid5(NODE_NAMESPACE, "rxnorm:12345")

    entity = ClinicalEntity(
        text="Acetaminophen",
        entity_type="medication",
        start_char=0,
        end_char=13,
        confidence=0.9,
    )
    entity.repository_node_id = str(repo_uuid)
    entity.repository_vocabulary = "rxnorm"
    entity.repository_code = "12345"
    entity.normalization_data = {"repository": {"code": "12345"}}

    builder._collect_repo_link(repo_links, meta_graph_id, chunk_id, entity_id, entity)

    assert len(repo_links) == 1
    record = repo_links[0]
    assert record[1] == meta_graph_id
    assert record[2] == chunk_id
    assert record[3] == entity_id
    assert record[4] == str(repo_uuid)
    assert record[5] == "rxnorm"
    assert record[6] == "12345"
    assert record[7] == pytest.approx(0.9)
    assert record[8] == "normalization"


def test_write_repo_links_executes_insert(monkeypatch):
    builder = KnowledgeGraphBuilder()
    cursor = MagicMock()
    repo_uuid = uuid.uuid5(NODE_NAMESPACE, "rxnorm:12345")
    record = (
        str(uuid.uuid4()),
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
        str(repo_uuid),
        "rxnorm",
        "12345",
        0.95,
        "normalization",
        Json({"repository": {"code": "12345"}}),
    )

    builder._write_repo_links(cursor, [record])

    cursor.executemany.assert_called_once()
    args, _ = cursor.executemany.call_args
    assert "repo_entity_links" in args[0].as_string(cursor)
    assert args[1][0][4] == str(repo_uuid)


@pytest.mark.asyncio
async def test_process_chunk_to_graph_uses_enhanced_extractor(monkeypatch):
    chunk_id = uuid.uuid4()
    expected_meta_graph_id = uuid.uuid4()
    fake_result = TripleExtractionResult(entities=[], relations=[], processing_metadata={})

    async def fake_extract(text: str, incoming_chunk_id: uuid.UUID):
        assert incoming_chunk_id == chunk_id
        assert text == "synthetic"
        return fake_result

    async def fake_create(self, incoming_chunk_id: uuid.UUID, result: TripleExtractionResult) -> uuid.UUID:
        assert incoming_chunk_id == chunk_id
        assert result is fake_result
        return expected_meta_graph_id

    monkeypatch.setattr(
        "docintel.knowledge_graph.enhanced_extraction.extract_and_normalize_clinical_data",
        fake_extract,
    )
    monkeypatch.setattr(KnowledgeGraphBuilder, "create_meta_graph", fake_create, raising=False)

    meta_graph_id = await process_chunk_to_graph(chunk_id, "synthetic")
    assert meta_graph_id == expected_meta_graph_id