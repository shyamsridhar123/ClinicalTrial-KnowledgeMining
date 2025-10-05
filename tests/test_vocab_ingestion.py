from __future__ import annotations

from pathlib import Path

import pytest

from docintel.repository.ingestion import (
    RepositoryIngestionOrchestrator,
    RxNormLoader,
    SnomedLoader,
    UMLSLoader,
)

FIXTURE_ROOT = Path(__file__).parent / "data" / "vocab"


def test_umls_loader_parses_nodes_and_edges() -> None:
    loader = UMLSLoader(FIXTURE_ROOT / "umls")
    nodes = list(loader.iter_nodes())
    edges = list(loader.iter_edges())

    assert {node.code for node in nodes} == {"C0000005", "C0000006"}
    human = next(node for node in nodes if node.code == "C0000005")
    assert human.display_name == "Human"
    assert "Homo sapiens" in human.metadata["synonyms"]
    assert human.metadata["semantic_types"] == ["Disease or Syndrome"]
    assert edges and edges[0].predicate == "PAR"


def test_rxnorm_loader_returns_preferred_terms() -> None:
    loader = RxNormLoader(FIXTURE_ROOT / "rxnorm")
    nodes = list(loader.iter_nodes())
    assert {node.code for node in nodes} == {"12345", "12346"}
    combo = next(node for node in nodes if node.code == "12345")
    assert "Acetaminophen" in combo.metadata["synonyms"]
    edges = list(loader.iter_edges())
    assert edges and edges[0].source_code == "12345"


def test_snomed_loader_parses_concepts() -> None:
    loader = SnomedLoader(FIXTURE_ROOT / "snomed")
    nodes = list(loader.iter_nodes())
    assert {node.code for node in nodes} == {"100001", "100002"}
    pain = next(node for node in nodes if node.code == "100001")
    assert pain.display_name == "Pain (finding)"
    assert "Pain" in pain.metadata["synonyms"]
    edges = list(loader.iter_edges())
    assert edges and edges[0].target_code == "100001"


def test_orchestrator_dry_run_counts() -> None:
    loader = UMLSLoader(FIXTURE_ROOT / "umls")
    orchestrator = RepositoryIngestionOrchestrator(
        conn=None,
        schema="docintel",
        batch_size=10,
        edge_batch_size=10,
        checksum_algorithm="sha256",
    )
    result = orchestrator.ingest(loader, dry_run=True)
    assert result.dry_run is True
    assert result.nodes_processed == 2
    assert result.edges_processed == 1


@pytest.mark.parametrize("algorithm", ["sha256", "sha512"])
def test_orchestrator_accepts_checksum_algorithm(algorithm: str) -> None:
    loader = RxNormLoader(FIXTURE_ROOT / "rxnorm")
    orchestrator = RepositoryIngestionOrchestrator(
        conn=None,
        schema="docintel",
        batch_size=5,
        edge_batch_size=5,
        checksum_algorithm=algorithm,
    )
    result = orchestrator.ingest(loader, dry_run=True, metadata={"invocation": "test"})
    assert result.nodes_processed == 2
    assert result.edges_processed == 1
