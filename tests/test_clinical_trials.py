from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pytest

from docintel.ingestion import (
    ClinicalTrialsCollector,
    ClinicalTrialsNotFoundError,
    DownloadedDocument,
)
from docintel.config import DataCollectionSettings
from docintel.storage import build_storage_layout, ensure_storage_layout


class DummyClient:
    def __init__(self, settings: DataCollectionSettings):
        self.settings = settings
        self._documents = [
            {
                "documentUrl": "https://example.org/doc.pdf",
                "documentType": "Study Protocol",
                "contentType": "application/pdf",
            }
        ]
        self._studies = [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00000001",
                        "briefTitle": "Example Study",
                        "officialTitle": "An Illustrative Trial",
                    },
                    "descriptionModule": {
                        "briefSummary": "Summary",
                    },
                    "designModule": {
                        "studyType": "Interventional",
                        "phases": ["Phase 3"],
                    },
                    "outcomesModule": {
                        "primaryOutcomes": [{"measure": "Primary"}],
                        "secondaryOutcomes": [{"measure": "Secondary"}],
                    },
                    "statusModule": {
                        "overallStatus": "COMPLETED",
                        "startDateStruct": {"date": "2020-01-01"},
                        "primaryCompletionDateStruct": {"date": "2021-01-01"},
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {"name": "DocIntel"}
                    },
                    "conditionsModule": {"conditions": ["Condition"]},
                },
                "documentSection": {
                    "largeDocumentModule": {
                        "largeDocs": [
                            {
                                "typeAbbrev": "ICF",
                                "label": "Informed Consent Form",
                                "filename": "ICF_000.pdf",
                                "size": 1234,
                            }
                        ]
                    }
                },
            }
        ]

    async def __aenter__(self) -> "DummyClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def search_studies(
        self,
        *,
        max_studies: int,
        therapeutic_areas: Iterable[str],
        phases: Iterable[str],
        status: str,
        query_term: str | None = None,
        page_size: int | None = None,
    ) -> List[Dict[str, Any]]:
        limit = page_size or len(self._studies)
        return self._studies[:limit]

    async def list_documents(self, nct_id: str) -> List[Dict[str, Any]]:
        return self._documents

    async def download_document(self, nct_id: str, document: Dict[str, Any]) -> DownloadedDocument:
        filename = document.get("filename") or "Study_Protocol.pdf"
        return DownloadedDocument(
            nct_id=nct_id,
            file_name=filename,
            content=b"PDFDATA",
            metadata=document,
        )


class NoDocumentsClient(DummyClient):
    def __init__(self, settings: DataCollectionSettings):
        super().__init__(settings)
        self._studies = [
            {
                "protocolSection": self._studies[0]["protocolSection"],
            }
        ]

    async def list_documents(self, nct_id: str) -> List[Dict[str, Any]]:
        raise ClinicalTrialsNotFoundError("No documents", status=404)


@pytest.mark.asyncio
async def test_collector_creates_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCINTEL_STORAGE_ROOT", str(tmp_path))
    settings = DataCollectionSettings(
        storage_root=tmp_path,
        max_concurrent_downloads=2,
    )
    storage = build_storage_layout(settings)
    ensure_storage_layout(storage)

    monkeypatch.setattr("docintel.ingestion.collector.ClinicalTrialsClient", DummyClient)

    collector = ClinicalTrialsCollector(settings, storage)
    report = await collector.run(max_studies=1)

    assert report["statistics"]["successful_downloads"] == 1
    assert report["statistics"]["studies_without_documents"] == 0
    metadata_file = storage.metadata / "NCT00000001.json"
    assert metadata_file.exists()
    metadata = json.loads(metadata_file.read_text())
    assert metadata["nct_id"] == "NCT00000001"

    document_dir = storage.documents / "NCT00000001"
    files = list(document_dir.glob("*.pdf"))
    assert files, "document PDF should exist"
    assert files[0].name == "ICF_000.pdf"
    metadata_path = files[0].with_suffix(files[0].suffix + ".json")
    assert metadata_path.exists()
    document_metadata = json.loads(metadata_path.read_text())
    assert document_metadata["documentUrl"].endswith("/ICF_000.pdf")

    report_path = storage.root / "collection_report.json"
    assert report_path.exists()


@pytest.mark.asyncio
async def test_collector_skips_missing_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCINTEL_STORAGE_ROOT", str(tmp_path))
    settings = DataCollectionSettings(
        storage_root=tmp_path,
        max_concurrent_downloads=2,
    )
    storage = build_storage_layout(settings)
    ensure_storage_layout(storage)

    monkeypatch.setattr("docintel.ingestion.collector.ClinicalTrialsClient", NoDocumentsClient)

    collector = ClinicalTrialsCollector(settings, storage)
    report = await collector.run(max_studies=1)

    assert report["statistics"]["successful_downloads"] == 0
    assert report["statistics"]["failed_downloads"] == 0
    assert report["statistics"]["studies_without_documents"] == 1
    metadata_file = storage.metadata / "NCT00000001.json"
    assert not metadata_file.exists()
    documents_dir = storage.documents / "NCT00000001"
    assert not list(documents_dir.glob("*.pdf"))


class MixedDocumentsClient(DummyClient):
    def __init__(self, settings: DataCollectionSettings):
        super().__init__(settings)
        self._studies = [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT99999999",
                        "briefTitle": "Docless Study",
                    }
                }
            },
            self._studies[0],
        ]

    async def list_documents(self, nct_id: str) -> List[Dict[str, Any]]:
        if nct_id == "NCT99999999":
            return []
        return await super().list_documents(nct_id)


@pytest.mark.asyncio
async def test_collector_prioritises_studies_with_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCINTEL_STORAGE_ROOT", str(tmp_path))
    settings = DataCollectionSettings(
        storage_root=tmp_path,
        max_concurrent_downloads=2,
    )
    storage = build_storage_layout(settings)
    ensure_storage_layout(storage)

    monkeypatch.setattr("docintel.ingestion.collector.ClinicalTrialsClient", MixedDocumentsClient)

    collector = ClinicalTrialsCollector(settings, storage)
    report = await collector.run(max_studies=1)

    assert report["statistics"]["successful_downloads"] == 1
    assert report["statistics"]["studies_without_documents"] == 0

    doc_dir = storage.documents / "NCT00000001"
    files = list(doc_dir.glob("*.pdf"))
    assert files, "expected PDF for study with documents"
    docless_dir = storage.documents / "NCT99999999"
    assert not docless_dir.exists()
