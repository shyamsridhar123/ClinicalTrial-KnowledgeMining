"""Ingestion coordinator that orchestrates ClinicalTrials.gov downloads."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tenacity import RetryError

from ..config import DataCollectionSettings
from ..storage import StorageLayout, ensure_storage_layout
from .client import (
    ClinicalTrialsAPIError,
    ClinicalTrialsClient,
    ClinicalTrialsNotFoundError,
    DownloadedDocument,
    build_provided_docs_url,
)

_LOGGER = logging.getLogger(__name__)


class ClinicalTrialsCollector:
    """Coordinates metadata retrieval and document downloads."""

    def __init__(
        self,
        settings: DataCollectionSettings,
        storage: StorageLayout,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._logger = logger or _LOGGER
        ensure_storage_layout(storage)
        self._download_semaphore = asyncio.Semaphore(settings.max_concurrent_downloads)

    async def run(self, *, max_studies: int) -> Dict[str, Any]:
        if max_studies <= 0:
            raise ValueError("max_studies must be a positive integer")
        stats = {
            "total_downloaded": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "skipped_files": 0,
            "studies_without_documents": 0,
            "total_size_mb": 0.0,
        }
        downloaded_studies: set[str] = set()
        async with ClinicalTrialsClient(self._settings) as client:
            try:
                page_size = self._settings.desired_page_size(max_studies)
                studies = await client.search_studies(
                    max_studies=max_studies,
                    therapeutic_areas=self._settings.target_therapeutic_areas,
                    phases=self._settings.target_phases,
                    status=self._settings.study_status,
                    query_term=self._settings.search_query_term,
                    page_size=page_size,
                )
            except RetryError as exc:
                raise ClinicalTrialsAPIError("Failed to query ClinicalTrials.gov after retries") from exc

            candidates = _prioritize_studies_with_documents(studies)
            for raw_study in candidates[:max_studies]:
                processed = _extract_study_metadata(raw_study)
                if not processed:
                    stats["skipped_files"] += 1
                    continue
                nct_id = processed["nct_id"]
                documents = _extract_study_documents(nct_id, raw_study)
                if not documents:
                    try:
                        documents = await client.list_documents(nct_id)
                    except ClinicalTrialsNotFoundError:
                        self._logger.info("No documents available for study %s", nct_id)
                        self._delete_study_metadata(nct_id)
                        stats["studies_without_documents"] += 1
                        continue
                if not documents:
                    self._delete_study_metadata(nct_id)
                    stats["studies_without_documents"] += 1
                    continue
                downloads = await asyncio.gather(
                    *(
                        self._download_with_semaphore(client, nct_id, document)
                        for document in documents
                    ),
                    return_exceptions=True,
                )
                successful_docs: List[DownloadedDocument] = []
                for result in downloads:
                    stats["total_downloaded"] += 1
                    if isinstance(result, DownloadedDocument):
                        stats["successful_downloads"] += 1
                        stats["total_size_mb"] += result.size_bytes / (1024 * 1024)
                        self._write_document(result)
                        successful_docs.append(result)
                    else:
                        stats["failed_downloads"] += 1
                        self._logger.warning("Failed to download document: %s", result)
                if successful_docs:
                    self._write_study_metadata(processed)
                    downloaded_studies.add(nct_id)
                else:
                    self._delete_study_metadata(nct_id)
                    stats["studies_without_documents"] += 1

        stats["total_size_mb"] = round(stats["total_size_mb"], 4)
        report = {
            "collection_date": datetime.now(timezone.utc).isoformat(),
            "statistics": stats,
            "configuration": {
                "max_concurrent_downloads": self._settings.max_concurrent_downloads,
                "target_therapeutic_areas": self._settings.target_therapeutic_areas,
                "target_phases": self._settings.target_phases,
                "study_status": self._settings.study_status,
                "search_query_term": self._settings.search_query_term,
                "search_overfetch_multiplier": self._settings.search_overfetch_multiplier,
            },
        }
        self._prune_orphan_metadata(downloaded_studies)
        self._write_collection_report(report)
        return report

    async def _download_with_semaphore(
        self,
        client: ClinicalTrialsClient,
        nct_id: str,
        document: Dict[str, Any],
    ) -> DownloadedDocument:
        async with self._download_semaphore:
            try:
                return await client.download_document(nct_id, document)
            except RetryError as exc:
                raise ClinicalTrialsAPIError("Download failed after retries") from exc

    def _metadata_path(self, nct_id: str) -> Path:
        return self._storage.metadata / f"{nct_id}.json"

    def _delete_study_metadata(self, nct_id: str) -> None:
        path = self._metadata_path(nct_id)
        if path.exists():
            path.unlink()

    def _prune_orphan_metadata(self, downloaded_studies: Iterable[str]) -> None:
        metadata_dir = self._storage.metadata
        documents_dir = self._storage.documents
        if not metadata_dir.exists():
            return
        valid_ids = set(downloaded_studies)
        if documents_dir.exists():
            valid_ids.update({path.name for path in documents_dir.iterdir() if path.is_dir()})
        for metadata_file in metadata_dir.glob("*.json"):
            stem = metadata_file.stem
            if not stem.upper().startswith("NCT"):
                continue
            if stem not in valid_ids:
                metadata_file.unlink()

    def _write_study_metadata(self, metadata: Dict[str, Any]) -> None:
        nct_id = metadata["nct_id"]
        path = self._metadata_path(nct_id)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _write_document(self, document: DownloadedDocument) -> None:
        study_dir = self._storage.documents / document.nct_id
        study_dir.mkdir(parents=True, exist_ok=True)
        file_path = study_dir / document.file_name
        file_path.write_bytes(document.content)
        metadata_path = file_path.with_suffix(file_path.suffix + ".json")
        metadata_path.write_text(json.dumps(document.metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _write_collection_report(self, report: Dict[str, Any]) -> None:
        report_path = self._storage.root / "collection_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _extract_study_metadata(raw_study: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    protocol = raw_study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    nct_id = identification.get("nctId")
    if not nct_id:
        return None
    description = protocol.get("descriptionModule", {})
    design = protocol.get("designModule", {})
    outcomes = protocol.get("outcomesModule", {})
    status = protocol.get("statusModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})

    return {
        "nct_id": nct_id,
        "brief_title": identification.get("briefTitle", ""),
        "official_title": identification.get("officialTitle", ""),
        "study_type": design.get("studyType", ""),
        "phases": design.get("phases", []),
        "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
        "primary_outcomes": [o.get("measure", "") for o in outcomes.get("primaryOutcomes", [])],
        "secondary_outcomes": [o.get("measure", "") for o in outcomes.get("secondaryOutcomes", [])],
        "overall_status": status.get("overallStatus", ""),
        "start_date": status.get("startDateStruct", {}).get("date", ""),
        "completion_date": status.get("primaryCompletionDateStruct", {}).get("date", ""),
        "lead_sponsor": sponsor.get("leadSponsor", {}).get("name", ""),
    }


def _extract_study_documents(nct_id: str, raw_study: Dict[str, Any]) -> List[Dict[str, Any]]:
    document_section = raw_study.get("documentSection", {})
    large_docs_module = document_section.get("largeDocumentModule", {})
    large_docs = large_docs_module.get("largeDocs", [])
    documents: List[Dict[str, Any]] = []
    for entry in large_docs:
        filename = entry.get("filename")
        url = build_provided_docs_url(nct_id, filename) if filename else None
        if not url:
            continue
        document_type = entry.get("label") or entry.get("typeAbbrev") or "Study Document"
        payload = dict(entry)
        payload["documentUrl"] = url
        payload.setdefault("documentType", document_type)
        payload.setdefault("contentType", "application/pdf")
        documents.append(payload)
    return documents


def _prioritize_studies_with_documents(studies: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    with_documents: List[Dict[str, Any]] = []
    without_documents: List[Dict[str, Any]] = []
    for study in studies:
        identification = study.get("protocolSection", {}).get("identificationModule", {})
        nct_id = identification.get("nctId")
        if nct_id and _extract_study_documents(nct_id, study):
            with_documents.append(study)
        else:
            without_documents.append(study)
    return with_documents + without_documents


__all__ = [
    "ClinicalTrialsCollector",
]
