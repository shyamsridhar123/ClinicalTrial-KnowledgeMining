"""Client utilities for interacting with the ClinicalTrials.gov API."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import aiohttp
from aiohttp import ClientResponse, ClientSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import DataCollectionSettings

_LOGGER = logging.getLogger(__name__)
_FILENAME_SANITISER = re.compile(r"[^A-Za-z0-9_.-]+")
_PROVIDED_DOCS_BASE = "https://clinicaltrials.gov/ProvidedDocs"


class ClinicalTrialsAPIError(RuntimeError):
    """Raised when the ClinicalTrials.gov API returns an invalid response."""

    def __init__(self, message: str, *, status: Optional[int] = None, body: Optional[str] = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class ClinicalTrialsNotFoundError(ClinicalTrialsAPIError):
    """Raised when ClinicalTrials.gov responds with HTTP 404 for a resource."""


@dataclass(slots=True)
class DownloadedDocument:
    """Represents a downloaded study document."""

    nct_id: str
    file_name: str
    content: bytes
    metadata: Dict[str, Any]

    @property
    def size_bytes(self) -> int:
        return len(self.content)


def build_provided_docs_url(nct_id: str, filename: str) -> Optional[str]:
    """Return the download URL for a ProvidedDocs asset if both values are present."""

    if not nct_id or not filename:
        return None
    cleaned_id = nct_id.strip().upper()
    suffix = cleaned_id[-2:] if len(cleaned_id) >= 2 else cleaned_id
    return f"{_PROVIDED_DOCS_BASE}/{suffix}/{cleaned_id}/{filename}"


def _safe_join(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _detect_extension(content_type: Optional[str], url: str) -> str:
    if content_type and "pdf" in content_type.lower():
        return ".pdf"
    if content_type and "word" in content_type.lower():
        return ".docx"
    if url.lower().endswith((".pdf", ".doc", ".docx")):
        return Path(url).suffix
    return ".pdf"


def _sanitise_filename(name: str) -> str:
    name = _FILENAME_SANITISER.sub("_", name).strip("._")
    return name or "document.pdf"


async def _raise_for_status(response: ClientResponse) -> None:
    try:
        response.raise_for_status()
    except aiohttp.ClientResponseError as exc:  # pragma: no cover - network error path
        body = await response.text()
        message = f"ClinicalTrials.gov request failed with {exc.status}: {exc.message}. Body: {body[:200]}"
        if exc.status == 404:
            raise ClinicalTrialsNotFoundError(message, status=exc.status, body=body) from exc
        raise ClinicalTrialsAPIError(message, status=exc.status, body=body) from exc


class ClinicalTrialsClient:
    """Asynchronous client for the ClinicalTrials.gov v2 API."""

    def __init__(self, settings: DataCollectionSettings):
        self._settings = settings
        self._session: Optional[ClientSession] = None

    async def __aenter__(self) -> "ClinicalTrialsClient":
        timeout = aiohttp.ClientTimeout(total=self._settings.request_timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _require_session(self) -> ClientSession:
        if not self._session:
            raise RuntimeError("Client session not initialized. Use as an async context manager.")
        return self._session

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        session = self._require_session()
        url = _safe_join(str(self._settings.clinicaltrials_api_base), path)
        async with session.get(url, params=params) as response:
            await _raise_for_status(response)
            return await response.json()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def _download_bytes(self, url: str) -> bytes:
        session = self._require_session()
        async with session.get(url) as response:
            await _raise_for_status(response)
            return await response.read()

    async def search_studies(
        self,
        *,
        max_studies: int,
        therapeutic_areas: Iterable[str],
        phases: Iterable[str],
        status: str,
        query_term: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query_terms: List[str] = []
        if query_term:
            query_terms.append(query_term)
        if therapeutic_areas:
            query_terms.append("(" + " OR ".join(therapeutic_areas) + ")")
        if phases:
            phase_terms = [f'"{phase}"' for phase in phases]
            query_terms.append("(" + " OR ".join(phase_terms) + ")")
        if status:
            query_terms.append(status.lower())
        query = " AND ".join(query_terms) if query_terms else "completed"

        effective_page_size = page_size or max_studies
        effective_page_size = max(1, effective_page_size)
        params = {
            "format": "json",
            "query.term": query,
            "pageSize": min(effective_page_size, 100),
        }
        payload = await self._get_json("studies", params=params)
        studies = payload.get("studies", [])
        return studies

    async def list_documents(self, nct_id: str) -> List[Dict[str, Any]]:
        try:
            payload = await self._get_json(f"studies/{nct_id}/documents")
        except ClinicalTrialsNotFoundError:
            _LOGGER.info("No documents found for study %s", nct_id)
            return []
        return payload.get("studyDocuments", [])

    async def download_document(self, nct_id: str, document: Dict[str, Any]) -> DownloadedDocument:
        url = document.get("documentUrl")
        if not url:
            raise ClinicalTrialsAPIError(f"Document for study {nct_id} is missing a download URL")
        content = await self._download_bytes(url)
        filename = document.get("filename")
        if filename:
            safe_name = _sanitise_filename(filename)
        else:
            raw_name = document.get("documentType", "clinical-document")
            file_extension = _detect_extension(document.get("contentType"), url)
            safe_name = _sanitise_filename(f"{raw_name}{file_extension}")
        return DownloadedDocument(
            nct_id=nct_id,
            file_name=safe_name,
            content=content,
            metadata=document,
        )


__all__ = [
    "ClinicalTrialsAPIError",
    "ClinicalTrialsClient",
    "ClinicalTrialsNotFoundError",
    "DownloadedDocument",
    "build_provided_docs_url",
]
