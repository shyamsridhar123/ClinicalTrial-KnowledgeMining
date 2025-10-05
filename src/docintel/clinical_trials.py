"""Compatibility shim exposing ingestion utilities.

The ingestion logic now lives under :mod:`docintel.ingestion`. This module keeps
public imports stable for existing code while new phases integrate via the
pipeline abstractions.
"""

from __future__ import annotations

from .ingestion import (
    ClinicalTrialsAPIError,
    ClinicalTrialsClient,
    ClinicalTrialsCollector,
    ClinicalTrialsNotFoundError,
    DownloadedDocument,
    build_provided_docs_url,
)

__all__ = [
    "ClinicalTrialsAPIError",
    "ClinicalTrialsClient",
    "ClinicalTrialsCollector",
    "ClinicalTrialsNotFoundError",
    "DownloadedDocument",
    "build_provided_docs_url",
]
