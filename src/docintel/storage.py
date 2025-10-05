"""Filesystem helpers for ingestion and processing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import DataCollectionSettings, EmbeddingSettings, ParsingSettings


@dataclass(frozen=True, slots=True)
class StorageLayout:
    """Represents the directory layout for ingestion artifacts."""

    root: Path
    documents: Path
    metadata: Path
    logs: Path
    temp: Path

    def as_iterable(self) -> Iterable[Path]:
        return (self.root, self.documents, self.metadata, self.logs, self.temp)


def build_storage_layout(settings: DataCollectionSettings) -> StorageLayout:
    """Create the directory layout based on provided settings."""

    directories = settings.storage_directories()
    return StorageLayout(
        root=directories["root"],
        documents=directories["documents"],
        metadata=directories["metadata"],
        logs=directories["logs"],
        temp=directories["temp"],
    )


def ensure_storage_layout(layout: StorageLayout) -> None:
    """Ensure all directories in the layout exist on disk."""

    for path in layout.as_iterable():
        path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class ProcessingLayout:
    """Represents the directory layout for parsing artifacts."""

    root: Path
    structured: Path
    markdown: Path
    html: Path
    tables: Path
    figures: Path
    text: Path
    chunks: Path
    provenance: Path
    logs: Path
    ocr: Path
    temp: Path

    def as_iterable(self) -> Iterable[Path]:
        return (
            self.root,
            self.structured,
            self.markdown,
            self.html,
            self.tables,
            self.figures,
            self.text,
            self.chunks,
            self.provenance,
            self.logs,
            self.ocr,
            self.temp,
        )


def build_processing_layout(settings: ParsingSettings) -> ProcessingLayout:
    """Create the directory layout for parsing artifacts."""

    directories = settings.processing_directories()
    return ProcessingLayout(
        root=directories["root"],
        structured=directories["structured"],
        markdown=directories["markdown"],
        html=directories["html"],
        tables=directories["tables"],
        figures=directories["figures"],
        text=directories["text"],
        chunks=directories["chunks"],
        provenance=directories["provenance"],
        logs=directories["logs"],
        ocr=directories["ocr"],
        temp=directories["temp"],
    )


def ensure_processing_layout(layout: ProcessingLayout) -> None:
    """Ensure all directories relevant to parsing exist on disk."""

    for path in layout.as_iterable():
        path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class EmbeddingLayout:
    """Filesystem layout for persisted embeddings and related logs."""

    root: Path
    vectors: Path
    logs: Path
    temp: Path

    def as_iterable(self) -> Iterable[Path]:
        return (self.root, self.vectors, self.logs, self.temp)


def build_embedding_layout(settings: EmbeddingSettings) -> EmbeddingLayout:
    """Build the embedding storage layout from settings."""

    directories = settings.embedding_directories()
    return EmbeddingLayout(
        root=directories["root"],
        vectors=directories["vectors"],
        logs=directories["logs"],
        temp=directories["temp"],
    )


def ensure_embedding_layout(layout: EmbeddingLayout) -> None:
    """Ensure embedding directories are present on disk."""

    for path in layout.as_iterable():
        path.mkdir(parents=True, exist_ok=True)
