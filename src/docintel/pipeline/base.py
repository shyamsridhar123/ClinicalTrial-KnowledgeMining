"""Common abstractions for composing docintel processing phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..config import DataCollectionSettings, EmbeddingSettings, ParsingSettings
from ..storage import EmbeddingLayout, ProcessingLayout, StorageLayout


@dataclass(slots=True)
class PipelineContext:
    """Runtime context shared across pipeline phases."""

    ingestion_settings: Optional[DataCollectionSettings] = None
    parsing_settings: Optional[ParsingSettings] = None
    embedding_settings: Optional[EmbeddingSettings] = None
    storage_layout: Optional[StorageLayout] = None
    processing_layout: Optional[ProcessingLayout] = None
    embedding_layout: Optional[EmbeddingLayout] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PhaseResult:
    """Represents the outcome of a pipeline phase."""

    name: str
    succeeded: bool
    details: Dict[str, Any] = field(default_factory=dict)


class PipelinePhase(Protocol):
    """Interface that all pipeline phases must implement."""

    name: str

    async def run(self, context: PipelineContext) -> PhaseResult:
        ...


class PipelineRunner:
    """Execute a sequence of pipeline phases with shared context."""

    def __init__(self, phases: Sequence[PipelinePhase], context: PipelineContext):
        self._phases = list(phases)
        self._context = context
        self._results: List[PhaseResult] = []

    @property
    def results(self) -> List[PhaseResult]:
        return list(self._results)

    async def run(self) -> List[PhaseResult]:
        self._results.clear()
        for phase in self._phases:
            result = await phase.run(self._context)
            self._results.append(result)
            if not result.succeeded:
                break
        return list(self._results)


__all__ = [
    "PhaseResult",
    "PipelineContext",
    "PipelinePhase",
    "PipelineRunner",
]
