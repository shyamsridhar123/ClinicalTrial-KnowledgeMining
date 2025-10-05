"""Pipeline phase for the ingestion stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..pipeline.base import PhaseResult, PipelineContext, PipelinePhase
from .collector import ClinicalTrialsCollector


@dataclass(slots=True)
class IngestionPhase(PipelinePhase):
    """Execute the ingestion collector as part of a multi-phase pipeline."""

    max_studies: int
    name: str = "ingestion"

    async def run(self, context: PipelineContext) -> PhaseResult:
        if not context.ingestion_settings:
            raise ValueError("ingestion_settings missing from pipeline context")
        if not context.storage_layout:
            raise ValueError("storage_layout missing from pipeline context")
        collector = ClinicalTrialsCollector(context.ingestion_settings, context.storage_layout)
        report = await collector.run(max_studies=self.max_studies)
        context.extra.setdefault("ingestion", {})["report"] = report
        return PhaseResult(name=self.name, succeeded=True, details={"report": report})


__all__ = ["IngestionPhase"]
