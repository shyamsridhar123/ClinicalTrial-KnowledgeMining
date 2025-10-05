"""Pipeline phase that executes the parsing orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import get_parsing_settings
from ..pipeline import PhaseResult, PipelineContext, PipelinePhase
from ..storage import build_processing_layout, ensure_processing_layout
from .artifacts import ArtifactWriter
from .client import DoclingClient
from .ocr import OcrEngine
from .orchestrator import ParsingOrchestrator


@dataclass(slots=True)
class ParsingPhase(PipelinePhase):
    """Execute Docling parsing across downloaded documents."""

    force_reparse: bool = False
    max_workers_override: Optional[int] = None
    name: str = "parsing"

    async def run(self, context: PipelineContext) -> PhaseResult:
        parsing_settings = context.parsing_settings or get_parsing_settings()
        context.parsing_settings = parsing_settings
        ingestion_layout = context.storage_layout
        if ingestion_layout is None:
            raise ValueError("storage_layout missing from pipeline context")
        processing_layout = context.processing_layout or build_processing_layout(parsing_settings)
        ensure_processing_layout(processing_layout)
        context.processing_layout = processing_layout

        docling_client = DoclingClient(parsing_settings)
        ocr_engine = OcrEngine(parsing_settings)
        artifact_writer = ArtifactWriter(processing_layout)
        orchestrator = ParsingOrchestrator(
            parsing_settings,
            ingestion_layout,
            processing_layout,
            docling_client,
            ocr_engine,
            artifact_writer,
        )
        report = await orchestrator.run(
            force_reparse=self.force_reparse,
            max_workers=self.max_workers_override,
        )
        context.extra.setdefault("parsing", {})["report"] = report
        return PhaseResult(name=self.name, succeeded=True, details={"report": report})


__all__ = ["ParsingPhase"]
