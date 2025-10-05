"""Helpers for writing structured parsing artefacts to disk."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import shutil
from binascii import Error as BinasciiError
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .client import DoclingParseResult
from ..storage import ProcessingLayout


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ArtifactPaths:
    """Locations of the artefacts generated for a parsed document."""

    structured: Path
    markdown: Path
    html: Path
    text: Path
    tables: Path
    figures_metadata: Path
    figure_images_dir: Path
    chunks: Path
    provenance: Path

    def as_iterable(self) -> Iterable[Path]:
        return (
            self.structured,
            self.markdown,
            self.html,
            self.text,
            self.tables,
            self.figures_metadata,
            self.chunks,
            self.provenance,
        )


class ArtifactWriter:
    """Persist parsed outputs into the processing storage layout."""

    def __init__(self, layout: ProcessingLayout) -> None:
        self._layout = layout

    def _relative_stub(self, nct_id: str, document_name: str) -> Path:
        stem = Path(document_name).stem or "document"
        return Path(nct_id) / stem

    def _build_paths(self, nct_id: str, document_name: str) -> ArtifactPaths:
        stub = self._relative_stub(nct_id, document_name)
        return ArtifactPaths(
            structured=(self._layout.structured / stub).with_suffix(".json"),
            markdown=(self._layout.markdown / stub).with_suffix(".md"),
            html=(self._layout.html / stub).with_suffix(".html"),
            text=(self._layout.text / stub).with_suffix(".txt"),
            tables=(self._layout.tables / stub).with_suffix(".json"),
            figures_metadata=(self._layout.figures / stub).with_suffix(".json"),
            figure_images_dir=self._layout.figures / stub,
            chunks=(self._layout.chunks / stub).with_suffix(".json"),
            provenance=(self._layout.provenance / stub).with_suffix(".json"),
        )

    def exists(self, nct_id: str, document_name: str) -> bool:
        """Return ``True`` when the structured artefact already exists on disk."""

        paths = self._build_paths(nct_id, document_name)
        return paths.structured.exists()

    def write(
        self,
        nct_id: str,
        document_name: str,
        result: DoclingParseResult,
        provenance: Dict[str, object],
    ) -> ArtifactPaths:
        """Write the artefacts to disk and return their paths."""

        paths = self._build_paths(nct_id, document_name)
        for path in paths.as_iterable():
            path.parent.mkdir(parents=True, exist_ok=True)
        paths.figure_images_dir.mkdir(parents=True, exist_ok=True)
        self._clear_directory(paths.figure_images_dir)

        paths.structured.write_text(
            json.dumps(result.document, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths.markdown.write_text(result.markdown, encoding="utf-8")
        paths.html.write_text(result.html, encoding="utf-8")
        paths.text.write_text(result.plain_text, encoding="utf-8")
        paths.tables.write_text(
            json.dumps(result.tables, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        figures_payload = self._write_figures(
            paths.figure_images_dir,
            result.figures,
            nct_id,
            Path(document_name).stem or "document",
        )
        paths.figures_metadata.write_text(
            json.dumps(figures_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths.chunks.write_text(
            json.dumps(result.chunks, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths.provenance.write_text(
            json.dumps(provenance, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return paths

    def _clear_directory(self, directory: Path) -> None:
        if not directory.exists():
            return
        for entry in directory.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except OSError as exc:  # pragma: no cover - defensive path
                _LOGGER.warning("Failed clearing figure directory | path=%s | error=%s", entry, exc)

    def _write_figures(
        self,
        figure_dir: Path,
        figures: List[Dict[str, object]],
        nct_id: str,
        document_stem: str,
    ) -> List[Dict[str, object]]:
        if not figures:
            return []

        payload: List[Dict[str, object]] = []
        base_rel_dir = figure_dir.relative_to(self._layout.figures)
        for index, figure in enumerate(figures, start=1):
            image_relpath: Optional[Path] = None
            mime_type = self._safe_str(figure.get("image_mime_type"))
            image_data = self._safe_str(figure.get("image_data"))
            image_uri = self._safe_str(figure.get("image_uri"))

            if image_data:
                extension = self._guess_extension(mime_type)
                image_name = f"{document_stem}_figure_{index:02d}{extension}"
                output_path = figure_dir / image_name
                try:
                    output_path.write_bytes(base64.b64decode(image_data, validate=True))
                    image_relpath = base_rel_dir / image_name
                except (BinasciiError, ValueError) as exc:  # pragma: no cover - defensive path
                    _LOGGER.warning(
                        "Failed to decode figure image | nct_id=%s | document=%s | figure_index=%d | error=%s",
                        nct_id,
                        document_stem,
                        index,
                        exc,
                    )
            record = {
                "id": figure.get("id"),
                "caption": figure.get("caption"),
                "image_path": str(Path("figures") / image_relpath) if image_relpath else None,
                "image_mime_type": mime_type,
                "image_uri": None if image_data else image_uri,
                "provenance": figure.get("provenance", []),
            }
            payload.append(record)
        return payload

    def _safe_str(self, value: object) -> Optional[str]:
        if isinstance(value, str):
            return value
        return None

    def _guess_extension(self, mime_type: Optional[str]) -> str:
        if mime_type:
            extension = mimetypes.guess_extension(mime_type, strict=False)
            if extension:
                return extension
        return ".png"


__all__ = [
    "ArtifactPaths",
    "ArtifactWriter",
]
