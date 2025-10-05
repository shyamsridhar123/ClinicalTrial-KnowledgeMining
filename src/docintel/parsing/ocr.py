"""OCR integration used when Granite Docling requires text hints."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..config import ParsingSettings


class OcrError(RuntimeError):
    """Raised when OCR sampling fails fatally."""


class OcrEngine:
    """Wrapper around the Tesseract binary for OCR fallbacks."""

    def __init__(self, settings: ParsingSettings, *, logger: Optional[logging.Logger] = None) -> None:
        self._settings = settings
        self._logger = logger or logging.getLogger(__name__)

    async def extract_text(self, document_path: Path) -> Optional[str]:
        """Return OCR text or ``None`` when OCR is disabled/unavailable."""

        if not self._settings.ocr_enabled:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_tesseract, document_path)

    def _run_tesseract(self, document_path: Path) -> Optional[str]:
        if not document_path.exists():
            raise OcrError(f"document not found for OCR: {document_path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_base = Path(temp_dir) / "ocr_output"
            cmd = [
                "tesseract",
                str(document_path),
                str(output_base),
                "-l",
                self._settings.tesseract_langs,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except FileNotFoundError:
                self._logger.warning("Tesseract executable not found; skipping OCR")
                return None
            except subprocess.CalledProcessError as exc:
                self._logger.warning("Tesseract failed with code %s", exc.returncode)
                return None
            text_path = output_base.with_suffix(".txt")
            if not text_path.exists():
                return None
            text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
            return text or None


__all__ = ["OcrEngine", "OcrError"]
