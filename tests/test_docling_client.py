from __future__ import annotations

from pathlib import Path

import fitz  # type: ignore[import-not-found]
import pytest  # type: ignore[import-not-found]

import docintel.parsing.client as client_module
from docintel.config import ParsingSettings
from docintel.parsing.client import DoclingClient


class _ExplodingConverter:
    def __init__(self, message: str) -> None:
        self._message = message

    def convert(self, *_args, **_kwargs):  # pragma: no cover - behaviour validated by fallback
        raise RuntimeError(self._message)


@pytest.mark.asyncio
async def test_docling_client_pymupdf_fallback(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "fallback.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Hello clinical trial world")
        pdf.save(pdf_path)

    monkeypatch.setattr(client_module, "_DOCLING_AVAILABLE", True)
    monkeypatch.setattr(
        DoclingClient,
        "_create_standalone_converter",
        lambda self, table_structure=True: _ExplodingConverter("basic_string::at failure"),
    )

    client = DoclingClient(ParsingSettings())

    result = await client.parse_document(document_path=pdf_path)

    assert result.metadata["status"] == "pymupdf_fallback"
    assert "Hello clinical trial world" in result.plain_text
    assert not result.tables


@pytest.mark.asyncio
async def test_docling_client_returns_pre_supplied_ocr_text(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "ocr.pdf"
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(pdf_path)

    monkeypatch.setattr(client_module, "_DOCLING_AVAILABLE", True)
    monkeypatch.setattr(
        DoclingClient,
        "_create_standalone_converter",
        lambda self, table_structure=True: _ExplodingConverter("should not be invoked"),
    )

    client = DoclingClient(ParsingSettings())

    result = await client.parse_document(document_path=pdf_path, ocr_text="Recovered text from OCR")

    assert result.metadata["status"] == "ocr_fallback"
    assert result.plain_text == "Recovered text from OCR"
    assert result.tables == []
