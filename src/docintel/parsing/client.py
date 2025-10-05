"""Docling parsing client backed by Modular MAX and the Docling SDK.

This implementation follows the Modular MAX guidance for serving Granite
Docling via an OpenAI-compatible endpoint (see
https://docs.modular.com/max/get-started/) and delegates parsing to the
Docling SDK's :class:`~docling.document_converter.DocumentConverter`. The
converter handles PDF-to-image rendering and multimodal requests, ensuring we
comply with Granite Docling's expectation of page images instead of raw PDF
bytes (see https://docs.modular.com/max/models/granite-docling/).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import base64
import json
from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from httpx import AsyncClient, HTTPStatusError, RequestError, Timeout
from openai import AsyncOpenAI

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
    from docling.exceptions import ConversionError
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling_core.types.doc import DoclingDocument, ImageRefMode, PictureItem, TableItem

DocumentConverter = PdfFormatOption = ConversionStatus = InputFormat = None  # type: ignore[assignment]
VlmPipelineOptions = ApiVlmOptions = ResponseFormat = None  # type: ignore[assignment]
ConversionError = VlmPipeline = None  # type: ignore[assignment]
DoclingDocument = ImageRefMode = PictureItem = TableItem = None  # type: ignore[assignment]

try:  # Optional runtime dependency: Docling SDK
    _docling_document_converter = import_module("docling.document_converter")
    _docling_datamodel_base = import_module("docling.datamodel.base_models")
    _docling_pipeline_options = import_module("docling.datamodel.pipeline_options")
    _docling_pipeline_options_vlm = import_module("docling.datamodel.pipeline_options_vlm_model")
    _docling_exceptions = import_module("docling.exceptions")
    _docling_pipeline_vlm = import_module("docling.pipeline.vlm_pipeline")
    _docling_core_doc = import_module("docling_core.types.doc")

    DocumentConverter = getattr(_docling_document_converter, "DocumentConverter")  # type: ignore[assignment]
    PdfFormatOption = getattr(_docling_document_converter, "PdfFormatOption")  # type: ignore[assignment]
    ConversionStatus = getattr(_docling_datamodel_base, "ConversionStatus")  # type: ignore[assignment]
    InputFormat = getattr(_docling_datamodel_base, "InputFormat")  # type: ignore[assignment]
    VlmPipelineOptions = getattr(_docling_pipeline_options, "VlmPipelineOptions")  # type: ignore[assignment]
    ApiVlmOptions = getattr(_docling_pipeline_options_vlm, "ApiVlmOptions")  # type: ignore[assignment]
    ResponseFormat = getattr(_docling_pipeline_options_vlm, "ResponseFormat")  # type: ignore[assignment]
    ConversionError = getattr(_docling_exceptions, "ConversionError")  # type: ignore[assignment]
    VlmPipeline = getattr(_docling_pipeline_vlm, "VlmPipeline")  # type: ignore[assignment]
    DoclingDocument = getattr(_docling_core_doc, "DoclingDocument")  # type: ignore[assignment]
    ImageRefMode = getattr(_docling_core_doc, "ImageRefMode")  # type: ignore[assignment]
    PictureItem = getattr(_docling_core_doc, "PictureItem")  # type: ignore[assignment]
    TableItem = getattr(_docling_core_doc, "TableItem")  # type: ignore[assignment]

    _DOCLING_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised when SDK is absent in tests
    _DOCLING_AVAILABLE = False

from ..config import ParsingSettings

_OCR_HTML_TEMPLATE = "<p>{text}</p>"

_LOGGER = logging.getLogger(__name__)


class DoclingClientError(RuntimeError):
    """Raised when the Docling MAX endpoint returns an error or invalid output."""


@dataclass(slots=True)
class DoclingParseResult:
    """Structured outputs returned by the Docling parsing stage."""

    document: Dict[str, Any]
    markdown: str
    html: str
    plain_text: str
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DoclingClient:
    """Asynchronous client for Granite Docling served through Modular MAX."""

    def __init__(
        self,
        settings: ParsingSettings,
        *,
        client: Optional[AsyncOpenAI] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._settings = settings
        request_timeout = float(settings.docling_request_timeout_seconds)
        self._timeout = Timeout(
            request_timeout,
            connect=min(30.0, request_timeout),
            read=request_timeout,
            write=request_timeout,
        )
        self._client = client or AsyncOpenAI(
            base_url=str(settings.docling_max_base_url),
            api_key=settings.docling_api_key,
            timeout=self._timeout,
        )
        self._logger = logger or _LOGGER
        # Always use the fast GPU-accelerated standalone converter
        # VLM pipeline with MAX is incompatible (image processing causes 40+ second delays)
        self._use_modular_max = False
        self._converter = None
        self._fallback_converter = None
        if not _DOCLING_AVAILABLE:
            raise DoclingClientError(
                "Docling SDK is required. Install the 'docling' package inside the Pixi environment."
            )
        self._converter = self._create_standalone_converter()

    async def health_check(self) -> bool:
        """Perform a lightweight request to verify that the MAX endpoint is reachable."""

        try:
            base_url = str(self._settings.docling_max_base_url).rstrip("/")
            models_url = f"{base_url}/models"
            timeout = Timeout(10.0, connect=5.0)
            async with AsyncClient(timeout=timeout) as client:
                response = await client.get(models_url)
                response.raise_for_status()
        except (RequestError, HTTPStatusError) as exc:  # pragma: no cover - network failures
            self._logger.error("Docling health check failed: %s", exc)
            return False

        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - invalid JSON response
            self._logger.error("Docling health check returned invalid JSON: %s", exc)
            return False

        models = payload.get("data") if isinstance(payload, dict) else None
        if not models:
            self._logger.error("Docling health check returned no models: %s", payload)
            return False
        expected_model = self._settings.docling_model_name
        if not any(item.get("id") == expected_model for item in models if isinstance(item, dict)):
            self._logger.error(
                "Docling health check missing model | expected=%s | payload=%s",
                expected_model,
                payload,
            )
            return False
        return True

    async def parse_document(
        self,
        *,
        document_path: Path,
        ocr_text: Optional[str] = None,
    ) -> DoclingParseResult:
        """Parse a document and return the structured Docling artefacts."""

        if not document_path.exists():
            raise DoclingClientError(f"Document not found: {document_path}")

        if ocr_text is not None:
            self._logger.info(
                "docling | returning pre-supplied OCR text | path=%s",
                document_path,
            )
            return self._build_text_fallback(
                document_path=document_path,
                text=ocr_text,
                source="ocr_fallback",
            )

        # Use fast standalone SDK (optimal performance)
        self._logger.info(
            "docling | using fast standalone SDK | path=%s",
            document_path,
        )
        return await self._parse_with_docling_sdk(document_path=document_path)

    async def _parse_with_docling_sdk(self, *, document_path: Path) -> DoclingParseResult:
        if not self._converter:
            raise DoclingClientError("Docling converter is not initialised; cannot parse document.")

        # Memory optimization: clear GPU cache before processing
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        request_started = perf_counter()
        fallback_used: Optional[str] = None
        try:
            conversion = await asyncio.to_thread(self._converter.convert, document_path)
        except Exception as exc:  # pragma: no cover - conversion failure path
            duration = perf_counter() - request_started
            if self._should_retry_without_tables(exc):
                fallback_used = "disable_table_structure"
                self._logger.warning(
                    "docling | retrying without table structure | path=%s | duration_s=%.2f | error=%s",
                    document_path,
                    duration,
                    exc,
                )
                if self._fallback_converter is None:
                    self._fallback_converter = self._create_standalone_converter(table_structure=False)
                try:
                    conversion = await asyncio.to_thread(self._fallback_converter.convert, document_path)
                except Exception as retry_exc:  # pragma: no cover - retry failure path
                    retry_duration = perf_counter() - request_started
                    self._logger.error(
                        "docling | fallback conversion error | path=%s | duration_s=%.2f | error=%s",
                        document_path,
                        retry_duration,
                        retry_exc,
                    )
                    text_fallback = self._extract_text_with_pymupdf(document_path)
                    if text_fallback:
                        self._logger.warning(
                            "docling | using PyMuPDF text fallback after docling failure | path=%s",
                            document_path,
                        )
                        return self._build_text_fallback(
                            document_path=document_path,
                            text=text_fallback,
                            source="pymupdf_fallback",
                        )
                    cause = DoclingClientError(f"Docling conversion failed for {document_path}")
                    raise cause from retry_exc
            else:
                self._logger.error(
                    "docling | conversion error | path=%s | duration_s=%.2f | error=%s",
                    document_path,
                    duration,
                    exc,
                )
                text_fallback = self._extract_text_with_pymupdf(document_path)
                if text_fallback:
                    self._logger.warning(
                        "docling | using PyMuPDF text fallback after docling failure | path=%s",
                        document_path,
                    )
                    return self._build_text_fallback(
                        document_path=document_path,
                        text=text_fallback,
                        source="pymupdf_fallback",
                    )
                cause = DoclingClientError(f"Docling conversion failed for {document_path}")
                raise cause from exc

        duration = perf_counter() - request_started
        if ConversionStatus and conversion.status not in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
            self._logger.error(
                "docling | conversion status not successful | path=%s | status=%s | errors=%s",
                document_path,
                conversion.status,
                [error.error_message for error in conversion.errors],
            )
            raise DoclingClientError(f"Docling conversion returned {conversion.status}")

        document = conversion.document
        plain_text = self._safe_export_text(document)
        markdown = self._safe_export_markdown(document)
        html = self._safe_export_html(document)
        tables = [self._serialize_table(table, document) for table in getattr(document, "tables", [])]
        figures = [self._serialize_figure(picture, document) for picture in getattr(document, "pictures", [])]
        metadata = self._build_metadata(conversion, fallback=fallback_used)

        # Memory optimization: cleanup after processing
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._logger.info(
            "docling | HIGH-PERFORMANCE conversion completed | path=%s | duration_s=%.2f | pages=%d | markdown_chars=%d | fallback=%s",
            document_path,
            duration,
            document.num_pages() if hasattr(document, "num_pages") else len(getattr(conversion, "pages", [])),
            len(markdown),
            fallback_used or "none"
        )

        return DoclingParseResult(
            document=document.export_to_dict() if hasattr(document, "export_to_dict") else {},
            markdown=markdown,
            html=html,
            plain_text=plain_text,
            tables=tables,
            figures=figures,
            chunks=[],
            metadata=metadata,
        )






    def _create_standalone_converter(self, *, table_structure: bool = True) -> Any:
        """Create a standalone Docling converter optimized for maximum GPU acceleration."""
        import torch
        import os
        
        # Enable TensorFloat-32 (TF32) for Ampere GPUs - 2-3x speedup
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Optimize CUDA memory allocation for better reuse
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
                'max_split_size_mb:512,'
                'garbage_collection_threshold:0.8,'
                'expandable_segments:True'
            )
            
            self._logger.info(
                "docling | GPU acceleration enabled: TF32=%s, CUDA memory optimization=%s",
                torch.backends.cuda.matmul.allow_tf32,
                'max_split_size_mb:512'
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._logger.info(
            "docling | using GPU-optimized pipeline with accelerated processing | device=%s | table_structure=%s",
            device,
            table_structure,
        )
        
        # Create optimized pipeline options for maximum GPU utilization
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, 
            EasyOcrOptions,
            TesseractOcrOptions,
        )
        
        # Configure pipeline for maximum GPU acceleration and memory efficiency
        pipeline_options = PdfPipelineOptions()

        # Disable slow OCR operations that are CPU-bound
        pipeline_options.do_ocr = False  # Skip OCR unless absolutely necessary
        pipeline_options.do_table_structure = table_structure  # Keep GPU-accelerated table detection when enabled

        # Aggressive performance optimizations - only use valid options
        if hasattr(pipeline_options, "images_scale"):
            pipeline_options.images_scale = float(self._settings.docling_images_scale)

        # Try to set advanced options if they exist in this Docling version
        for opt_name, opt_value in [
            ("generate_page_images", self._settings.docling_generate_page_images),
            ("generate_picture_images", self._settings.docling_generate_picture_images),
            ("generate_table_images", self._settings.docling_generate_table_images),
            ("generate_thumbnails", False),
            ("accelerator_device", "cuda" if torch.cuda.is_available() else "cpu"),
        ]:
            try:
                if hasattr(pipeline_options, opt_name):
                    setattr(pipeline_options, opt_name, opt_value)
                    self._logger.debug("Set %s = %s", opt_name, opt_value)
            except (AttributeError, ValueError):
                self._logger.debug("Option %s not available or invalid", opt_name)
                continue
        
        # Use GPU-optimized format options with memory efficiency
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
        
        # Configure converter with optimized format options
        converter = DocumentConverter(format_options=format_options)
        
        # Log optimization status
        gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
        tf32_status = "enabled" if torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32 else "disabled"
        
        if table_structure:
            self._logger.info(
                "docling | HIGH-PERFORMANCE CONVERTER: %s processing, TF32 %s, OCR disabled, thumbnails disabled, memory optimized",
                gpu_status, tf32_status
            )
        else:
            self._logger.info(
                "docling | FALLBACK CONVERTER: %s processing, table structure disabled, optimized for stability",
                gpu_status
            )
        return converter

    def _should_retry_without_tables(self, exc: Exception) -> bool:
        message = str(exc)
        retriable = "basic_string::at" in message or "parse_pdf_from_key_on_page" in message
        if retriable:
            self._logger.warning("docling | detected docling core IndexError - attempting fallback without tables")
        return retriable





    def _serialize_table(self, table: Any, document: Any) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of a Docling table."""

        return {
            "id": getattr(table, "self_ref", None),
            "caption": table.caption_text(document) if hasattr(table, "caption_text") else "",
            "markdown": table.export_to_markdown(doc=document) if hasattr(table, "export_to_markdown") else "",
            "html": table.export_to_html(doc=document) if hasattr(table, "export_to_html") else "",
        }

    def _serialize_figure(self, picture: Any, document: Any) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of a Docling figure/picture."""

        image_ref = getattr(picture, "image", None)
        image_uri = getattr(image_ref, "uri", None) if image_ref else None
        mime_type = getattr(image_ref, "mime_type", None) if image_ref else None
        parsed_uri = str(image_uri) if image_uri is not None else None
        image_data: Optional[str] = None

        if isinstance(parsed_uri, str) and parsed_uri.startswith("data:"):
            header, _, encoded = parsed_uri.partition(",")
            if not mime_type:
                mime_type = self._extract_mime_type_from_header(header)
            image_data = encoded or None
        elif image_ref is not None:
            data_attr = getattr(image_ref, "data_as_base64", None)
            if callable(data_attr):
                image_data = data_attr()

        provenance = [self._serialize_provenance(item) for item in getattr(picture, "prov", []) or []]

        return {
            "id": getattr(picture, "self_ref", None),
            "caption": picture.caption_text(document) if hasattr(picture, "caption_text") else "",
            "image_uri": parsed_uri,
            "image_mime_type": mime_type,
            "image_data": image_data,
            "provenance": provenance,
        }

    def _serialize_provenance(self, item: Any) -> Dict[str, Any]:
        if hasattr(item, "model_dump"):
            data = item.model_dump()
        elif hasattr(item, "__dict__"):
            data = dict(item.__dict__)
        else:
            return item  # type: ignore[return-value]

        bbox = data.get("bbox")
        if hasattr(bbox, "model_dump"):
            bbox = bbox.model_dump()
        if isinstance(bbox, dict):
            coord_origin = bbox.get("coord_origin")
            if hasattr(coord_origin, "value"):
                bbox["coord_origin"] = coord_origin.value
        if bbox is not None:
            data["bbox"] = bbox

        coord_origin = data.get("coord_origin")
        if hasattr(coord_origin, "value"):
            data["coord_origin"] = coord_origin.value

        charspan = data.get("charspan")
        if isinstance(charspan, tuple):
            data["charspan"] = list(charspan)
        return data

    def _extract_mime_type_from_header(self, header: str) -> Optional[str]:
        if not header.startswith("data:"):
            return None
        meta = header[5:]
        if ";" in meta:
            return meta.split(";", 1)[0] or None
        return meta or None

    def _safe_export_markdown(self, document: Any) -> str:
        try:
            return document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
        except Exception as exc:  # pragma: no cover - defensive path
            self._logger.warning("docling | markdown export failed: %s", exc)
            return ""

    def _safe_export_html(self, document: Any) -> str:
        try:
            return document.export_to_html(image_mode=ImageRefMode.EMBEDDED)
        except Exception as exc:  # pragma: no cover - defensive path
            self._logger.warning("docling | html export failed: %s", exc)
            return ""

    def _safe_export_text(self, document: Any) -> str:
        try:
            return document.export_to_text()
        except Exception as exc:  # pragma: no cover - defensive path
            self._logger.warning("docling | text export failed: %s", exc)
            return ""

    def _build_metadata(self, conversion, *, fallback: Optional[str] = None) -> Dict[str, Any]:
        metadata = {
            "status": conversion.status.value,
            "errors": [error.model_dump() for error in conversion.errors],
            "confidences": conversion.confidence.model_dump() if hasattr(conversion, "confidence") else {},
            "timings": {
                name: item.model_dump()
                for name, item in getattr(conversion, "timings", {}).items()
            },
            "page_count": len(conversion.pages),
            "document_hash": getattr(conversion.input, "document_hash", None),
            "input_file": str(getattr(conversion.input, "file", "")),
        }
        if fallback:
            metadata["fallback"] = fallback
        return metadata

    def _build_text_fallback(self, *, document_path: Path, text: str, source: str) -> DoclingParseResult:
        plain_text = text
        html = _OCR_HTML_TEMPLATE.format(text=plain_text.replace("\n", "<br/>"))
        metadata = {
            "status": source,
            "source": str(document_path),
            "character_count": len(text),
        }
        return DoclingParseResult(
            document={
                "name": document_path.name,
                "source": source,
                "plain_text": plain_text,
            },
            markdown=plain_text,
            html=html,
            plain_text=plain_text,
            tables=[],
            figures=[],
            chunks=[],
            metadata=metadata,
        )

    def _extract_text_with_pymupdf(self, document_path: Path) -> Optional[str]:
        try:
            import fitz  # type: ignore[import-not-found]
        except ModuleNotFoundError:  # pragma: no cover - dependency missing in tests
            self._logger.debug("docling | pymupdf not installed - skipping text fallback")
            return None

        try:
            with fitz.open(document_path) as pdf:
                text_segments = [page.get_text("text") for page in pdf]
        except Exception as exc:  # pragma: no cover - PyMuPDF runtime failure
            self._logger.warning(
                "docling | pymupdf text extraction failed | path=%s | error=%s",
                document_path,
                exc,
            )
            return None

        combined = "\n".join(segment.strip() for segment in text_segments if segment)
        return combined.strip() or None


__all__ = [
    "DoclingClient",
    "DoclingClientError",
    "DoclingParseResult",
]
