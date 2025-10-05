"""Configuration utilities for the Clinical Trial Knowledge Mining ingestion pipeline."""

from __future__ import annotations

import os

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import (
    AnyHttpUrl,
    Field,
    FieldValidationInfo,
    PositiveFloat,
    PositiveInt,
    PostgresDsn,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataCollectionSettings(BaseSettings):
    """Settings model driven by environment variables.

    Environment variables are prefixed with ``DOCINTEL_``. For example, set
    ``DOCINTEL_STORAGE_ROOT=/var/lib/docintel`` to override the storage root directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    clinicaltrials_api_base: AnyHttpUrl = Field(
        default="https://clinicaltrials.gov/api/v2",
        description="Base URL for ClinicalTrials.gov API v2 endpoints.",
    )
    request_timeout_seconds: PositiveFloat = Field(
        default=30.0,
        description="Total request timeout applied to ClinicalTrials.gov API calls.",
    )
    retry_attempts: PositiveInt = Field(
        default=3,
        description="Number of retries for recoverable HTTP failures.",
    )
    max_concurrent_downloads: PositiveInt = Field(
        default=5,
        description="Maximum number of concurrent document downloads.",
    )
    storage_root: Path = Field(
        default_factory=lambda: Path("data") / "ingestion",
        description="Root directory for downloaded documents, metadata, and logs.",
    )
    target_therapeutic_areas: List[str] = Field(
        default_factory=list,
        description="Therapeutic areas to prioritise when searching for studies.",
    )
    target_phases: List[str] = Field(
        default_factory=list,
        description="Clinical trial phases to include in metadata collection.",
    )
    study_status: str = Field(
        default="COMPLETED",
        description="Clinical trial recruitment status filter (e.g., COMPLETED, RECRUITING).",
    )
    search_query_term: Optional[str] = Field(
        default=None,
        description="Optional custom ClinicalTrials.gov query term expression to seed study search.",
    )
    search_overfetch_multiplier: PositiveInt = Field(
        default=4,
        description="Multiplier applied to max_studies to determine the number of studies requested per search (capped at 100).",
    )

    def storage_directories(self) -> Dict[str, Path]:
        """Return resolved storage directories keyed by purpose."""

        root = self.storage_root.expanduser().resolve()
        directories = {
            "root": root,
            "documents": root / "pdfs",
            "metadata": root / "metadata",
            "logs": root / "logs",
            "temp": root / "temp",
        }
        return directories

    def ensure_storage(self) -> None:
        """Create required directories for the ingestion pipeline."""

        for path in self.storage_directories().values():
            path.mkdir(parents=True, exist_ok=True)

    def desired_page_size(self, max_studies: int) -> int:
        """Return the number of studies to request per API call, over-fetching to discover documents."""

        if max_studies <= 0:
            return 0
        requested = max_studies * self.search_overfetch_multiplier
        return min(100, max(max_studies, requested))


@lru_cache(maxsize=1)
def get_settings() -> DataCollectionSettings:
    """Return a cached ``DataCollectionSettings`` instance.

    Directory creation is performed once per process to guarantee downstream
    components have the expected filesystem layout.
    """

    settings = DataCollectionSettings()
    settings.ensure_storage()
    return settings


class ParsingSettings(BaseSettings):
    """Settings for the Docling parsing pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    docling_max_base_url: AnyHttpUrl = Field(
        default="http://localhost:8000/v1",
        description="Base URL for the Modular MAX endpoint serving Granite Docling.",
    )
    docling_model_name: str = Field(
        default="ibm-granite/granite-docling-258M",
        description="Model identifier exposed by Modular MAX for document parsing.",
    )
    docling_api_key: str = Field(
        default="EMPTY",
        description="API key forwarded to the OpenAI-compatible client (usually 'EMPTY' for local MAX).",
    )
    docling_request_timeout_seconds: PositiveFloat = Field(
        default=600.0,
        description="Timeout applied to end-to-end Docling parsing requests (set higher for large documents).",
    )
    docling_retry_attempts: PositiveInt = Field(
        default=3,
        description="Maximum number of retries for recoverable Docling failures.",
    )
    docling_structured_output_enabled: bool = Field(
        default=False,
        description=(
            "Enable MAX structured output decoding. Granite Docling currently hangs when "
            "llguidance-constrained decoding is enforced; leave disabled unless Modular "
            "confirms support."
        ),
    )
    max_concurrent_parses: PositiveInt = Field(
        default=2,
        description="Number of documents processed concurrently by the parsing orchestrator.",
    )
    processed_storage_root: Path = Field(
        default_factory=lambda: Path("data") / "processing",
        description="Root directory for structured parsing artifacts.",
    )
    docling_generate_page_images: bool = Field(
        default=False,
        description=(
            "Generate rendered images for each page. Disable by default to conserve storage; "
            "enable when downstream consumers require page-level thumbnails."
        ),
    )
    docling_generate_picture_images: bool = Field(
        default=True,
        description=(
            "Generate rendered assets for detected figures/pictures and persist them alongside "
            "captions and provenance metadata."
        ),
    )
    docling_generate_table_images: bool = Field(
        default=False,
        description=(
            "Generate rendered table images. Leave disabled unless downstream workflows need "
            "table snapshots in addition to structured data."
        ),
    )
    docling_images_scale: PositiveFloat = Field(
        default=1.0,
        description=(
            "Scaling factor applied to Docling-rendered imagery. 1.0 preserves original "
            "resolution; lower values reduce file size."
        ),
    )
    chunk_token_size: PositiveInt = Field(
        default=1000,
        description="Approximate token count per chunk emitted for downstream embedding.",
    )
    chunk_overlap: PositiveInt = Field(
        default=200,
        description="Approximate token overlap between adjacent chunks.",
    )
    ocr_enabled: bool = Field(
        default=True,
        description="Toggle Tesseract OCR fallback for image-only or low-text documents.",
    )
    tesseract_langs: str = Field(
        default="eng",
        description="Space-delimited languages passed to Tesseract during OCR fallback.",
    )

    def processing_directories(self) -> Dict[str, Path]:
        """Return resolved directories for parsing artifacts."""

        root = self.processed_storage_root.expanduser().resolve()
        directories = {
            "root": root,
            "structured": root / "structured",
            "markdown": root / "markdown",
            "html": root / "html",
            "tables": root / "tables",
            "figures": root / "figures",
            "text": root / "text",
            "chunks": root / "chunks",
            "provenance": root / "provenance",
            "logs": root / "logs",
            "ocr": root / "ocr",
            "temp": root / "temp",
        }
        return directories

    def ensure_processing_storage(self) -> None:
        """Ensure directories backing the parsing pipeline exist."""

        for path in self.processing_directories().values():
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_parsing_settings() -> ParsingSettings:
    """Return cached parsing settings and create storage directories."""

    settings = ParsingSettings()
    settings.ensure_processing_storage()
    return settings


class EmbeddingSettings(BaseSettings):
    """Settings controlling the embedding generation pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    embedding_max_base_url: AnyHttpUrl = Field(
        default="http://localhost:8000/v1",
        description=(
            "OpenAI-compatible endpoint for Modular MAX embedding models. "
            "See https://docs.modular.com/max/get-started/ for deployment guidance."
        ),
    )
    embedding_model_name: str = Field(
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        description=(
            "BiomedCLIP multimodal embedding model for clinical text and image understanding. "
            "Combines PubMedBERT for text with Vision Transformer for images, specifically "
            "trained on biomedical data for clinical trial documents and figures."
        ),
    )
    embedding_api_key: str = Field(
        default="EMPTY",
        description="API key forwarded to the MAX embedding endpoint (typically 'EMPTY' for local deployments).",
    )
    embedding_request_timeout_seconds: PositiveFloat = Field(
        default=120.0,
        description="Total timeout applied to each embedding batch request.",
    )
    embedding_batch_size: PositiveInt = Field(
        default=32,
        description="Number of text chunks submitted to the embedding endpoint per request.",
    )
    embedding_max_tokens: PositiveInt = Field(
        default=256,
        description=(
            "Maximum token length per chunk submitted to the embedding endpoint. "
            "Set to 256 to match BiomedCLIP's optimal context window. "
            "Chunks exceeding this limit are automatically re-chunked before inference."
        ),
    )
    embedding_storage_root: Path = Field(
        default_factory=lambda: Path("data") / "processing" / "embeddings",
        description="Root directory for persisted embedding vectors and manifests.",
    )
    enable_fallback_encoder: bool = Field(
        default=True,
        description=(
            "Toggle deterministic fallback embeddings when MAX requests fail. "
            "The fallback keeps downstream tests deterministic while MAX is offline."
        ),
    )
    embedding_quantization_encoding: str = Field(
        default="none",
        description=(
            "Quantization encoding applied to persisted vectors. "
            "Supported options: 'none', 'bfloat16', 'int8'."
        ),
    )
    embedding_quantization_store_float32: bool = Field(
        default=True,
        description=(
            "Persist original float32 vectors alongside quantized payloads. "
            "Set to False to minimise storage when quantization is enabled."
        ),
    )

    @field_validator("embedding_quantization_encoding")
    @classmethod
    def _normalise_quantization_encoding(cls, value: str) -> str:
        normalized = (value or "none").strip().lower()
        if normalized not in {"none", "bfloat16", "int8"}:
            raise ValueError(
                "embedding_quantization_encoding must be one of {'none', 'bfloat16', 'int8'}"
            )
        return normalized

    def embedding_directories(self) -> Dict[str, Path]:
        """Return resolved directories for the embedding pipeline."""

        root = self.embedding_storage_root.expanduser().resolve()
        return {
            "root": root,
            "vectors": root / "vectors",
            "logs": root / "logs",
            "temp": root / "temp",
        }

    def ensure_embedding_storage(self) -> None:
        """Create the directories required for embedding persistence."""

        for path in self.embedding_directories().values():
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_embedding_settings() -> EmbeddingSettings:
    """Return cached embedding settings after provisioning storage directories."""

    settings = EmbeddingSettings()
    settings.ensure_embedding_storage()
    return settings


class VectorDatabaseSettings(BaseSettings):
    """Configuration for the pgvector-backed vector database."""

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description=(
            "Toggle to enable writing embeddings to PostgreSQL with pgvector. "
            "Set to True once the database schema and extension are provisioned."
        ),
    )
    dsn: Optional[PostgresDsn] = Field(
        default=None,
        description="PostgreSQL connection string used for pgvector storage.",
    )
    schema: str = Field(
        default="docintel",
        description="Database schema that stores DocIntel vector tables.",
        min_length=1,
    )
    embeddings_table: str = Field(
        default="embeddings",
        description="Table name for persisted chunk embeddings.",
        min_length=1,
    )
    migrations_table: str = Field(
        default="db_migrations",
        description="Bookkeeping table storing applied migration versions.",
        min_length=1,
    )
    embedding_dimensions: PositiveInt = Field(
        default=512,
        description=(
            "Expected dimensionality of stored embedding vectors. Must match BiomedCLIP's output dimensions."
        ),
    )
    pool_min_size: PositiveInt = Field(
        default=1,
        description="Minimum number of pooled connections for async Postgres clients.",
    )
    pool_max_size: PositiveInt = Field(
        default=10,
        description="Maximum number of pooled connections for async Postgres clients.",
    )
    statement_timeout_seconds: PositiveFloat = Field(
        default=30.0,
        description="Timeout applied to database statements issued by the embedding pipeline.",
    )

    @field_validator("dsn")
    @classmethod
    def _ensure_dsn_when_enabled(
        cls,
        value: Optional[PostgresDsn],
        info: FieldValidationInfo,
    ) -> Optional[PostgresDsn]:
        data = info.data or {}
        enabled = bool(data.get("enabled"))
        if enabled and not value:
            raise ValueError(
                "DOCINTEL_VECTOR_DB_DSN must be set when DOCINTEL_VECTOR_DB_ENABLED is true"
            )
        return value

    def model_post_init(self, __context: object) -> None:  # noqa: D401 - pydantic hook
        """Apply compatibility environment overrides for vector database settings."""

        override_enabled = os.getenv("DOCINTEL_VECTOR_DB_ENABLED")
        if override_enabled is not None:
            object.__setattr__(self, "enabled", _parse_bool_env_flag(override_enabled))

        override_dsn = os.getenv("DOCINTEL_VECTOR_DB_DSN")
        if override_dsn:
            object.__setattr__(self, "dsn", PostgresDsn(override_dsn))
            if override_enabled is None:
                # Align behaviour with legacy configuration: providing a DSN implicitly enables
                # the vector sink when no explicit flag is set.
                object.__setattr__(self, "enabled", True)


def _parse_bool_env_flag(value: str) -> bool:
    truthy = {"1", "true", "yes", "on", "enable", "enabled"}
    falsy = {"0", "false", "no", "off", "disable", "disabled"}

    normalised = value.strip().lower()
    if normalised in truthy:
        return True
    if normalised in falsy:
        return False
    raise ValueError(
        "DOCINTEL_VECTOR_DB_ENABLED must be one of: 1,0,true,false,yes,no,on,off,enable,disable"
    )


@lru_cache(maxsize=1)
def get_vector_db_settings() -> VectorDatabaseSettings:
    """Return cached settings governing vector database connectivity."""

    return VectorDatabaseSettings()


class AzureOpenAISettings(BaseSettings):
    """Settings for Azure OpenAI integration."""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = Field(
        description="Azure OpenAI API key"
    )
    endpoint: AnyHttpUrl = Field(
        description="Azure OpenAI endpoint URL"
    )
    api_version: str = Field(
        default="2024-12-01-preview",
        description="Azure OpenAI API version"
    )
    deployment_name: str = Field(
        description="Azure OpenAI deployment name (e.g., gpt-4.1)"
    )
    model: str = Field(
        description="Azure OpenAI model name (e.g., gpt-4.1)"
    )


class AgeGraphSettings(BaseSettings):
    """Settings governing Apache AGE graph configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_AGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description="Toggle AGE graph integration without removing relational storage paths.",
    )
    graph_name: str = Field(
        default="clinical_graph",
        description="Name of the AGE property graph used for cypher queries.",
    )
    search_path: str = Field(
        default="ag_catalog, public",
        description="Search path applied after loading the AGE extension.",
    )
    load_extension: bool = Field(
        default=True,
        description="Load the AGE extension for each session before issuing graph queries.",
    )

    @field_validator("graph_name", mode="before")
    @classmethod
    def _validate_graph_name(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("DOCINTEL_AGE_GRAPH_NAME must not be empty")
        return cleaned


@lru_cache(maxsize=1)
def get_age_graph_settings() -> AgeGraphSettings:
    """Return cached settings for Apache AGE graph usage."""

    return AgeGraphSettings()


class RepositoryIngestionSettings(BaseSettings):
    """Settings governing clinical vocabulary ingestion."""

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_REPO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    staging_root: Path = Field(
        default_factory=lambda: Path("data") / "vocabulary_cache",
        description="Root directory used to stage downloaded or extracted vocabulary resources.",
    )
    use_medspacy_vocabularies: bool = Field(
        default=True,
        description=(
            "Use medspaCy QuickUMLS and clinical patterns as vocabulary sources instead of "
            "licensed UMLS/RxNorm/SNOMED. Set to False only when full licensed vocabularies are available."
        ),
    )
    medspacy_quickumls_path: Optional[Path] = Field(
        default=None,
        description=(
            "Path to medspaCy QuickUMLS installation. If None, will auto-detect or use sample data."
        ),
    )
    umls_source_root: Optional[Path] = Field(
        default=None,
        description=(
            "Directory containing UMLS RRF exports (MRCONSO.RRF, MRSTY.RRF, MRREL.RRF).\n"
            "Provide path to a local subset that complies with licensing requirements."
        ),
    )
    rxnorm_source_root: Optional[Path] = Field(
        default=None,
        description=(
            "Directory containing RxNorm RRF exports (RXNCONSO.RRF, RXNREL.RRF)."
        ),
    )
    snomed_source_root: Optional[Path] = Field(
        default=None,
        description=(
            "Directory containing SNOMED CT RF2 exports (Concept, Description, Relationship files)."
        ),
    )
    umls_download_uri: Optional[str] = Field(
        default=None,
        description=(
            "Optional HTTPS URI for retrieving a pre-authorised UMLS subset archive. "
            "Leave unset to rely solely on local paths."
        ),
    )
    rxnorm_download_uri: Optional[str] = Field(
        default=None,
        description="Optional HTTPS URI for retrieving an RxNorm archive.",
    )
    snomed_download_uri: Optional[str] = Field(
        default=None,
        description="Optional HTTPS URI for retrieving a SNOMED CT archive.",
    )
    batch_size: PositiveInt = Field(
        default=2000,
        description="Number of nodes written per transaction during ingestion.",
    )
    edge_batch_size: PositiveInt = Field(
        default=4000,
        description="Number of edges written per transaction during ingestion.",
    )
    dry_run: bool = Field(
        default=False,
        description="Toggle dry-run mode that parses sources without writing to the database.",
    )
    download_timeout_seconds: PositiveFloat = Field(
        default=900.0,
        description="Timeout applied when downloading remote vocabulary archives.",
    )
    download_chunk_size_bytes: PositiveInt = Field(
        default=2_097_152,
        description="Chunk size (bytes) used when streaming archive downloads.",
    )
    checksum_algorithm: str = Field(
        default="sha256",
        description="Digest algorithm applied to archives and per-node content hashes.",
    )
    pixi_command_name: str = Field(
        default="ingest-vocab",
        description="Name of the Pixi task responsible for invoking vocabulary ingestion.",
    )

    def resolve_path(self, value: Optional[Path]) -> Optional[Path]:
        """Return a normalized absolute path when provided."""

        if value is None:
            return None
        return value.expanduser().resolve()

    @field_validator("checksum_algorithm")
    @classmethod
    def _validate_checksum_algorithm(cls, value: str) -> str:
        normalised = (value or "sha256").lower()
        if normalised not in {"sha256", "sha512"}:
            raise ValueError("checksum_algorithm must be either 'sha256' or 'sha512'")
        return normalised

    def ensure_staging_root(self) -> Path:
        """Ensure the staging directory exists and return its absolute path."""

        directory = self.staging_root.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        return directory


@lru_cache(maxsize=1)
def get_repository_ingestion_settings() -> RepositoryIngestionSettings:
    """Return cached repository ingestion settings."""

    return RepositoryIngestionSettings()


class DocIntelConfig(BaseSettings):
    """Main configuration class combining all settings."""

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    dsn: Optional[PostgresDsn] = Field(
        default=None,
        description="Main PostgreSQL connection string"
    )

    @property
    def data_collection(self) -> DataCollectionSettings:
        return get_settings()

    @property
    def parsing(self) -> ParsingSettings:
        return get_parsing_settings()

    @property
    def embedding(self) -> EmbeddingSettings:
        return get_embedding_settings()

    @property
    def vector_db(self) -> VectorDatabaseSettings:
        return get_vector_db_settings()

    @property
    def azure_openai(self) -> AzureOpenAISettings:
        return AzureOpenAISettings()

    @property
    def azure_openai_api_key(self) -> str:
        return self.azure_openai.api_key

    @property
    def azure_openai_endpoint(self) -> str:
        return str(self.azure_openai.endpoint)

    @property
    def azure_openai_api_version(self) -> str:
        return self.azure_openai.api_version

    @property
    def azure_openai_deployment_name(self) -> str:
        return self.azure_openai.deployment_name

    @property
    def docintel_dsn(self) -> str:
        return str(self.dsn) if self.dsn else str(self.vector_db.dsn)

    @property
    def age_graph(self) -> AgeGraphSettings:
        return get_age_graph_settings()

    @property
    def repository_ingestion(self) -> RepositoryIngestionSettings:
        return get_repository_ingestion_settings()


@lru_cache(maxsize=1)
def get_config() -> DocIntelConfig:
    """Return cached main configuration."""
    return DocIntelConfig()
