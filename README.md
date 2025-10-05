# Clinical Trial Knowledge Mining – Ingestion & Parsing

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shyamsridhar123/ClinicalTrial-KnowledgeMining)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-active%20development-yellow.svg)](https://github.com/shyamsridhar123/ClinicalTrial-KnowledgeMining)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[![GPU Accelerated](https://img.shields.io/badge/GPU-CUDA%20enabled-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-models-yellow.svg)](https://huggingface.co/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-enabled-blue.svg)](https://github.com/pgvector/pgvector)
[![Apache AGE](https://img.shields.io/badge/Apache%20AGE-graph%20DB-orange.svg)](https://age.apache.org/)

This repository delivers the ingestion and parsing layers for the Clinical Trial Knowledge Mining Platform. The scope covers automated document download from [ClinicalTrials.gov](https://clinicaltrials.gov/) and GPU-accelerated parsing with the Granite Docling 258M SDK in alignment with the Modular TRD (`docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md`).

## Prerequisites

All tooling is orchestrated with Pixi for reproducible dependency management. Install Pixi and add it to your `PATH` before running any project commands:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
```

**Note:** Pixi is used for environment management only. Modular MAX/Mojo are NOT operational in this system (see `docs/MODULAR_MAX_STATUS.md`).

Confirm GPU visibility early—the Docling SDK takes advantage of CUDA automatically when invoked inside the Pixi environment:

```bash
pixi run -- nvidia-smi
```

## Quick start

Create the Pixi environment, run ingestion, and parse the downloaded PDFs:

```bash
pixi install
pixi run -- python -m docintel.ingest
pixi run -- env PYTHONPATH=src DOCINTEL_STORAGE_ROOT=$(pwd)/data/ingestion DOCINTEL_PROCESSED_STORAGE_ROOT=$(pwd)/data/processing python -m docintel.parse --max-workers=1
```

The ingestion phase writes into `data/ingestion` (overridable with `DOCINTEL_STORAGE_ROOT`):

- `pdfs/` – downloaded study documents grouped by NCT ID.
- `metadata/` – normalised study metadata in JSON format.
- `logs/` – reserved for runtime logs.
- `temp/` – scratch workspace for intermediate files.
- `collection_report.json` – execution summary with counts and failure reasons.

The parsing phase consumes the `pdfs/` tree and emits GPU-accelerated Docling artefacts under `data/processing` (overridable with `DOCINTEL_PROCESSED_STORAGE_ROOT`). Outputs include structured JSON, Markdown, HTML, plain text, table exports, figure manifests, chunk manifests, and provenance records.

## Operating pipelines on demand

Ingestion and parsing are separate CLI phases. You can rerun either phase independently, supply your own documents under `pdfs/<NCT_ID>/`, or point the commands at external storage locations using the `DOCINTEL_` environment variables. This keeps the downstream RAG pipeline decoupled from the ClinicalTrials.gov crawler while preserving the storage contract described in the TRD.

## Parsing with the Docling SDK

The parsing CLI invokes Granite Docling directly through the SDK with PyTorch CUDA acceleration. This path proved faster and more reliable than server-based approaches for document-heavy workloads. The CLI auto-detects CUDA availability via PyTorch (`torch.cuda.is_available()`) and logs the chosen device.

1. **Warm the Docling model cache (optional).** Start a parsing run once so the SDK downloads weights into `models/`. Subsequent runs reuse the cache automatically.

2. **Run parsing** from the project root, ensuring `PYTHONPATH` includes `src/` and storage roots point at your document tree:

    ```bash
    pixi run -- env PYTHONPATH=src DOCINTEL_STORAGE_ROOT=$(pwd)/data/ingestion DOCINTEL_PROCESSED_STORAGE_ROOT=$(pwd)/data/processing python -m docintel.parse --max-workers=1 --force-reparse
    ```

3. **Inspect results.** Each NCT ID gets a dedicated folder beneath `data/processing` containing structured outputs ready for embedding and knowledge-graph stages.

### Generate embeddings for parsed chunks

The embedding pipeline uses **BiomedCLIP**, a multimodal medical embedding model that handles both text and image content from clinical trial documents. The system generates 512-dimensional vectors optimized for medical domain understanding:

```bash
pixi run -- env PYTHONPATH=src \
    DOCINTEL_STORAGE_ROOT="$(pwd)/data/ingestion" \
    DOCINTEL_PROCESSED_STORAGE_ROOT="$(pwd)/data/processing" \
    DOCINTEL_EMBEDDING_STORAGE_ROOT="$(pwd)/data/processing/embeddings" \
    python -m docintel.embed --force-reembed --batch-size=32
```

The pipeline processes text chunks, table condensations, figure captions, and figure images through **BiomedCLIP-PubMedBERT_256-vit_base_patch16_224**, combining PubMedBERT for clinical text understanding with Vision Transformer for medical figures. Embeddings are persisted in both JSONL format (`data/processing/embeddings/vectors/`) and PostgreSQL with pgvector extension for semantic similarity search. Tokenizer loading automatically strips the `hf-hub:` prefix and falls back to `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`, eliminating earlier warning noise.

## Features

- **Clinical Trial Data Collection**: Download documents from ClinicalTrials.gov
- **GPU-Accelerated Parsing**: Extract text, tables, and figures using IBM Granite Docling SDK (PyTorch CUDA)
- **Semantic Embeddings**: Generate BiomedCLIP embeddings with pgvector storage
- **Context-Aware Entity Extraction**: GPT-4.1 + medspaCy with clinical context detection (negation, historical, hypothetical, etc.)
- **Knowledge Graph**: Build Apache AGE graph from entities and relations
- **Intelligent Query Rewriting**: Automatically expands short queries (e.g., "What is X?") for better semantic search
- **U-Retrieval**: Hierarchical graph-aware semantic search
- **Interactive CLI**: Comprehensive command-line interface for all operations

**Database Integration:**
The embedding pipeline automatically populates a PostgreSQL database with pgvector extension enabled. Vectors are stored with rich metadata including `nct_id`, `document_type`, `chunk_id`, `page_reference`, and study phase information for precise retrieval.

**Performance:**
- Processes 15 clinical trial documents with sub-1GB GPU memory usage
- 256-token context window optimized for clinical text segments
- Batch processing up to 32 chunks simultaneously

### Document Status & Processing Pipeline

The system has successfully processed 15 clinical trial documents through the complete pipeline:

**Database State (verified Oct 5, 2025):**
- 15 NCT studies indexed
- 3,735 embeddings generated
- 37,657 entities extracted with UMLS normalization
- Full semantic search and knowledge graph capabilities active

**Ingested Studies:** NCT02030834, NCT02467621, NCT02792192, NCT03840967, NCT03981107, NCT04560335, NCT04875806, NCT05991934, and others

**Processing Outputs:**
- Structured JSON with document hierarchy and metadata
- Markdown and HTML formats for human readability  
- Extracted tables, figures, and text chunks
- Generated embeddings for all text content
- Provenance tracking for audit compliance

**Vector Database:**
All processed documents are indexed in PostgreSQL with pgvector, enabling semantic search across:
- Study protocols and methodologies
- Clinical endpoints and outcomes
- Adverse events and safety data
- Patient demographics and inclusion criteria
- Statistical analyses and results

## Configuration

Settings live in `docintel.config`. They accept `.env` overrides or environment variables prefixed with `DOCINTEL_`. Key ingestion controls include:

- `DOCINTEL_TARGET_THERAPEUTIC_AREAS` / `DOCINTEL_TARGET_PHASES` – optional filters that bias the ClinicalTrials.gov search.
- `DOCINTEL_SEARCH_QUERY_TERM` – advanced search expression for targeted runs.
- `DOCINTEL_SEARCH_OVERFETCH_MULTIPLIER` – number of extra studies fetched to favour document-rich trials.

Parsing options such as `DOCINTEL_PROCESSED_STORAGE_ROOT`, `DOCINTEL_MAX_WORKERS`, and `DOCINTEL_DOCLING_REQUEST_TIMEOUT_SECONDS` mirror the fields defined in `ParsingSettings`.

## Testing

Run the test suite via Pixi to respect the managed environment:

```bash
pixi run -- pytest
```

The fixtures exercise the ingestion client, storage helpers, and configuration validation with deterministic responses.

## Documentation

**Architecture:**
- 📘 [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md) - Complete system overview
- 📘 [`docs/QUERY_ARCHITECTURE.md`](docs/QUERY_ARCHITECTURE.md) - Query system details
- 📘 [`docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md`](docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md) - Technical requirements

**User Guides:**
- 📖 [`docs/QUERY_REWRITING_GUIDE.md`](docs/QUERY_REWRITING_GUIDE.md) - Query rewriting usage
- 📖 [`CLI_GUIDE.md`](CLI_GUIDE.md) - Interactive CLI reference
- 📖 [`QUICKSTART.md`](QUICKSTART.md) - Quick start guide

**Technical References:**
- 🔧 [`docs/Entity_Normalization_Guide.md`](docs/Entity_Normalization_Guide.md) - Entity processing
- 🔧 [`docs/uretrieval_architecture.md`](docs/uretrieval_architecture.md) - U-Retrieval details
- 🔧 [`docs/docling_parsing_architecture.md`](docs/docling_parsing_architecture.md) - Parsing pipeline

## Roadmap

- Expand GPU-aware OCR fallback wiring described in the TRD.
- Scale to 50+ NCT studies with multi-trial comparison capabilities.
- Implement multi-hop graph queries for advanced reasoning.
- Feed parsing telemetry into the observability stack (Prometheus, OpenTelemetry, NVIDIA DCGM).
