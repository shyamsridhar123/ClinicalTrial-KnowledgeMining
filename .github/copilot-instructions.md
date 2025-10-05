# Clinical Trial Knowledge Mining Platform – AI Agent Instructions

> Always ground your work in the latest TRD: `docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md`.

## 1. Critical Runtime & Verification Requirements

- **Pixi-managed environment is mandatory**. Follow Modular's guidance for Pixi projects (`pixi shell` or `pixi run …`) to guarantee reproducible toolchains (see https://docs.modular.com/mojo/manual/get-started/).
- **Pixi execution is required**. Invoke all tooling through `pixi run …` (or defined pixi tasks). Never call `python`, `max`, `pytest`, etc. directly—wrap them with Pixi to ensure dependency parity.
- **GPU acceleration is active**. CUDA PyTorch 2.6.0+cu124 installed, GPU detected as `cuda:0` (NVIDIA RTX A500), and Docling uses GPU acceleration automatically.
- **Local model cache**. Granite Docling 258M (501MB) cached in `models/models--ibm-granite--granite-docling-258M/` for offline usage.
- **No speculation**. If information is unavailable or unverified, explicitly state the gap instead of guessing.

### Authoritative References (consult before answering)
- 📘 `docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md`
- 📘 `docs/clinical-trial-mining-prd (1).md`
- 🌐 https://docs.modular.com/ — **cite this for every Modular MAX/Mojo/Mammoth detail; never invent features**.
- 🌐 Official vendor docs for: IBM Granite Docling, scispaCy/medspaCy, ChromaDB/Qdrant, NVIDIA CUDA/WSL2.

### Required Web Lookups
Perform a fresh fetch/search for any technical question involving:
- IBM Granite Docling 258M usage or configuration details.
- Modular MAX/Mojo/Mammoth deployment, APIs, or performance tuning.
- scispaCy/medspaCy updates.
- Vector database configuration (ChromaDB/Qdrant/Milvus).
- Docker + WSL2 GPU passthrough, NVIDIA driver/toolkit changes.
- Clinical vocabularies (UMLS, RxNorm, SNOMED-CT, ICD-10).

## 2. Platform Overview (align with Modular TRD)

- Mission: Convert unstructured clinical trial content into a secure, queryable knowledge fabric with sub-10-minute turnaround for 3000-page documents.
- Core stack: IBM Granite Docling 258M served through Modular MAX, clinical NLP via scispaCy/medspaCy, Mojo kernels for GPU-accelerated entity/context processing, and a hybrid semantic search layer over ChromaDB/Qdrant.
- Target environment: Windows 10/11 host + WSL2 Ubuntu 22.04, Docker Desktop (GPU enabled), NVIDIA CUDA 11.x/12.x, nvidia-container-toolkit.

## 3. Architecture & Pipeline Duties

1. **Ingestion & Preprocessing**
    - Accept PDF/DOCX/PPTX/HTML ≤500 MB, support resumable uploads and checksum validation.
    - Apply OCR (Tesseract fallback) for scanned pages; capture provenance metadata.

2. **Modular-Accelerated Parsing**
    - Serve Granite Docling through `max serve` with OpenAI-compatible endpoints (local only).
    - Preserve hierarchy (sections, tables, figures) and output JSON/Markdown/HTML + LaTeX for formulas.

3. **Clinical NLP & Mojo Kernels**
    - Use scispaCy/medspaCy for entity extraction, context detection (negation, temporality, severity).
    - Invoke Mojo GPU kernels for UMLS/SNOMED matching and adverse-event scoring.

4. **Knowledge Graph & Vector Indexing**
    - Chunk text (1,000 tokens, 200-token overlap) via recursive splitter.
    - Generate embeddings with Modular MAX (ClinicalBERT/MiniLM) and attach metadata: `nct_id`, `study_phase`, `therapeutic_area`, `document_type`, `page_ref`, population, endpoints.

5. **API & Delivery**
    - REST endpoints: `/upload`, `/status/{doc_id}`, `/query`, `/analysis/entities`, `/export/{format}`, `/webhook`, `/healthz`.
    - Ensure authentication (OAuth2/OIDC, SAML, API keys) and RBAC enforcement.

## 4. Development Patterns & Constraints

- **Async data collection**: `aiohttp.ClientSession` + `tenacity` retries; keep NCT IDs as primary keys.
- **Configuration**: centralize constants in `data_collection/config.py` (missing; create when needed).
- **Storage discipline**: retain original documents, processed artifacts, and lineage metadata with encryption at rest.
- **Vector DB schema**: maintain collections by document type/therapeutic area; choose HNSW/IVF indexes per TRD.
- **Export formats**: CSV, JSON, Markdown, FHIR/CDISC, HL7.

## 5. Modular Acceleration Guidance

- Always reference https://docs.modular.com/ for MAX/Mojo/Mammoth behavior, CLI flags, deployment recipes, and cite relevant sections when summarizing.
- Treat the `openai` Python client as an HTTP shim: set `base_url` to the local MAX endpoint and never route traffic to OpenAI-operated services.
- Enable MAX features per TRD: quantization (target `q4_k`; use `bfloat16` fallback when required), kernel fusion, adaptive batching, Mammoth orchestration for auto-scaling—wire these through pixi-managed commands.
- Provide fallbacks: wrap Modular-specific code paths with feature flags and document how to revert to legacy VLLM/Transformers.

## 6. Performance & Validation Targets

- 3000-page document processing ≤10 minutes (goal), with throughput ≥150 concurrent documents and GPU utilization 85–95%.
- Semantic query latency ≤1.2 s for 95% of requests.
- Entity extraction precision ≥95%, adverse-event recall ≥90%; verify against gold-standard datasets.
- Log benchmarking artifacts (processing time, accuracy deltas) per Appendix C in the TRD.

## 7. Monitoring & Operations

- Metrics: ingestion queue depth, processing duration, GPU/CPU utilization, embedding latency, accuracy drift.
- Tooling: Prometheus + Grafana, NVIDIA DCGM exporters, ELK/OpenSearch, OpenTelemetry tracing.
- Alerts: GPU OOM, queue saturation, SLA breach, PHI masking failure, Modular service degradation.

## 8. Common Commands & Snippets

```bash
# Enter the managed environment when needed
pixi shell  # or prefix commands with `pixi run -- …`

# Validate GPU access and CUDA setup
pixi run -- nvidia-smi
pixi run -- python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Start Modular MAX with Granite Docling and local model cache
pixi run -- bash scripts/start_max_server.sh

# Run optimized document parsing (uses GPU acceleration automatically)
pixi run -- env PYTHONPATH=src DOCINTEL_STORAGE_ROOT=./data/ingestion DOCINTEL_PROCESSED_STORAGE_ROOT=./data/processing python -m docintel.parse --max-workers=1 --force-reparse

# Test simple PDF parsing directly
pixi run -- python test_simple_docling.py

# Warm up model cache (downloads to local models/ folder)
pixi run -- env MODULAR_CACHE_DIR=$(pwd)/models max warm-cache --model ibm-granite/granite-docling-258M --devices=gpu --quantization-encoding bfloat16
```

```python
# Using the OpenAI client against local MAX (do not hit OpenAI servers)
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
completion = client.chat.completions.create(
    model="ibm-granite/granite-docling-258M",
     messages=[{"role": "system", "content": "Extract clinical endpoints"},
                  {"role": "user", "content": document_payload}],
     temperature=0.1,
     max_tokens=4096
)
# Run this module within `pixi run python …` to preserve the managed environment.
```

## Database
- The project's PostgreSQL instance already runs inside the Docker container shipped with the repo, so there is no need to install a local database manually.
- Use the connection string provided via `DOCINTEL_VECTOR_DB_DSN` in `.env`; it points at the Docker-hosted database.

## LLM Access
- Whenever the code invokes an LLM, route the request through the Azure OpenAI GPT-4.1 deployment configured for this project.
- Credentials and deployment metadata (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME`, etc.) are stored in `.env`. Keep this file private and never commit live secrets.

## 9. Security & Compliance Checklist

- Enforce encryption at rest (AES-256) and in transit (TLS 1.3).
- Apply PHI/PII masking, anonymization, and role-based access controls.
- Maintain audit trails (uploads, processing steps, exports) for ≥7 years.
- Adhere to HIPAA, GDPR, FDA 21 CFR Part 11, GxP, SOC 2 Type II requirements.

## 10. Do & Don’t Summary

- ✅ Cross-reference the Modular TRD and https://docs.modular.com/ for every acceleration-related change.
- ✅ Run all Python tasks via Pixi (`pixi run …` or inside `pixi shell`) and document any deviations with justification.
- ✅ State clearly when information is unknown or needs confirmation.
- ❌ Never fabricate configuration values, APIs, or performance metrics.
- ❌ Never send proprietary clinical data to third-party services.

Focus on delivering accurate, regulation-ready pipelines that respect the documented architecture while leveraging Modular acceleration safely.