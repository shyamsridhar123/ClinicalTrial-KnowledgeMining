# Clinical Trial Knowledge Mining Platform — Technical Requirements Document (Modular-Accelerated Edition)

> **⚠️ IMPORTANT: This is an ASPIRATIONAL specification, not the current implementation.**  
> **For actual system architecture, see:** [`SYSTEM_ARCHITECTURE.md`](./SYSTEM_ARCHITECTURE.md)  
> **Modular MAX/Mojo status:** [`MODULAR_MAX_STATUS.md`](./MODULAR_MAX_STATUS.md) — **NOT OPERATIONAL**

> **Purpose**: Define the end-to-end technical requirements for the clinical trial knowledge mining platform, incorporating Modular MAX/Mojo acceleration to achieve sub-10-minute processing for 3000-page documents while maintaining regulatory-grade accuracy and security.

**Current Reality:** System uses PyTorch CUDA + Docling SDK + PostgreSQL. Modular MAX/Mojo features described herein are future considerations, not production code.

---

## 1. Executive Summary

- Orchestrate end-to-end clinical trial document conversion inside the Pixi-managed environment, combining IBM Granite Docling 258M parsing, clinical NLP, and GPU-accelerated post-processing to keep 3,000-page dossiers under the 10-minute SLA.
- Persist structured outputs, embeddings, and graph primitives in PostgreSQL—leveraging pgvector and the Apache AGE extension—so vectors, metadata, and property-graph views share a single governed data plane.
- Establish a tiered repository graph inspired by MedGraphRAG: Phase 1 links DocIntel entities to UMLS vocabularies; Phase 2 adds curated medical literature/book corpora once sourcing and licensing hurdles are cleared.
- Decouple the system into two workstreams: (1) ingestion + graph construction that parses, chunks, embeds, and extracts triples using GPT-powered agents; (2) a RAG inference layer that fuses pgvector + AGE retrieval with managed OpenAI/Anthropic models for high-accuracy answers.
- Prioritize speed and answer fidelity over strict data residency when necessary while maintaining encryption, PHI redaction heuristics, and multi-year audit trails.
- Deliver ≤1.2 s p95 query latency, ≥150 concurrent-document throughput, and end-to-end provenance from upload through knowledge-graph-backed responses.

## 2. Scope and Objectives

### 2.1 Business Objectives
- Deliver <2s query latency and <10-minute document processing for 95% of workloads.
- Provide regulatory-grade traceability from ingestion through export with full auditability.
- Enable scalable onboarding of 50+ concurrent clinical studies and 100+ active users.

### 2.2 Functional Scope
- **In scope**: document ingestion (PDF/DOCX/PPTX/HTML), multimodal parsing, clinical NLP normalization, knowledge graph and vector search, REST APIs, Modular acceleration, monitoring, and compliance tooling.
- **Out of scope**: manual data entry interfaces, downstream trial management workflows, non-clinical ontologies.

## 3. Stakeholders and User Personas
- Clinical Research Associates, Regulatory Affairs, Pharmacovigilance, Biostatisticians, Medical Affairs, Clinical Data Scientists, Platform Operations (DevOps/SRE), Compliance auditors.

## 4. Solution Overview

### 4.1 High-Level Architecture
```
Clinical Docs ─┐
               ▼
        Ingestion Queue ──► Parsing Workers (Docling + Pixi)
                               │
                               ├─► Chunking & Embeddings (BiomedCLIP → pgvector)
                               │
                               └─► Triple Extraction (GPT-4.1 / Claude via Pixi tasks)
                                              │
                        ┌─────────────────────┴───────────────────────┐
                        ▼                                             ▼
            PostgreSQL (metadata + pgvector)         PostgreSQL + AGE (property graph)
                        │                                             │
                        └──────────────┬──────────────────────────────┘
                                       ▼
                         Hybrid Retrieval Services (pgvector + AGE)
                                       │
                                       ▼
                       Managed LLM Reasoning (OpenAI/Anthropic APIs)
                                       │
                                       ▼
                        Answers + Provenance (REST / CLI / Web UI)
```
- Graph construction produces Meta-Graphs per chunk and links them to a tiered repository graph (UMLS in Phase 1; curated medical literature/books in Phase 2) to mirror MedGraphRAG’s triple graph while respecting sourcing realities.
- Retrieval services implement MedGraphRAG-style U-Retrieval: hierarchical tag summaries drive top-down navigation of AGE views, then bottom-up refinement enriches the final context before managed LLM synthesis.
- Hybrid retrieval exposes both graph paths and dense vector neighbors so downstream applications can blend structured facts with semantic snippets during reasoning.

### 4.2 Key Technology Stack
- **Models**: IBM Granite Docling 258M, BiomedCLIP multimodal embeddings, custom Mojo kernels for UMLS/SNOMED matching, medspaCy context detectors.
- **Acceleration**: Modular MAX (OpenAI-compatible local endpoints), Mojo kernels, Mammoth deployment, CUDA-enabled PyTorch.
- **Datastores**: Encrypted object storage (documents), relational metadata DB, ChromaDB/Qdrant vector store, Prometheus/Elastic stack for monitoring.
- **Managed APIs**: OpenAI (GPT-4.x family) and Anthropic Claude models permitted for inference when they measurably improve throughput or answer quality.
- **Graph Runtime**: PostgreSQL augmented with the Apache AGE extension to expose openCypher-compatible views of the knowledge graph; optional Neo4j deployments reserved for latency-critical graph traversals.

## 5. Functional Requirements

### 5.1 Document Ingestion & Preprocessing
- Accept PDF, DOCX, PPTX, HTML (≤500MB, password-protected support with user credential input).
- Provide asynchronous batch submission with resumable uploads, checksum verification, and queue prioritization.
- Execute OCR (Tesseract fallback) for scanned docs; detect language and flag unsupported locales.
- Record provenance: uploader identity, submission time, checksum, version tags.

### 5.2 Multimodal Parsing & Extraction
- Invoke Granite Docling via Modular MAX for full/region/bbox processing modes.
- Preserve layout hierarchy (headings, sections, tables, footnotes), convert formulas to LaTeX, capture figures with captions and alt-text suggestions.
- Produce structured JSON + Markdown + HTML outputs; maintain page-to-element mapping.

### 5.3 Clinical NLP & Entity Normalization
- Extract medications, dosages, routes; diseases/conditions; procedures; endpoints; eligibility criteria; adverse events; study arms; statistical outcomes.
- Normalize to RxNorm, SNOMED-CT, ICD-10, CPT/HCPCS, UMLS CUIs; provide confidence scores.
- Apply medspaCy for negation, uncertainty, temporality, severity; differentiate family vs patient history.
- Compute hazard ratios, odds ratios, survival metrics, and record contextual qualifiers.

### 5.4 Knowledge Graph & Semantic Search
- Execute a MedGraphRAG-inspired pipeline: semantic chunking feeds GPT-powered entity + relation extraction that emits subject–predicate–object triples with evidence spans and confidence scores.
- Materialize Meta-Graphs for each chunk in PostgreSQL + AGE; link entities to repository tiers (Phase 1: UMLS subset + RxNorm/SNOMED crosswalk; Phase 2: vetted medical literature/textbook corpus pending licensing and sourcing analysis).
- Persist embeddings for every entity and evidence snippet (BiomedCLIP for text/figures) alongside pgvector chunk embeddings so AGE queries can join back to dense similarity results.
- Generate hierarchical tag summaries per Meta-Graph using managed LLM prompts; store tag layers in AGE views to support top-down navigation (U-Retrieval) before bottom-up response refinement.
- Hybrid retrieval workflow: (a) tag-guided traversal selects candidate graphs; (b) combine AGE traversals (k-hop neighborhoods, repository backlinks) with pgvector nearest neighbors; (c) pass structured context + citations to the managed LLM for answer synthesis.
- Define guardrails: configurable hop limits (default 2, max 4), entity confidence thresholds, and repository freshness SLAs to maintain answer precision and latency.

### 5.5 API & Workflow Automation
- Endpoints: `/upload`, `/status/{doc_id}`, `/query`, `/analysis/entities`, `/export/{format}`, `/webhook`, `/healthz`.
- Support streaming responses for long queries, pagination, and asynchronous export jobs.
- Provide OpenAPI/Swagger documentation, SDK snippets, and sample notebooks (all executed through Pixi-managed commands).

### 5.6 User Experience Requirements
- Web dashboard: upload management, processing status, error drill-down, preview of parsed content, query builder, analytics charts (entity distribution, adverse event trends).
- Alerting: email/webhook notifications for job completion/failure, SLA breaches.

## 6. Acceleration & Runtime Layer

-## 6.1 Environment Baseline
- **Mandatory**: developers and pipelines must execute commands via Pixi to ensure consistent environments (`pixi shell` or `pixi run …` as outlined in [Modular's Pixi quickstart](https://docs.modular.com/mojo/manual/get-started/#1-create-a-mojo-project)).
- Runtime targets: WSL2 Ubuntu 22.04 with Docker Desktop GPU passthrough, NVIDIA GPUs (CUDA 11.x/12.x), NVIDIA Container Toolkit, `nvidia-smi` validation at host and container levels.

### 6.2 Modular MAX Integration
- Serve Granite Docling and embedding models via `max serve` using OpenAI-compatible endpoints (`http://localhost:8000/v1`).
- Prefer using the `openai` Python client against local MAX endpoints; however, direct invocation of OpenAI or Anthropic SaaS endpoints is permitted when it materially improves speed or accuracy, with explicit acknowledgement that clinical content will leave controlled infrastructure.
- Enable automatic quantization (target `q4_k`; deploy with `bfloat16` until Granite Docling adds q4 support), kernel fusion, and adaptive batching for throughput gains.
- Provide fallback path to legacy VLLM/Transformers pipeline with feature flagging.

### 6.3 Mojo Clinical Kernels
- Implement GPU-accelerated Mojo modules for UMLS/SNOMED concept matching, negation detection, and adverse-event scoring.
- Expose kernels as reusable services invoked by Python orchestration via MAX Graph APIs.
- Include profiling hooks (NVTX markers) for optimization.

### 6.4 Mammoth Orchestration & Scaling
- Define deployment manifests for auto-scaling GPU workers, dynamic batching, and heterogeneous GPU pools (NVIDIA/AMD/Apple).
- Configure disaggregated inference to separate embedding, parsing, and NLP services for independent scaling.
- Support blue/green deployments with traffic shifting and rollback.

### 6.5 Performance Validation
| Metric | Legacy Baseline | Modular Target | Validation Method |
| --- | --- | --- | --- |
| 3000-page doc processing | 30 min | **≤10 min** | Automated benchmark suite (10 mixed-format docs) |
| Concurrent docs | 50 | **≥150** | Load test harness with backpressure monitoring |
| Query latency (p95) | 2 s | **≤1.2 s** | Synthetic + real clinical query set |
| GPU utilization | 65–70% | **85–95%** | `nvidia-smi`, DCGM exporters |
| VRAM footprint | 1× | **≤0.75×** | Memory profiling during peak load |

## 7. Data Architecture

### 7.1 Storage Inventory
- **Raw Documents**: versioned, immutable, AES-256 encrypted object store.
- **Processed Artifacts**: JSON/Markdown/HTML outputs, table CSVs, embeddings stored with metadata IDs.
- **Metadata DB**: relational store (PostgreSQL) for study metadata, processing lineage, audit trails.
- **Knowledge Graph**: PostgreSQL schemas materialized through Apache AGE; store per-document Meta-Graphs, tiered repository nodes, tag hierarchies, and versioned AGE views for downstream consumers.
- **Repository Data**: Phase 1 ingests authoritative vocabularies (UMLS, RxNorm, SNOMED-CT) via scheduled Pixi pipelines; Phase 2 roadmap adds curated literature/textbook corpora subject to licensing and storage cost review.

### 7.2 Vector Store Schema
- Collections segmented by document type and therapeutic area.
- Index on `nct_id`, `study_phase`, `document_section`, `concept_type`, `timestamp`.
- Support HNSW/IVF indexes with metadata filtering, cold storage for retired studies.
- Publish graph snapshots via AGE views with versioned schemas so downstream services can run openCypher queries without coupling to physical tables.
- Maintain tag summary tables (layer, tag_id, embedding) and retrieval policies (max hops, entity confidence, repository freshness) for the U-Retrieval service layer.

### 7.3 Data Governance
- Enforce schema validation, data quality scoring, deduplication heuristics.
- Maintain lineage records linking raw document → chunk → vector → query results.

### 7.4 Backup & Retention
- Nightly encrypted backups, point-in-time recovery for metadata DB, lifecycle rules for object storage, retention policy configurable per therapeutic area.

## 8. API & Integration Requirements

### 8.1 REST & Streaming APIs
- Provide JSON responses with pagination, retry-safe idempotency keys, webhook signature verification.
- Offer streaming endpoints for long-running analytics via Server-Sent Events (SSE).

### 8.2 Authentication & Authorization
- OAuth2/OIDC for UI, API keys for service accounts, SAML federation for enterprise SSO.
- Role-based permissions: Admin, Clinical Reviewer, Data Scientist, External Partner.
- Field-level security for PHI, redaction policies enforced at serialization layer.

### 8.3 External Connectors
- CTMS bidirectional sync (NCT data, milestones).
- Regulatory submissions (FDA eCTD, EMA CESP) export packages.
- FHIR/HL7/CDISC ODM export templates, ETL to analytics warehouses (Snowflake/BigQuery).

## 9. Performance, Scalability & Reliability
- Horizontal scaling for ingestion and processing workers; queue-based autoscaling triggers on backlog depth.
- Graceful degradation: prioritize critical doc types, throttle non-essential jobs during peaks.
- High availability: ≥99.9% uptime, rolling upgrades, health checks, circuit breakers.
- Disaster recovery plan with warm standby environment and RPO ≤15 minutes, RTO ≤1 hour.

## 10. Security, Privacy & Compliance
- **Priority note**: Speed and accuracy take precedence; apply the following safeguards where feasible without materially degrading performance, and document any intentional deviations.
- Encryption in transit (TLS 1.3), at rest (AES-256), key rotation via HSM/KMS.
- Automated PHI/PII detection with configurable masking/anonymization pipelines.
- Compliance coverage: HIPAA, GDPR, FDA 21 CFR Part 11, GxP, SOC 2 Type II.
- Comprehensive audit logging (user actions, model inference metadata, exports) retained ≥7 years.

## 11. Monitoring & Observability
- Metrics: processing throughput, GPU/CPU utilization, queue depth, accuracy drift, error rates.
- Tools: Prometheus + Grafana, NVIDIA DCGM, ELK/Opensearch for logs, distributed tracing (OpenTelemetry).
- Alerting: SLA breaches, GPU OOM, queue saturation, data drift, PHI detection failures.

## 12. Quality Assurance & Validation
- Automated unit/integration tests for ingestion, parsing, NLP, vector search, APIs.
- Benchmark harness comparing Modular vs. legacy stack (processing time, accuracy, cost).
- Clinical gold-standard validation with SME review across oncology, cardiology, neurology, rare disease cohorts.
- Regression gates for model updates; A/B testing with shadow deployments.

## 13. Implementation & Migration Strategy
- **Phase 0 (Pilot)**: Stand up Modular MAX sandbox, benchmark against legacy pipeline, validate Granite Docling compatibility.
- **Phase 1 (Parallel Run)**: Deploy Modular-accelerated pipeline alongside current stack, mirror 10% traffic, compare metrics.
- **Phase 2 (Gradual Cutover)**: Increase Modular traffic share (25%→50%→100%), enable Mojo kernels, monitor for regressions.
- **Phase 3 (Optimize & Harden)**: Activate Mammoth autoscaling, fine-tune kernels, finalize documentation and training.
- Blue/green deployments with automated rollback, feature flags for toggling inference backends.

## 14. Risk & Mitigation
- **Model compatibility**: Maintain dual-path inference; pre-validate new MAX releases.
- **Accuracy regression**: Automated diffing vs. gold dataset; rollback triggers on KPI breach.
- **Vendor dependency**: Modular APIs kept OpenAI-compatible; containerized deployment ensures portability. PostgreSQL+AGE keeps graph workloads on the same data platform, with Neo4j as an optional acceleration path rather than a hard dependency.
- **Cost overrun**: Track GPU hours and storage; adopt heterogeneous hardware via Mammoth.
- **Compliance breach**: Regular audits, tabletop exercises, penetration testing, data-loss prevention policies.

## 15. Appendices

### Appendix A — Environment Activation & Tooling
```bash
# ALWAYS work inside the Pixi-managed environment before running tooling
pixi shell  # or prefix commands with `pixi run -- …` per Modular's guidance on Pixi workflows

# Example: start Modular MAX with Granite Docling using Pixi
pixi run -- env MAX_LOG_LEVEL=info max serve \
  --model ibm-granite/granite-docling-258M \
  --port 8000 \
  --device-memory-utilization 0.9 \
  --quantization-encoding bfloat16  # Granite Docling currently publishes bfloat16 weights for MAX
```

### Appendix B — Sample Mammoth Deployment Snippet
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modular-granite-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: modular-granite
  template:
    metadata:
      labels:
        app: modular-granite
    spec:
      containers:
      - name: max-engine
        image: clinical-trial-modular:latest
        resources:
          limits:
            nvidia.com/gpu: 2
            cpu: "16"
            memory: 64Gi
          requests:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: 32Gi
        env:
        - name: MAX_GPU_MEMORY_FRACTION
          value: "0.9"
        - name: CLINICAL_VOCAB_PATH
          value: "/app/data/clinical_vocabularies"
```

### Appendix C — Benchmark Reporting Template
| Test Doc | Pages | Format | Processing Time (Legacy) | Processing Time (Modular) | Entity Precision | Adverse Event Recall | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CT-001 | 2,450 | Scanned PDF | 27m 45s | 8m 52s | 95.4% | 90.2% | Minor OCR fallbacks |
| CT-002 | 1,100 | Digital PDF | 9m 12s | 3m 01s | 96.8% | 91.7% | Achieved bfloat16 deployment (q4_k pending upstream support) |
| CT-003 | 3,050 | DOCX | 31m 05s | 9m 44s | 95.9% | 92.3% | Spike in GPU util. to 94% |

### Appendix D — Glossary
- **MAX Engine**: Modular's high-performance inference runtime with OpenAI-compatible APIs.
- **Mojo**: Systems programming language from Modular enabling GPU kernels with Pythonic ergonomics.
- **Mammoth**: Modular's orchestration layer for heterogeneous GPU clusters.
- **Granite Docling 258M**: IBM multimodal VLM for document structure extraction.
- **medspaCy**: Clinical context analysis toolkit (negation, temporality, uncertainty).

---

**Document owner**: Platform Architecture Team  
**Revision date**: 2025-09-24  
**Review cadence**: Quarterly or upon major model/runtime upgrade.
