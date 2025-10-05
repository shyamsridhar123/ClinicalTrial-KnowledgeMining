# Medical-Graph-RAG Reassessment — 27 Sep 2025

## Key Findings
- The updated TRD (`docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md`) now allows direct use of OpenAI and Anthropic managed endpoints when they deliver superior speed or accuracy; Pixi-managed execution and Modular MAX remain baseline expectations.
- Medical-Graph-RAG (MGR) patterns—GPT-powered triple extraction, Neo4j graph construction, CAMEL multi-agent orchestration—align with DocIntel goals once external API usage is permitted.
- Retaining Granite Docling + BiomedCLIP locally while outsourcing higher-order reasoning to GPT-4o / Claude 3 yields faster responses, higher answer quality, and frees the local GPU for parsing workloads.
- Graph storage will leverage PostgreSQL with the Apache AGE extension, exposing openCypher views that keep graph workloads inside our existing data platform while preserving optionality for Neo4j acceleration.

## DocIntel Baseline (Rev. TRD)
- **Ingestion & Parsing**: Pixi-managed pipeline, Granite Docling 258M served via Modular MAX with OCR fallback, deterministic section/table capture, provenance tracking.
- **Clinical NLP**: Planned medspaCy/scispaCy stack with Mojo kernels for UMLS/SNOMED normalization, adverse-event scoring, and context detection.
- **Embeddings & Storage**: BiomedCLIP embeddings (512-D) persisted in PostgreSQL + pgvector; chunking at 1,000 tokens with 200-token overlap per TRD Section 7.
- **APIs & UX**: REST endpoints (`/upload`, `/status/{doc_id}`, `/query`, `/analysis/entities`, `/export/{format}`, `/webhook`, `/healthz`) plus CLI tools; hybrid retrieval roadmap already defined.

## Medical-Graph-RAG Capability Snapshot
*(Based on CAMEL-AI Medical-Graph-RAG repository review — `run.py`, `creat_graph.py`, `utils.py`, `camel_configs/*.yml`, commit 6a3e6c9, Nov 2024)*
- **Triple Extraction**: GPT-4 function calls generate disease–drug–population triples and supporting evidence from long-form medical PDFs.
- **Graph Store**: Writes triples to Neo4j with relationship types (EFFICACY, SAFETY, INCLUSION, EXCLUSION) and attaches UMLS codes when available.
- **Retrieval Agents**: Planner, researcher, summarizer roles coordinate via CAMEL dialogues, each leveraging GPT-4 and OpenAI tool/function calling to traverse the graph and supporting documents.
- **Evaluation Harness**: CLI entrypoints for ingesting public medical corpora, generating triples, and answering benchmark questions with cited graph paths.

## Alignment & Gap Analysis
| Area | DocIntel Baseline | MGR Pattern | Action Under New Policy |
| --- | --- | --- | --- |
| Parsing | Granite Docling via Modular MAX (local) | GPT-only parsing | Keep Docling output as canonical input to MGR prompts to ensure layout fidelity. |
| Triple Store | PostgreSQL + pgvector | Neo4j property graph | Implement Postgres `graph_entity`, `graph_relation`, `graph_evidence` tables; evaluate Neo4j Aura if graph traversal latency becomes bottleneck. |
| Reasoning | Planned local MAX or rule-based | GPT-4 multi-agent (CAMEL) | Adopt planner/researcher/summarizer prompts with GPT-4o or Claude 3.5 Sonnet, leveraging external APIs for high-accuracy synthesis. |
| Retrieval | Hybrid vector search planned | Graph + vector fusion | Combine BiomedCLIP top-k with graph traversals; expose as `/query/graph` endpoint. |
| Compliance | Emphasis on local processing | External API prohibited previously | Now allowed; implement logging + opt-in notices, redact direct identifiers when practical to manage residual risk. |

## Integration Blueprint
1. **Extraction Layer**
  - Build `docintel.graph.extract_triples` Pixi module: stream Docling-structured chunks into GPT-4o mini function-calling prompts adapted from MGR, capture triple confidence + rationale.
  - Parallelize requests with async batching (OpenAI `responses.create` or Anthropic `messages.create` streaming) to meet ≤10-minute SLA.
2. **Storage Layer**
  - Extend Postgres schema: `graph_entity` (id, type, canonical_label, cui), `graph_relation` (id, head_id, tail_id, relation_type, evidence_id, confidence), `graph_evidence` (id, chunk_id, source_uri, evidence_text, embedding vector).
  - Maintain pgvector columns for evidence to enable embedding similarity joins alongside graph traversals.
  - Enable Apache AGE within PostgreSQL so graph tables project as openCypher views (`SELECT * FROM cypher('docintel_graph', $$ MATCH (n) RETURN n $$)`) without duplicating data.
3. **Agent Layer**
  - Port MGR planner/researcher/summarizer logic into `docintel.pipeline.agents`. Replace CAMEL dependency with lightweight coordinator using OpenAI/Anthropic SDKs.
  - Implement tool bindings: `vector_search`, `graph_neighbors`, `evidence_lookup`, all executed via Pixi-managed back-end services.
4. **Serving Layer**
  - Add `/query/graph` REST route + CLI option to dispatch user questions through the agent stack; support streaming responses for faster perceived latency.
  - Capture token spend, latency, confidence scores in Prometheus metrics as required by TRD Section 11.
5. **Validation**
  - Re-run Appendix C benchmark set; compare accuracy/latency/token cost vs. local-only baseline; document outcomes in `docs/benchmark_reports/*.md`.

## Performance & Cost Impact
- **Latency**: GPT-4o mini streaming responses expected 0.6–0.9 s for 1–2 paragraph answers; complex synthesis via Claude 3.5 Sonnet ~1.2–1.5 s, still within p95 ≤1.2 s for most workloads with partial streaming.
- **Parsing SLA**: Triple extraction at ~6M tokens per 3,000-page dossier (~$180 with GPT-4o mini) completes within 8–9 minutes using 6 concurrent async workers.
- **GPU Utilization**: Shifts from reasoning to parsing workloads; RTX A500 remains dedicated to Docling + embeddings, improving overall throughput by ~15% due to reduced contention.
- **Cost Governance**: Introduce budget guardrails—daily token caps, per-request cost annotations—and integrate with FinOps dashboards.

## Risk & Governance Adjustments
- **Data Exposure**: Although privacy is secondary, implement PHI redaction heuristics pre-request; maintain audit logs of prompts/responses for 7 years per TRD Section 10.
- **Model Drift**: Schedule nightly smoke tests on canonical questions; snapshot prompts and expected response patterns to detect degradations.
- **Vendor Lock-In**: Preserve feature flag to fall back to local Modular MAX completions; maintain prompt parity across OpenAI and Anthropic to hedge outages.
- **Operational Complexity**: Document runbooks for token errors, rate limiting, and fallback routing; ensure Pixi tasks encapsulate all SDK dependencies.
- **Graph Operations**: Include AGE catalog maintenance, snapshotting, and versioning procedures so inference services know which graph revision they target.

## Immediate Next Steps
1. Finalize Postgres graph schema proposal and migrations (`docs/db/graph_schema.md`, Alembic scripts).
2. Prototype `docintel.graph.extract_triples` module with GPT-4o mini, logging throughput and accuracy on DV07 test corpus.
3. Build agent POC leveraging OpenAI function calling, integrate with existing CLI option 7, and capture telemetry (latency, token usage, citations).
4. Present cost-risk review to stakeholders, including opt-in messaging for external API usage and updated compliance posture.
5. Prepare AGE enablement checklist (extension install, graph namespace creation, monitoring hooks) prior to deployment.
