# DocIntel System Architecture

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shyamsridhar123/ClinicalTrial-KnowledgeMining)
[![Architecture](https://img.shields.io/badge/docs-architecture-blue.svg)](SYSTEM_ARCHITECTURE.md)
[![Status](https://img.shields.io/badge/status-active%20development-yellow.svg)](SYSTEM_ARCHITECTURE.md)

**Last Updated:** October 5, 2025  
**Status:** Active Development

## System Overview

DocIntel is a clinical trial knowledge mining system that combines semantic search with entity extraction to answer questions about clinical trials.

**Core Pipeline:** Document Parsing → Embedding → Entity Extraction → Query Interface

**Technology Stack:**
- **Parsing:** IBM Granite Docling SDK with PyTorch CUDA (GPU-accelerated)
- **Embeddings:** BiomedCLIP via Hugging Face Transformers
- **NLP:** medspaCy + scispaCy for clinical context
- **LLM:** Azure OpenAI GPT-4.1
- **Database:** PostgreSQL + pgvector
- **Environment:** Pixi for dependency management

**NOT USING:** Modular MAX, Mojo kernels, Mammoth orchestration (see `docs/MODULAR_MAX_STATUS.md`)

---

## Actual System State (Verified Oct 5, 2025)

| Component | Count | Status |
|-----------|-------|--------|
| Embeddings | 3,735 | ✅ |
| Entities | 37,657 | ✅ |
| NCT Trials | 15 | ✅ |
| Entity-Chunk Links | 100% | ✅ |

---

## Architecture Components

### 1. Document Processing

**Parser:** IBM Granite Docling 258M SDK (PyTorch CUDA acceleration)
- Extracts text, tables, figures from PDFs
- Output: Markdown, HTML, JSON
- GPU detection: Automatic via PyTorch (`torch.cuda.is_available()`)
- Location: `src/docintel/parse.py`
- **Note:** Direct SDK integration, NOT served via Modular MAX

### 2. Embeddings

**Model:** BiomedCLIP-PubMedBERT (512-dim vectors)
- Multimodal: text + images
- Storage: PostgreSQL with pgvector
- Location: `src/docintel/embeddings/client.py`

### 3. Entity Extraction

**NLP:** spaCy + medspaCy + GPT-4.1
- Extracts: medications, conditions, procedures, adverse events
- Context detection: negation, historical, hypothetical
- Location: `src/docintel/extract.py`
- Output: 37,657 entities with UMLS normalization

### 4. Normalization

**Vocabularies:** UMLS, SNOMED-CT, RxNorm, LOINC
- 3.2M cached terms
- Location: `scripts/normalize_entities.py`
- Storage: `data/vocabulary_cache/`

### 5. Query System

**Interface:** `query_clinical_trials.py`
- Uses U-Retrieval (semantic + graph expansion)
- LLM: Azure OpenAI GPT-4.1
- Query rewriting for short queries
- Location: `src/docintel/knowledge_graph/u_retrieval.py`

---

## Database Schema

### Core Tables

**embeddings** (3,735 rows)
- chunk_id, nct_id, embedding (vector 512), chunk_text
- Indexes: IVFFLAT for vector similarity

**entities** (37,657 rows)
- entity_text, entity_type, normalized_id
- source_chunk_id (links to embeddings)
- context_flags (negation, historical, etc.)

**meta_graphs** (426 rows)
- Document-level graph metadata

**relations** (5,266 rows)
- Subject-predicate-object triples

**repo_nodes** (vocabulary terms)
- UMLS, SNOMED, RxNorm, LOINC mappings

---

## Data Flow

```
1. PDF Upload
   ↓
2. Docling Parsing (GPU) → Text + Tables + Figures
   ↓
3. BiomedCLIP Embedding → 512-dim vectors
   ↓
4. Entity Extraction (GPT-4.1 + medspaCy)
   ↓
5. Normalization (UMLS/SNOMED)
   ↓
6. Store in PostgreSQL
   ↓
7. Query via U-Retrieval → GPT-4.1 Answer
```

---

## Query Pipeline

### User Query → Answer

1. **Query Rewriting** (if needed)
   - "What is X?" → "Define X. X mechanism..."

2. **Embedding Generation**
   - BiomedCLIP encode query → 512-dim vector

3. **U-Retrieval**
   - Semantic search (pgvector)
   - Extract entities from top chunks
   - Graph expansion (1-hop relations)

4. **Context Assembly**
   - Retrieve chunk_text from database
   - Attach entity metadata + context flags

5. **GPT-4.1 Generation**
   - Structured prompt with context
   - Generate answer with citations

**Performance:** ~3-6 seconds end-to-end

---

## Key Features

### Query Rewriting
Expands short queries for better semantic matching
- "What is niraparib?" → "Define niraparib. Niraparib mechanism..."
- Location: `src/docintel/query/query_rewriter.py`

### Context-Aware Entities
Clinical context flags prevent hallucinations
- Negation: "no evidence of toxicity"
- Historical: past medical history
- Hypothetical: protocol conditions

### U-Retrieval
Hierarchical graph-aware retrieval
- Combines semantic search + graph expansion
- Improves recall for related concepts

---

## Hardware & Environment

- **GPU:** NVIDIA RTX A500 (4GB VRAM)
- **OS:** WSL2 Ubuntu 22.04
- **Database:** PostgreSQL 14 (Docker)
- **Environment:** Pixi package manager (dependency management only)
- **GPU Stack:** PyTorch 2.6.0+cu124 with CUDA support

**Note:** Pixi manages Python/package dependencies. We do NOT use Modular MAX runtime or Mojo kernels.

---

## Configuration

Key environment variables (`.env`):

```bash
# Database
DOCINTEL_VECTOR_DB_DSN=postgresql://dbuser:dbpass123@localhost:5432/docintel

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1

# Storage
DOCINTEL_STORAGE_ROOT=./data/ingestion
DOCINTEL_PROCESSED_STORAGE_ROOT=./data/processing
```

---

## Limitations & Future Work

### Current Limitations
- 15 NCTs processed (small dataset)
- No multi-trial comparison
- No temporal reasoning
- No adverse event causality inference

### Planned
- Scale to 50+ NCTs
- Multi-hop graph expansion (currently 1-hop)
- GPT-4o for vision analysis
- Real-time trial monitoring

---

## Usage

### CLI
```bash
pixi run python -m docintel.cli
```

### Direct Query
```bash
pixi run python query_clinical_trials.py "What is niraparib?"
```

---

## Documentation

- **Query Guide:** `docs/query_rewriting_guide.md`
- **U-Retrieval:** `docs/uretrieval_architecture.md`
- **Entity Extraction:** `docs/Entity_Normalization_Guide.md`
- **CLI Guide:** `CLI_GUIDE.md`

---

**Maintained by:** Clinical Trial Knowledge Mining Team  
**Repository:** https://github.com/shyamsridhar123/ClinicalTrial-KnowledgeMining
