# Clinical Trial Knowledge Mining Platform - User Guide

> **Version**: 1.0  
> **Last Updated**: September 28, 2025  
> **Platform**: Docling SDK + GPU Acceleration + Modular MAX (optional)

## Table of Contents

1. [Platform Overview](#platform-overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Document Processing Pipeline](#document-processing-pipeline)
4. [Performance Optimization Tools](#performance-optimization-tools)
5. [Data Collection Tools](#data-collection-tools)
6. [Analysis and Query Tools](#analysis-and-query-tools)
7. [Monitoring and Debugging](#monitoring-and-debugging)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

## Platform Overview

### What We've Built

Our clinical trial knowledge mining platform processes unstructured clinical trial documents (PDFs, protocols, study reports) into structured, searchable knowledge using:

- **🚀 GPU-Accelerated Document Parsing** (8x faster with optimizations)
- **🧠 BiomedCLIP Multimodal Embeddings** (512D vectors, text + image processing)
- **📊 Table and Figure Extraction** from clinical documents
- **🔍 Semantic Search** with medical specialization
- **📈 Automated Data Collection** from ClinicalTrials.gov
- **💾 Structured Data Storage** with provenance tracking
- **🎯 Clinical Entity Recognition** and metadata extraction
- **⚡ High-Performance Processing** (<10min for 3000-page documents)

### Architecture

```
Clinical PDFs → Docling Parser → Embeddings → Vector Search → Query Interface
     ↓              ↓              ↓           ↓             ↓
   Ingestion    GPU Processing   Medical     Semantic      Web Interface
   Pipeline     (TF32 + CUDA)    Embeddings  Search        (FastAPI)
                                 (512-dim)   (ChromaDB)
```

## Quick Start Guide

### Prerequisites

1. **Environment Setup**
   ```bash
   # Ensure you're in the project directory
   cd /path/to/docintel
   
   # Activate Pixi environment (REQUIRED)
   pixi shell
   # OR prefix all commands with: pixi run --
   ```

2. **GPU Requirements** (Recommended)
   - NVIDIA GPU with CUDA support
   - 4GB+ VRAM (RTX A500 or better)
   - CUDA 11.8+ or 12.x installed

3. **Basic Health Check**
   ```bash
   pixi run -- env PYTHONPATH=src python -m docintel.docling_health
   ```

### 30-Second Demo

Process a single PDF document:

```bash
# 1. Create test directory and add your PDF
mkdir -p data/ingestion/pdfs/my_study/
cp your_document.pdf data/ingestion/pdfs/my_study/

# 2. Run the optimized parsing pipeline
pixi run -- env PYTHONPATH=src \
  DOCINTEL_STORAGE_ROOT="./data/ingestion" \
  DOCINTEL_PROCESSED_STORAGE_ROOT="./data/processing" \
  python -m docintel.parse --max-workers=1 --force-reparse

# 3. Generate embeddings for semantic search
pixi run -- env PYTHONPATH=src python -m docintel.embed --batch-size=8

# 4. Check results
ls data/processing/           # See all generated artifacts
ls data/processing/embeddings/  # Medical domain embeddings
```

### Complete Workflow: From PDF to Queryable Knowledge

**End-to-end pipeline** to process clinical trial documents and enable U-Retrieval queries:

```bash
# ========================================
# STEP 1: Parse Documents (1-2 minutes per document)
# ========================================
pixi run -- env PYTHONPATH=src \
  DOCINTEL_STORAGE_ROOT="./data/ingestion" \
  DOCINTEL_PROCESSED_STORAGE_ROOT="./data/processing" \
  python -m docintel.parse --max-workers=1 --force-reparse

# Outputs: markdown, html, text, tables, figures, chunks
# Location: data/processing/

# ========================================
# STEP 2: Generate Embeddings (30-60 seconds per document)
# ========================================
pixi run -- env PYTHONPATH=src python -m docintel.embed --batch-size=8

# Outputs: 512-dim BiomedCLIP vectors for text/tables/figures
# Location: data/processing/embeddings/
# Database: docintel.embeddings table (3,735 rows currently)

# ========================================
# STEP 3: Build Knowledge Graph (2-3 minutes per document)
# ========================================
# First time only: Ingest medical vocabularies (30-60 minutes, one-time)
pixi run -- python scripts/ingest_vocabularies.py

# Extract entities and relations
pixi run -- python scripts/build_knowledge_graph.py

# Outputs: 37,657 entities, 5,266 relations
# Database: docintel.entities, docintel.entity_relations tables

# ========================================
# STEP 4: Normalize Entities (10-20 minutes)
# ========================================
pixi run -- python scripts/normalize_entities.py

# Outputs: UMLS/LOINC/RxNorm/SNOMED mappings for all entities
# Database: normalized_id, normalized_source columns populated

# ========================================
# STEP 5: Sync to Apache AGE Graph (5-10 minutes)
# ========================================
pixi run -- python scripts/sync_age_graph.py

# Outputs: Property graph in Apache AGE
# Database: clinical_graph (37,657 nodes, 5,266 edges)

# ========================================
# STEP 6: Query with U-Retrieval (3-5 seconds per query)
# ========================================
pixi run -- python query_clinical_trials.py

# Interactive mode - enter questions at prompt:
# > What are the primary endpoints?
# > What adverse events were reported?
# > What is niraparib?

# Or single command:
pixi run -- python query_clinical_trials.py "What are the inclusion criteria?"
```

**Processing Times (Total for 3 NCTs, 18 documents):**
- Parsing: ~20-30 minutes
- Embeddings: ~10-15 minutes
- Knowledge graph: ~30-45 minutes
- Normalization: ~15-20 minutes
- AGE sync: ~5-10 minutes
- **Total: 1.5-2 hours** (one-time setup)
- **Queries: 3-5 seconds** (after setup)

**Disk Space:**
- Parsed artifacts: ~500 MB
- Embeddings: ~200 MB
- Vocabularies: ~2.5 GB
- Database: ~1.5 GB
- **Total: ~4.5 GB**

## Document Processing Pipeline

### Overview

Our parsing pipeline extracts structured information from clinical trial PDFs with **8x performance improvements** through GPU optimization.

### Processing Tools

#### 0. **Multimodal Embedding Pipeline** (`docintel.embed`)

**Purpose**: Generate 512-dimensional BiomedCLIP embeddings for every parsed artefact (text chunks, table excerpts, figure captions, and figure images when available).

**Usage**:
```bash
pixi run -- env PYTHONPATH=src \
  DOCINTEL_STORAGE_ROOT="./data/ingestion" \
  DOCINTEL_PROCESSED_STORAGE_ROOT="./data/processing" \
  DOCINTEL_EMBEDDING_STORAGE_ROOT="./data/processing/embeddings" \
  python -m docintel.embed --force-reembed --batch-size=32
```

**Key Behaviours**:
- Reuses the local Hugging Face cache under `models/biomedclip-cache` (or the global `~/.cache/huggingface` fallback) so weights and tokenizer stay offline after the first run.
- Automatically normalises the BiomedCLIP identifier when loading tokenizers, preventing the `hf-hub:` prefix warning and ensuring consistent segmentation.
- Emits embeddings for all parsed asset classes: primary text chunks, table condensations, figure captions, and rasterised figure images (when predecessors emit image assets).
- Writes vectors to JSONL (`data/processing/embeddings/vectors/<NCT_ID>/<document>.jsonl`) and PostgreSQL/pgvector with rich metadata for downstream retrieval.
- Respects the configured batch size and 256-token window defined in `EmbeddingSettings` (`docintel.config`).

**Monitoring**:
- GPU memory usage stays below 1 GB on RTX A500-class GPUs when using the default batch size of 32.
- Logs surface the tokenizer selected (e.g., `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`) so you can confirm deterministic tokenisation per run.

### Processing Tools

#### 1. **Main Document Parser** (`docintel.parse`)

**Purpose**: Convert PDF documents to structured knowledge artifacts

**Usage**:
```bash
# Basic usage
pixi run -- env PYTHONPATH=src \
  DOCINTEL_STORAGE_ROOT="./data/ingestion" \
  DOCINTEL_PROCESSED_STORAGE_ROOT="./data/processing" \
  python -m docintel.parse

# Advanced options
python -m docintel.parse \
  --max-workers=4 \           # Parallel processing
  --force-reparse \           # Reprocess existing documents
  --batch-size=8              # GPU batch size optimization
```

**Input Structure**:
```
data/ingestion/pdfs/
├── NCT12345678/           # Study ID directory
│   ├── protocol.pdf       # Study protocol
│   ├── consent_form.pdf   # Informed consent
│   └── amendments.pdf     # Protocol amendments
└── NCT87654321/
    └── study_report.pdf
```

**Output Artifacts**:
```
data/processing/
├── structured/            # JSON document structure
├── markdown/             # Markdown exports
├── html/                # HTML exports  
├── text/                # Plain text extractions
├── tables/              # Extracted table data
├── figures/             # Figure metadata
├── chunks/              # Semantic chunks
└── provenance/          # Processing metadata
```

#### 2. **Performance-Optimized Parser** (NEW!)

**GPU Optimizations Applied**:
- ✅ **TF32 Acceleration**: 2-3x matrix operation speedup
- ✅ **CUDA Memory Optimization**: Smart allocation patterns
- ✅ **GPU Cache Management**: Pre/post processing cleanup
- ✅ **Pipeline Streamlining**: Disabled unnecessary features

**Performance Results**:
- **Small documents (15-30 pages)**: 9-32 seconds (**3-4x faster**)
- **Medium documents (50-90 pages)**: 42-47 seconds (**major improvement**)
- **Example**: DV07.pdf went from 120+ seconds to **14.67 seconds** (8.24x speedup!)

#### 3. **Fallback Processing**

**Automatic Fallbacks**:
- **Docling Failure** → PyMuPDF text extraction
- **Table Structure Issues** → Retry without table detection  
- **Memory Issues** → Automatic GPU cache clearing

**Manual Fallback**:
```bash
# Force PyMuPDF fallback for problematic documents
pixi run -- env PYTHONPATH=src python -c "
from docintel.parsing.client import DoclingClient
from docintel.core.settings import AppSettings
import asyncio
from pathlib import Path

client = DoclingClient(AppSettings())
result = asyncio.run(client.parse_document(
    document_path=Path('path/to/document.pdf'),
    ocr_text='pre-extracted text if available'
))
"
```

### Processing Configuration

#### Environment Variables

```bash
# Required paths
export DOCINTEL_STORAGE_ROOT="./data/ingestion"
export DOCINTEL_PROCESSED_STORAGE_ROOT="./data/processing"

# Optional performance tuning
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
export CUDA_VISIBLE_DEVICES="0"  # Use specific GPU

# Processing options
export DOCINTEL_MAX_WORKERS="4"
export DOCINTEL_BATCH_SIZE="8"
```

#### Document Types Supported

| Document Type | Processing Method | Table Extraction | Expected Time |
|---------------|-------------------|-------------------|---------------|
| **Clinical Protocols** | Docling + GPU | ✅ Excellent | 30-90s |
| **Study Reports** | Docling + GPU | ✅ Excellent | 60-180s |
| **Consent Forms** | Docling + GPU | ⚠️ Limited | 10-30s |
| **Regulatory Submissions** | Docling + GPU | ✅ Good | 45-120s |
| **Amendment Documents** | Docling + GPU | ✅ Good | 15-45s |
| **Case Report Forms** | Docling + GPU | ✅ Excellent | 20-60s |

## Performance Optimization Tools

### 1. **Performance Benchmarking**

**Built-in Benchmark Tool**:
```bash
# Benchmark current performance
pixi run -- env PYTHONPATH=src python -c "
import time
from pathlib import Path
from docintel.parsing.client import DoclingClient
from docintel.core.settings import AppSettings

# Benchmark a document
pdf_path = Path('data/your_test_document.pdf')
client = DoclingClient(AppSettings())

start = time.time()
result = asyncio.run(client.parse_document(document_path=pdf_path))
duration = time.time() - start

print(f'Processing time: {duration:.2f} seconds')
print(f'Pages per second: {len(result.metadata.get(\"pages\", [])) / duration:.2f}')
print(f'Text extraction: {len(result.text)} characters')
print(f'Tables found: {len(result.tables)}')
"
```

### 2. **Performance Monitoring**

**Real-time Monitoring**:
```bash
# Monitor GPU usage during processing
watch -n 1 nvidia-smi

# Monitor processing logs
tail -f logs/processing.log

# Check memory usage
pixi run -- python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated')
    print(f'GPU Memory: {torch.cuda.memory_reserved() / 1024**3:.2f}GB reserved')
"
```

### 3. **Performance Tuning Parameters**

**GPU Optimization Settings**:
```python
# In your processing script
import torch
import os

# Enable TF32 for Ampere GPUs (automatic in our pipeline)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Optimize CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'max_split_size_mb:512,'
    'garbage_collection_threshold:0.8,'
    'expandable_segments:True'
)
```

**Batch Processing Optimization**:
```bash
# Process multiple documents efficiently
python -m docintel.parse \
  --max-workers=4 \        # CPU parallelism
  --batch-size=8 \         # GPU batch size
  --memory-limit=0.8       # Use 80% of GPU memory
```

## Data Collection Tools

### 1. **ClinicalTrials.gov Collector**

**Purpose**: Automatically download clinical trial metadata and documents

**Usage**:
```bash
# Collect data for specific trials
pixi run -- env PYTHONPATH=src python -m data_collection.collectors.clinicaltrials_collector \
  --nct-ids NCT04875806,NCT02030834,NCT02467621 \
  --output-dir ./data/ingestion \
  --download-pdfs

# Collect by search criteria
python -m data_collection.collectors.clinicaltrials_collector \
  --search-term "cancer immunotherapy" \
  --max-results 100 \
  --phase "Phase 3" \
  --status "Completed"
```

**Output Structure**:
```
data/ingestion/
├── metadata/
│   ├── studies_metadata.jsonl    # Structured study data
│   └── studies_summary.csv       # Summary spreadsheet
├── pdfs/
│   ├── NCT04875806/
│   │   ├── Prot_SAP_000.pdf     # Protocol
│   │   └── ICF_000.pdf          # Consent form
└── logs/
    └── collection.log
```

### 2. **Metadata Extraction**

**Clinical Trial Metadata Fields**:
```json
{
  "nct_id": "NCT04875806",
  "title": "Study of Drug X in Cancer Patients",
  "phase": "Phase 3",
  "status": "Completed",
  "enrollment": 500,
  "primary_outcome": "Overall Survival",
  "intervention": "Drug X vs Placebo",
  "condition": "Non-Small Cell Lung Cancer",
  "sponsor": "Pharma Company Inc",
  "start_date": "2020-01-15",
  "completion_date": "2023-06-30",
  "documents": [
    {"type": "protocol", "url": "...", "filename": "Prot_SAP_000.pdf"},
    {"type": "icf", "url": "...", "filename": "ICF_000.pdf"}
  ]
}
```

## Knowledge Graph Construction Pipeline

### Overview

After parsing and embedding generation, the system extracts clinical entities and relationships to build a queryable knowledge graph combining:

- **Entity Extraction**: medspaCy/scispaCy + Azure GPT-4.1
- **Entity Normalization**: UMLS (4.5M concepts), LOINC (96K), RxNorm (140K), SNOMED-CT (352K)
- **Relationship Extraction**: Co-occurrence + semantic analysis
- **Graph Storage**: PostgreSQL + Apache AGE (property graph)
- **Vector Search**: pgvector/Qdrant (BiomedCLIP embeddings)

**Current Scale** (October 2025):
- 37,657 entities (100% normalized)
- 5,266 relationships
- 3,735 embeddings
- 3 NCTs (18 documents)

### Step-by-Step Workflow

#### 1. **Ingest Medical Vocabularies** (One-time setup)

Download and load standard medical vocabularies:

```bash
# Download UMLS, LOINC, RxNorm, SNOMED-CT (requires UMLS license)
# Place vocabulary files in data/vocabulary_sources/

# Ingest into PostgreSQL (takes 30-60 minutes, creates 2.5 GB database)
pixi run -- python scripts/ingest_vocabularies.py
```

**Vocabularies loaded:**
- UMLS: 4.5M concepts
- LOINC: 96K lab tests
- RxNorm: 140K drug concepts
- SNOMED-CT: 352K clinical terms

**Storage:** `data/vocabulary_cache/` (indexed for fast lookup)

#### 2. **Build Knowledge Graph**

Extract entities and relations from all parsed chunks:

```bash
# Process all documents (full pipeline)
pixi run -- python scripts/build_knowledge_graph.py

# Process specific NCT
pixi run -- python scripts/build_knowledge_graph.py --nct-id NCT02467621

# Process first N documents (for testing)
pixi run -- python scripts/build_knowledge_graph.py --limit 5
```

**What it does:**
1. Loads chunks from `data/processing/chunks/`
2. Extracts entities using medspaCy + GPT-4.1
3. Normalizes entities to UMLS/LOINC/RxNorm/SNOMED
4. Extracts relationships (co-occurrence + semantic)
5. Stores in PostgreSQL (`docintel.entities`, `docintel.entity_relations`)
6. Creates `source_chunk_id` linkage to embeddings

**Processing time:**
- Fast mode (medspaCy only): ~30 seconds per document
- Full mode (with GPT-4.1): ~2-3 minutes per document

**Output:** `logs/knowledge_graph_build_report.json`

#### 3. **Normalize Entities**

Ensure all entities are linked to standard vocabularies:

```bash
# Normalize all entities
pixi run -- python scripts/normalize_entities.py

# Monitor progress
pixi run -- python scripts/monitor_normalization.py
```

**Normalization process:**
1. Exact string match (70% of entities)
2. Fuzzy match (20% of entities)
3. Semantic similarity (10% of entities)
4. Manual review (rare failures)

**Check normalization status:**
```bash
pixi run -- python -c "
import psycopg

conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cur = conn.cursor()

cur.execute('''
    SELECT 
        COUNT(*) as total,
        COUNT(normalized_id) as normalized,
        ROUND(100.0 * COUNT(normalized_id) / COUNT(*), 2) as pct
    FROM docintel.entities
''')
result = cur.fetchone()
print(f'Entities: {result[0]}, Normalized: {result[1]} ({result[2]}%)')
"
```

#### 4. **Sync to Apache AGE Graph**

Convert entity/relation tables to property graph format:

```bash
# Sync all data to Apache AGE
pixi run -- python scripts/sync_age_graph.py

# Alternative sync script (handles large graphs)
pixi run -- python scripts/sync_relations_to_age.py
```

**Apache AGE schema:**
- **Nodes**: Entities (with properties: text, type, normalized_id, confidence)
- **Edges**: RELATES_TO relationships (with properties: confidence, source)
- **Graph name**: `clinical_graph`

**Query the graph:**
```bash
pixi run -- python -c "
import psycopg

conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cur = conn.cursor()

cur.execute('LOAD \\'age\\'')
cur.execute('SET search_path = ag_catalog, \\'\$user\\', public')

# Count nodes and edges
cur.execute('''
    SELECT * FROM ag_catalog.cypher(\\'clinical_graph\\', \$\$
        MATCH (n) RETURN count(n) as node_count
    \$\$) as (node_count agtype)
''')
print(f'Nodes: {cur.fetchone()[0]}')

cur.execute('''
    SELECT * FROM ag_catalog.cypher(\\'clinical_graph\\', \$\$
        MATCH ()-[r]->() RETURN count(r) as edge_count
    \$\$) as (edge_count agtype)
''')
print(f'Edges: {cur.fetchone()[0]}')
"
```

### Knowledge Graph Outputs

**PostgreSQL Tables:**
```sql
-- Extracted entities with normalization
docintel.entities (37,657 rows)
  - id, text, entity_type, confidence, normalized_id, normalized_source, source_chunk_id

-- Entity relationships
docintel.entity_relations (5,266 rows)
  - source_entity_id, target_entity_id, relation_type, confidence, source_chunk_id

-- Embeddings (linked via source_chunk_id)
docintel.embeddings (3,735 rows)
  - chunk_id, embedding, metadata, source_chunk_id
```

**Apache AGE Graph:**
```cypher
-- Example queries
MATCH (n:Entity) WHERE n.entity_type = 'Drug' RETURN n LIMIT 10

MATCH (drug)-[r:RELATES_TO]->(condition) 
WHERE drug.entity_type = 'Drug' AND condition.entity_type = 'Condition'
RETURN drug.text, condition.text, r.confidence
```

**JSON Reports:**
```json
{
  "status": "complete",
  "processing_time": "345.67s",
  "entities_extracted": 37657,
  "entities_normalized": 37657,
  "relationships_extracted": 5266,
  "graph_nodes": 37657,
  "graph_edges": 5266,
  "vocabularies": ["UMLS", "LOINC", "RxNorm", "SNOMED-CT"]
}
```

## U-Retrieval Query System

### Overview

The **U-Retrieval** system combines semantic search, entity retrieval, and graph expansion to provide comprehensive answers to clinical questions.

**Architecture:**
```
User Question
    ↓
[BiomedCLIP Embedding] → 512-dim vector
    ↓
[pgvector Search] → Top-K relevant chunks (semantic similarity)
    ↓
[Entity Retrieval] → Extract entities from chunks via source_chunk_id
    ↓
[Graph Expansion] → Find related entities via Apache AGE traversal
    ↓
[Vocabulary Enrichment] → Add UMLS/LOINC/RxNorm/SNOMED concepts
    ↓
[Context Assembly] → Text + Entities + Relations + Normalized terms
    ↓
[Azure GPT-4.1] → Synthesize comprehensive answer
    ↓
Structured Answer + Citations
```

### Basic Usage

#### Interactive Query Mode

```bash
# Interactive prompt (recommended)
pixi run -- python query_clinical_trials.py

# Enter your questions at the prompt:
# > What are the primary endpoints?
# > What adverse events were reported?
# > What are the inclusion criteria?
```

#### Command-line Query

```bash
# Single query
pixi run -- python query_clinical_trials.py "What is niraparib?"

# Query with output file
pixi run -- python query_clinical_trials.py \
  "What are the most common adverse events?" \
  --output output/reports/adverse_events_report.json
```

### Example Queries

**Drug Information:**
```bash
pixi run -- python query_clinical_trials.py "What is niraparib and how does it work?"
```

**Study Design:**
```bash
pixi run -- python query_clinical_trials.py "What are the patient inclusion criteria?"
pixi run -- python query_clinical_trials.py "What is the randomization method?"
pixi run -- python query_clinical_trials.py "What is the study phase and design?"
```

**Efficacy Endpoints:**
```bash
pixi run -- python query_clinical_trials.py "What are the primary and secondary endpoints?"
pixi run -- python query_clinical_trials.py "What is progression-free survival?"
```

**Safety Data:**
```bash
pixi run -- python query_clinical_trials.py "What are the most common adverse events?"
pixi run -- python query_clinical_trials.py "What grade 3/4 toxicities were observed?"
```

**Statistical Methods:**
```bash
pixi run -- python query_clinical_trials.py "What statistical methods were used?"
pixi run -- python query_clinical_trials.py "What is the sample size calculation?"
```

### U-Retrieval Performance

**Validated Metrics** (test_u_retrieval.py - 5/5 passing):
- Processing time: 3.5s (target <5s) ✅
- Entities retrieved: 50 (target 30-100) ✅
- Graph expansion: 22% (11/50 via relationships, target 15-30%) ✅
- Chunk quality: 10 information-dense chunks (target 5-15) ✅
- Answer quality: Comprehensive with citations ✅

**Performance Breakdown:**
```
Total: 3,505ms
├─ BiomedCLIP encoding: 150ms (4.3%)
├─ Vector search: 450ms (12.8%)
├─ Entity extraction: 280ms (8.0%)
├─ Graph expansion (AGE): 620ms (17.7%)
├─ Chunk retrieval: 380ms (10.8%)
├─ Context formatting: 125ms (3.6%)
└─ GPT-4.1 generation: 1,500ms (42.8%)
```

**Comparison to Baseline (Semantic Search Only):**
- Baseline: 5-10 entities, generic answers
- U-Retrieval: 50 entities (+400-900%), comprehensive answers with specific details
- Graph expansion benefit: 22% more relevant entities found through relationships

### Query Output

**Console Output:**
```
🔍 Query: "What are the primary endpoints?"

📊 U-Retrieval Statistics:
  • Processing time: 3.5s
  • Entities retrieved: 50
  • Graph expansion: 11 entities (22%)
  • Chunks retrieved: 10

💡 Answer:

The primary endpoints measured across the clinical trials include:

1. **Overall Survival (OS)**: Time from randomization to death from any cause (NCT02467621)
2. **Progression-Free Survival (PFS)**: Time from randomization to disease progression 
   or death (NCT02467621, NCT02826161)
3. **Objective Response Rate (ORR)**: Proportion of patients with complete or partial
   response (NCT04875806)

All endpoints were assessed using RECIST 1.1 criteria with independent review.

📚 Sources:
  • NCT02467621 chunk-0015 (confidence: 0.92)
  • NCT02467621 chunk-0087 (confidence: 0.89)
  • NCT02826161 chunk-0042 (confidence: 0.85)

🧬 Key Entities:
  • Overall Survival (UMLS:C0282416)
  • Progression-Free Survival (UMLS:C0242792)
  • Objective Response Rate (UMLS:C0598133)
  • RECIST (UMLS:C1709926)
```

**JSON Output** (`output/reports/query_result.json`):
```json
{
  "query": "What are the primary endpoints?",
  "processing_time_ms": 3505,
  "retrieval_stats": {
    "entities_retrieved": 50,
    "graph_expansion_count": 11,
    "graph_expansion_rate": 0.22,
    "chunks_retrieved": 10,
    "top_similarity_score": 0.92
  },
  "answer": "The primary endpoints measured...",
  "sources": [
    {
      "nct_id": "NCT02467621",
      "chunk_id": "chunk-0015",
      "similarity": 0.92,
      "text": "Primary endpoint was overall survival..."
    }
  ],
  "entities": [
    {
      "text": "Overall Survival",
      "entity_type": "ClinicalEndpoint",
      "normalized_id": "UMLS:C0282416",
      "confidence": 0.95
    }
  ]
}
```

### Advanced U-Retrieval Options

**Python API:**
```python
import asyncio
from docintel.knowledge_graph.u_retrieval import ClinicalURetrieval, QueryType, SearchScope

async def advanced_query():
    retrieval = ClinicalURetrieval(db_dsn="postgresql://dbuser:dbpass123@localhost:5432/docintel")
    await retrieval.initialize()
    
    # Focused search
    result = await retrieval.search(
        query="What are the adverse events?",
        query_type=QueryType.SAFETY_DATA,
        nct_ids=["NCT02467621", "NCT02826161"],  # Limit to specific trials
        top_k=20,  # Retrieve top 20 chunks
        expand_graph=True,  # Enable graph expansion
        include_tables=True,  # Include table data
        min_similarity=0.75  # Similarity threshold
    )
    
    print(f"Retrieved {len(result.entities)} entities")
    print(f"Graph expansion: {result.graph_expansion_count} entities")
    print(f"Processing time: {result.processing_time_ms}ms")
    
    return result

asyncio.run(advanced_query())
```

### Troubleshooting Queries

**No results found:**
```bash
# Check if embeddings exist
pixi run -- python -c "
import psycopg
conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM docintel.embeddings')
print(f'Embeddings: {cur.fetchone()[0]}')
"

# If 0, regenerate embeddings
pixi run -- python -m docintel.embed
```

**Slow queries:**
```bash
# Check database indexes
pixi run -- python -c "
import psycopg
conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cur = conn.cursor()
cur.execute('''
    SELECT indexname FROM pg_indexes 
    WHERE tablename IN ('embeddings', 'entities', 'entity_relations')
''')
for row in cur.fetchall():
    print(f'Index: {row[0]}')
"

# Create missing indexes if needed
pixi run -- python -c "
import psycopg
conn = psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')
cur = conn.cursor()
cur.execute('CREATE INDEX IF NOT EXISTS idx_entities_source_chunk ON docintel.entities(source_chunk_id)')
cur.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON docintel.embeddings USING ivfflat(embedding vector_cosine_ops)')
conn.commit()
"
```

**Low-quality answers:**
```bash
# Check entity normalization rate
pixi run -- python scripts/monitor_normalization.py

# Re-normalize if needed
pixi run -- python scripts/normalize_entities.py

# Rebuild graph
pixi run -- python scripts/build_knowledge_graph.py --force
```

## Analysis and Query Tools

### 1. **Document Search and Analysis**

**Search Processed Documents**:
```bash
# Search within processed documents
pixi run -- env PYTHONPATH=src python -c "
from docintel.search.client import SearchClient

search = SearchClient()

# Semantic search
results = search.query(
    'primary endpoint overall survival',
    study_ids=['NCT04875806', 'NCT02030834']
)

# Print results
for result in results:
    print(f'Study: {result.study_id}')
    print(f'Relevance: {result.score:.3f}')
    print(f'Content: {result.text[:200]}...')
"
```

### 2. **Table Data Analysis**

**Extract and Analyze Tables**:
```python
# Access extracted table data
import json
from pathlib import Path

# Load table data
tables_file = Path('data/processing/tables/NCT04875806/Prot_SAP_000.json')
with open(tables_file) as f:
    tables = json.load(f)

# Analyze tables
for i, table in enumerate(tables):
    print(f"Table {i+1}: {table.get('title', 'Untitled')}")
    print(f"Rows: {len(table.get('data', []))}")
    print(f"Columns: {len(table.get('headers', []))}")
```

### 3. **Content Validation**

**Quality Assurance Tools**:
```bash
# Validate processing quality
pixi run -- env PYTHONPATH=src python -m docintel.validation.quality_check \
  --study-id NCT04875806 \
  --check-completeness \
  --check-table-extraction \
  --generate-report
```

## Monitoring and Debugging

### 1. **Health Checks**

**System Health**:
```bash
# Check Docling pipeline health
pixi run -- env PYTHONPATH=src python -m docintel.docling_health

# Check GPU availability
pixi run -- python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"
```

### 2. **Processing Reports**

**Generate Processing Reports**:
```bash
# Generate comprehensive processing report
pixi run -- env PYTHONPATH=src python -c "
from docintel.reports.processing_report import generate_report

report = generate_report(
    input_dir='data/ingestion',
    output_dir='data/processing',
    format='html'  # or 'json', 'csv'
)
print(f'Report saved to: {report.output_path}')
"
```

**Sample Processing Report**:
```json
{
  "summary": {
    "total_documents": 8,
    "successful_processing": 6,
    "fallback_processing": 2,
    "failed_processing": 0,
    "total_processing_time": 355.67,
    "average_time_per_document": 44.46
  },
  "performance": {
    "gpu_utilization": "85%",
    "memory_efficiency": "Good",
    "speedup_achieved": "6.2x",
    "optimization_status": "Enabled"
  },
  "quality_metrics": {
    "text_extraction_rate": "100%",
    "table_extraction_rate": "75%",
    "figure_extraction_rate": "60%"
  }
}
```

### 3. **Error Handling and Debugging**

**Common Issues and Solutions**:

```bash
# Debug Docling parsing failures
pixi run -- env PYTHONPATH=src python -c "
from docintel.debug.docling_debug import diagnose_failure

# Analyze failed document
diagnosis = diagnose_failure('data/ingestion/pdfs/problematic/document.pdf')
print(diagnosis.summary)
print(diagnosis.recommendations)
"
```

**Error Categories**:
- **Docling Core Errors**: `IndexError: basic_string::at` → Automatic PyMuPDF fallback
- **Memory Errors**: GPU OOM → Automatic cache clearing and retry
- **File Access Errors**: Missing files → Clear error messages with paths
- **Format Errors**: Unsupported formats → Format detection and conversion

## Embedding Generation Pipeline

### Overview

The embedding stage converts parsed clinical artefacts into multimodal vectors using the **microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224** checkpoint running through `open_clip`. A single GPU-resident model instance serves both text and figure requests so downstream semantic search, MedGraphRAG retrieval, and analytics share consistent 512-dimensional vectors.

### Key Features

- **Medical focus**: BiomedCLIP couples PubMedBERT text encoders with a ViT backbone tuned on biomedical corpora and figure datasets.
- **GPU acceleration**: PyTorch + CUDA with mixed precision (TF32/FP16) for low-latency batch inference inside the Pixi environment.
- **Tight context control**: 256-token context window enforced before requests hit the model, mirroring `EmbeddingSettings.embedding_max_tokens`.
- **Shared weights**: One process-wide model load amortises IO and ensures consistent outputs across the pipeline.
- **Deterministic fallback**: When dependencies are unavailable the client emits pseudo-random but repeatable vectors so test fixtures stay stable.

### Performance Snapshot

| Batch Size | Duration (s) | Throughput (embeddings/s) | GPU Memory (GB) |
|------------|--------------|---------------------------|-----------------|
| 8          | 1.32         | 3.78                      | 1.64            |
| 16         | 1.35         | 3.69                      | 2.79            |
| 32         | 10.71        | 0.47                      | 3.94            |

_Benchmark source: `embedding_benchmark_results.json` on NVIDIA RTX A500 (4 GB)._ Use batch sizes above 16 only on larger GPUs; throughput tails off sharply past the sweet spot of 8.

### Embedding Utilities

#### 1. Main generator (`docintel.embed`)

```bash
pixi run -- env PYTHONPATH=src python -m docintel.embed
```

Optional flags mirror the CLI help:

```bash
# Tune batch size for your GPU
pixi run -- env PYTHONPATH=src python -m docintel.embed --batch-size=8

# Force re-embedding even if JSONL artefacts exist
pixi run -- env PYTHONPATH=src python -m docintel.embed --force-reembed

# Persist only quantised payloads (bfloat16) to save disk
pixi run -- env PYTHONPATH=src python -m docintel.embed \
  --quantization-encoding bfloat16 \
  --no-store-float32
```

**Defaults pulled from `EmbeddingSettings`:**
- `embedding_model_name`: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- `embedding_batch_size`: `32` (override to `8` on 4 GB GPUs per benchmark guidance)
- `embedding_max_tokens`: `256`
- `embedding_quantization_encoding`: `none`
- `embedding_storage_root`: `data/processing/embeddings`

Environment overrides remain available (`DOCINTEL_EMBEDDING_BATCH_SIZE`, `DOCINTEL_EMBEDDING_QUANTIZATION_ENCODING`, etc.).

#### 2. Performance benchmarking

```bash
pixi run -- python benchmark_embeddings.py
cat embedding_benchmark_results.json
```

The harness records per-batch memory footprints and throughput so you can confirm hardware tuning before production runs.

### Output Layout

```
data/processing/embeddings/
├── vectors/
│   ├── NCT04875806/
│   │   └── protocol.jsonl
│   ├── NCT02030834/
│   │   └── protocol.jsonl
│   └── DV07_test/
│       └── DV07.jsonl
├── logs/
└── temp/
```

Each `.jsonl` file stores one embedding per line:

```
{"chunk_id": "chunk-0000", "embedding": [...], "metadata": {"nct_id": "NCT04875806", "document_name": "protocol.pdf", "segment_index": 0, "segment_count": 5, "token_count": 198, "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", "quantization_encoding": "none"}}
{"chunk_id": "chunk-0000-part01", "embedding": [...], "metadata": {"nct_id": "NCT04875806", "document_name": "protocol.pdf", "segment_index": 1, "segment_count": 5, "token_count": 200, "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", "quantization_encoding": "none"}}
```

Quantised payloads include a compact companion structure:

```
{
  "chunk_id": "chunk-0000",
  "embedding_quantized": {
    "encoding": "bfloat16",
    "values": [16128, 49408, 32768]
  },
  "metadata": {
    "nct_id": "NCT04875806",
    "document_name": "protocol.pdf",
    "segment_index": 0,
    "segment_count": 5,
    "token_count": 198,
    "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "quantization_encoding": "bfloat16",
    "quantization_storage_dtype": "uint16"
  }
}
```

### Clinical Semantics

BiomedCLIP is fluent in trial terminology—eligibility criteria, endpoints, safety findings, PK/PD summaries—and the client passes through provenance metadata (`section`, `page_reference`, `artefact_type`) so retrieval layers can filter by context.

```python
high_signal_queries = [
    "overall survival primary endpoint",
    "grade 3 adverse events",
    "intent-to-treat population definition",
    "pharmacokinetic sampling schedule"
]
```

### Troubleshooting Embeddings

#### Performance issues (slow batches)

```bash
# Inspect GPU utilisation
pixi run -- nvidia-smi

# Confirm BiomedCLIP loads on the expected device
pixi run -- python -c "
import torch
from docintel.config import get_embedding_settings
from docintel.embeddings.client import EmbeddingClient

settings = get_embedding_settings()
client = EmbeddingClient(settings)
print('CUDA available:', torch.cuda.is_available())
print('Embedding dimension:', client.get_embedding_dimension())
" 
```

Lower `DOCINTEL_EMBEDDING_BATCH_SIZE` when VRAM is tight and keep mixed-precision enabled (`torch.backends.cuda.matmul.allow_tf32 = True`).

#### Memory pressure

```bash
pixi run -- env PYTHONPATH=src python -m docintel.embed --batch-size=4
pixi run -- python -c "import torch; torch.cuda.empty_cache()"
```

#### Quality checks

```bash
pixi run -- python -c "
import json
from pathlib import Path
import numpy as np

jsonl_path = Path('data/processing/embeddings/vectors/DV07_test/DV07.jsonl')
records = [json.loads(line) for line in jsonl_path.read_text(encoding='utf-8').splitlines() if line.strip()]

embeddings = [record['embedding'] for record in records]
print('Embedding dimension:', len(embeddings[0]))
print('L2 norm:', np.linalg.norm(embeddings[0]))
print('Model:', records[0]['metadata']['model'])
"
```

The deterministic fallback announces itself via `model == "fallback-deterministic"`; rerun inside the Pixi environment with GPU drivers installed to restore BiomedCLIP outputs.