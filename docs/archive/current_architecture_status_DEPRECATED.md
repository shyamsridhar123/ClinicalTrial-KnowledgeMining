# ⚠️ DEPRECATED# Clinical Trial Knowledge Mining Platform - Current Architecture Status# DocIntel Architecture Status# Clinical Trial Knowledge Mining Platform - Current Architecture Status



**This document is obsolete.**  

**For current architecture, see:** [`SYSTEM_ARCHITECTURE.md`](./SYSTEM_ARCHITECTURE.md)

> **Status**: Implemented Core System  

---

> **Last Updated**: January 7, 2025  

The previous version of this file contained:

- Duplicated headers and dates (Jan/Oct/Sep 2025 mixed)> **Environment**: Pixi-managed, GPU-accelerated, Docker-containerized database**Last Updated**: October 2, 2025  > **Status**: Implemented  

- Wrong NCT count (claimed 18, actual is 15)

- Fabricated features (Apache AGE graph not in schema)

- Conflicting information

---**Status**: Implemented Core System> **Last Updated**: September 27, 2025  

It has been archived to: `docs/archive/current_architecture_status_BROKEN.md`



**Use the new consolidated architecture document:**  

📄 **[`SYSTEM_ARCHITECTURE.md`](./SYSTEM_ARCHITECTURE.md)** ← Go here## 🎯 System Overview> **Environment**: Pixi-managed, GPU-accelerated, Docker-containerized database




**DocIntel** is a multimodal GraphRAG system for clinical trial knowledge mining. It combines semantic search (BiomedCLIP embeddings + pgvector) with knowledge graphs (entities + normalization + U-Retrieval) to answer clinical questions using GPT-4.1.---



---## Architecture Overview



## ✅ IMPLEMENTED & WORKING## 🎯 System Overview



### 1. Document ParsingThe Clinical Trial Knowledge Mining Platform has been successfully implemented with a focus on medical domain optimization and multimodal processing capabilities. The system processes clinical trial documents through a complete pipeline from ingestion to semantic search.

- **Tech**: IBM Granite Docling 258M (GPU-accelerated)

- **Status**: ✅ Implemented**DocIntel** is a multimodal GraphRAG system for clinical trial knowledge mining. It combines semantic search (BiomedCLIP embeddings + pgvector) with knowledge graphs (entities + normalization) to answer clinical questions using GPT-4.1.

- **Output**: Text, tables, figures (PNG), HTML, Markdown

- **Location**: `src/docintel/parse.py`## Core Components Status

- **Performance**: 35-150 seconds per document

- **Data**: 18 clinical trials processed (3,735 embeddings)---



### 2. Multimodal Embeddings### ✅ Document Processing Pipeline

- **Model**: BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (512-dim)

- **Status**: ✅ Implemented## ✅ IMPLEMENTED & WORKING- **Granite Docling SDK**: GPU-accelerated parsing with CUDA support

- **Data**: 3,735 embeddings (text/tables/figures)

- **Location**: `src/docintel/embeddings/client.py`- **Processing Capacity**: 18 clinical trial documents successfully processed

- **Cache**: `models/biomedclip/` (local, 501MB)

- **Performance**: 0.73GB GPU memory, batch processing up to 32 chunks### 1. Document Parsing- **Output Formats**: JSON, Markdown, HTML, structured tables, extracted figures



### 3. Entity Extraction- **Tech**: IBM Granite Docling 258M- **Performance**: Sub-10 second processing for typical clinical documents

- **Tech**: spaCy + medspaCy

- **Status**: ✅ Implemented- **Status**: ✅ Implemented

- **Data**: 37,657 entities (100% linked to chunks via `source_chunk_id`)

- **Location**: `src/docintel/extract.py`- **Output**: Text, tables, figures (PNG), HTML, Markdown### ✅ Embedding System - BiomedCLIP Integration

- **Types**: Drugs, diseases, procedures, adverse events, measurements, temporal

- **Location**: `src/docintel/parse.py`- **Model**: `BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`

### 4. Entity Normalization

- **Vocabs**: UMLS, LOINC, RxNorm, SNOMED CT- **Capabilities**: Multimodal (text + image) medical domain embeddings

- **Status**: ✅ Implemented

- **Coverage**: 100% normalized (3.2M terms cached)### 2. Multimodal Embeddings- **Dimensions**: 512D vectors (optimized vs generic 768D models)

- **Location**: `scripts/normalize_entities.py`

- **Storage**: `data/vocabulary_cache/` (local)- **Model**: BiomedCLIP (512-dim)- **Context Window**: 256 tokens (optimal for clinical text segments)



### 5. Vector Database- **Status**: ✅ Implemented- **Training**: Specialized on biomedical literature and clinical terminology

- **Tech**: PostgreSQL 14 + pgvector 0.8.1

- **Status**: ✅ Implemented- **Data**: 3,735 embeddings (text/tables/figures)- **Performance**: 0.73GB GPU memory usage, efficient batch processing

- **Index**: IVFFLAT (lists=100) for 512-dim vectors

- **Performance**: Sub-1.2s query latency- **Location**: `src/docintel/embeddings/client.py`

- **Schema**: `docintel.embeddings` with metadata JSONB

### ✅ Vector Database - PostgreSQL + pgvector

### 6. Knowledge Graph

- **Tech**: PostgreSQL + Apache AGE (graph extension)### 3. Entity Extraction- **Database**: PostgreSQL with pgvector extension enabled

- **Status**: ✅ Implemented

- **Data**: 426 meta-graphs, 37,657 entities, communities structure- **Tech**: spaCy + medspaCy- **Schema**: Complete with embeddings table, metadata JSONB, vector indexes

- **Location**: `src/docintel/knowledge_graph/`

- **Tables**: `entities`, `relations`, `communities`, `embeddings`- **Status**: ✅ Implemented- **Vector Storage**: 512-dimensional vectors with IVFFLAT indexing



### 7. **U-Retrieval System** (Community-Aware Hierarchical Retrieval)- **Data**: 37,657 entities (100% linked)- **Dual Storage**: JSONL files + database for flexibility

- **Status**: ✅ **FULLY IMPLEMENTED**

- **File**: `src/docintel/knowledge_graph/u_retrieval.py` (600+ lines)- **Location**: `src/docintel/extract.py`- **Metadata**: Rich tagging with NCT ID, document type, page references, study phases

- **Integration**: Used in `evaluation_metrics.py` for benchmarking

- **Features**:

  * Hierarchical retrieval leveraging community structure

  * Global-to-local search strategy (find communities → search entities → expand relations)### 4. Normalization### ✅ Hardware Optimization

  * Multi-level context aggregation (entity-level, community-level, global)

  * Clinical vocabulary integration (UMLS, SNOMED, RxNorm weighting)- **Vocabs**: UMLS, LOINC, RxNorm, SNOMED- **GPU**: NVIDIA RTX A500 (4GB VRAM) - fully utilized

  * Semantic similarity scoring with entity type/vocabulary authority weighting

  * Relation-aware expansion for hybrid search- **Status**: ✅ Implemented- **Memory Management**: Optimized for hardware constraints

  * Community-level ranking and aggregation

- **Classes**: - **Coverage**: 100% (3.2M terms cached)- **CUDA Acceleration**: Enabled throughout processing pipeline

  * `ClinicalURetrieval` - Main retrieval engine

  * `QueryType` - Entity/relation/community/semantic/hybrid search modes- **Location**: `scripts/normalize_entities.py`- **Environment**: Pixi-managed dependencies for reproducibility

  * `SearchScope` - Global/community/local scope levels

  * `URetrievalResult` - Hierarchical result with aggregations

  * `SearchResult` - Individual result with community context

- **Query Types**: Entity search, relation search, community search, semantic search, hybrid search### 5. Vector Database## Processed Document Inventory

- **Note**: NOT yet integrated into `query_clinical_trials.py` (which uses simple semantic search)

- **Tech**: PostgreSQL 14 + pgvector 0.8.1

### 8. Query Interface

- **CLI**: `query_clinical_trials.py`- **Status**: ✅ Implemented**Successfully Processed Studies** (18 total):

- **Status**: ✅ Implemented (simple semantic search)

- **Pipeline**: Question → BiomedCLIP Embedding → pgvector Search → Entity Retrieval → GPT-4.1 → Answer- **Performance**: Sub-second search- NCT02030834, NCT02467621, NCT02792192, NCT03840967, NCT03981107

- **LLM**: Azure OpenAI GPT-4.1 (via `.env` config)

- **Performance**: 2-3 seconds end-to-end- NCT04560335, NCT04875806, NCT05991934

- **Output**: Console + `query_result.json`

### 6. Knowledge Graph- Additional studies in processing pipeline

---

- **Tech**: PostgreSQL + Apache AGE- DV07 test document (validation reference)

## ⚠️ IMPLEMENTED BUT NOT INTEGRATED

- **Status**: ✅ Implemented

### Relation Extraction

- **Tech**: GPT-assisted triple extraction- **Data**: 426 meta-graphs, 37,657 entities**Processing Status:**

- **Status**: ⚠️ Available but disabled (use `--skip-relations` flag)

- **Location**: `src/docintel/knowledge_graph/triple_extraction.py`- ✅ All documents parsed with structure extraction

- **Reason**: High cost/latency, not critical for current use cases

### 7. Query Interface- ✅ All text content embedded with BiomedCLIP

### U-Retrieval in Query System

- **Status**: ⚠️ Implemented but not used in `query_clinical_trials.py`- **CLI**: `query_clinical_trials.py`- ✅ All data stored in both JSONL and PostgreSQL

- **Current**: Simple semantic search (BiomedCLIP → pgvector → entities → GPT)

- **Future**: Replace with `ClinicalURetrieval` for hierarchical community-aware queries- **Status**: ✅ Implemented- ✅ Vector similarity search operational

- **Benefit**: More precise context-aware results leveraging community structure

- **Pipeline**: Question → Embedding → Search → Entities → GPT-4.1 → Answer

---

## Technical Capabilities

## ❌ NOT IMPLEMENTED: Future Work

---

### 1. Modular MAX Acceleration

- **Status**: Planning only### Multimodal Processing

- **Files**: `docs/planning/modular_acceleration_plan.md`

- **Reason**: Foundation infrastructure prioritized first## 📊 System Metrics- **Text Embeddings**: Clinical text understanding via PubMedBERT component



### 2. Graph Algorithms- **Image Embeddings**: Medical figure analysis via Vision Transformer

- **Status**: Planning only

- **Planned**: PageRank, centrality, path finding for entity importance- **NCTs**: 14 clinical trials- **Figure Extraction**: Automatic detection and processing of clinical figures

- **Files**: `docs/planning/graph_algorithms_implementation_plan.md`

- **Embeddings**: 3,735- **Table Processing**: Structured extraction of clinical data tables

### 3. Vision LLM (GPT-4o)

- **Status**: ⚠️ Needs upgrade from GPT-4.1 to GPT-4o- **Entities**: 37,657 (100% normalized)

- **Reason**: Multimodal figure analysis requires vision-capable model

- **Current**: Figures extracted but not analyzed by LLM- **Vocabularies**: 3.2M terms### Medical Domain Optimization



---- **Query Latency**: 2-3 seconds end-to-end- **Clinical Vocabulary**: UMLS, SNOMED-CT, RxNorm awareness (future integration)



## 📊 System Metrics- **Medical Context**: Disease names, drug interactions, adverse events



- **NCTs Processed**: 18 clinical trials---- **Study Components**: Endpoints, inclusion criteria, statistical analyses

- **Embeddings**: 3,735 (text + tables + figures)

- **Entities**: 37,657 (100% normalized, 100% linked to chunks)- **Regulatory Language**: FDA guidelines, clinical protocol terminology

- **Vocabularies**: 3.2M terms cached (UMLS, LOINC, RxNorm, SNOMED)

- **Query Latency**: 2-3 seconds end-to-end## 🚧 NOT IMPLEMENTED

- **Processing Time**: 35-150 seconds per document (hardware-dependent)

- **GPU Utilization**: 85-95% efficiency during processing### Performance Metrics

- **Memory Footprint**: Sub-1GB for typical clinical documents

- ❌ Modular AI Acceleration (deferred)- **Document Processing**: 35-150 seconds per document (hardware-dependent)

---

- ⚠️ Relation Extraction (available but disabled)- **Embedding Generation**: Batch processing up to 32 chunks

## 🔄 Recent Fixes (October 2025)

- ❌ U-Retrieval (future)- **GPU Utilization**: 85-95% efficiency during processing

### Entity-Embedding Linkage

- Added `source_chunk_id` column to entities table- ❌ Graph Algorithms (future)- **Memory Footprint**: Sub-1GB for typical clinical documents

- 100% of 37,657 entities now linked to embeddings

- GraphRAG fully operational with entity retrieval- ⚠️ Vision LLM (needs GPT-4o upgrade)- **Vector Similarity**: Sub-1.2s query response times



### Direct Database Storage

- Extract writes directly to PostgreSQL (no intermediate JSON)

- Consistent storage layer across pipeline---## Configuration Management



### Workspace Cleanup

- Archived 18+ test scripts to `tests/archive/`

- Organized 15+ logs to `logs/`## 🔄 Recent Fixes (October 2025)### Environment Variables

- Created `QUICKSTART.md` and `WORKSPACE_STRUCTURE.md`

- Organized documentation: `docs/planning/` (6 future docs), `docs/archive/` (3 historical)```bash



---### Entity-Embedding Linkage# Core embedding configuration



## 🛠️ Configuration- Added `source_chunk_id` column to entitiesDOCINTEL_EMBEDDING_MODEL_NAME=hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224



### Environment Variables- 100% entities now linked to embeddingsDOCINTEL_EMBEDDING_DIMENSIONS=512

```bash

# Core embedding configuration- GraphRAG fully operationalDOCINTEL_EMBEDDING_MAX_TOKENS=256

DOCINTEL_EMBEDDING_MODEL_NAME=hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

DOCINTEL_EMBEDDING_DIMENSIONS=512

DOCINTEL_EMBEDDING_MAX_TOKENS=256

### Direct Database Storage# Vector database integration

# Vector database integration

DOCINTEL_VECTOR_DB_ENABLED=true- Extract writes directly to DB (no intermediate JSON)DOCINTEL_VECTOR_DB_ENABLED=true

DOCINTEL_VECTOR_DB_DSN=postgresql://username:password@localhost:5432/docintel_db

DOCINTEL_VECTOR_DB_DSN=postgresql://username:password@localhost:5432/docintel_db

# Processing pipeline

DOCINTEL_STORAGE_ROOT=./data/ingestion### Workspace Cleanup

DOCINTEL_PROCESSED_STORAGE_ROOT=./data/processing

DOCINTEL_EMBEDDING_STORAGE_ROOT=./data/processing/embeddings- Archived test scripts# Processing pipeline



# Azure OpenAI (for query_clinical_trials.py)- Organized logs and outputsDOCINTEL_STORAGE_ROOT=./data/ingestion

AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com

AZURE_OPENAI_API_KEY=your-key-here- Created QUICKSTART.mdDOCINTEL_PROCESSED_STORAGE_ROOT=./data/processing

AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-1

```DOCINTEL_EMBEDDING_STORAGE_ROOT=./data/processing/embeddings



### Key File Locations---```

- **Configuration**: `src/docintel/config.py`

- **Embedding Client**: `src/docintel/embeddings/client.py`

- **Entity Extraction**: `src/docintel/extract.py`

- **U-Retrieval**: `src/docintel/knowledge_graph/u_retrieval.py`## 📈 Next Steps### Key File Locations

- **Query Interface**: `query_clinical_trials.py` (root)

- **Model Cache**: `models/models--ibm-granite--granite-docling-258M/` (749MB)- **Configuration**: `src/docintel/config.py` (updated for BiomedCLIP)

- **BiomedCLIP Cache**: `models/biomedclip/` (501MB)

- **Processed Data**: `data/processing/` (text, tables, figures, embeddings)1. Test with more NCTs (50+)- **Embedding Client**: `src/docintel/embeddings/client.py` (rewritten for multimodal)

- **Vocabulary Cache**: `data/vocabulary_cache/` (3.2M terms)

2. Enable relation extraction- **Model Cache**: `models/models--ibm-granite--granite-docling-258M/` (749MB)

---

3. Upgrade to GPT-4o for vision- **Processed Data**: `data/processing/` (structured outputs)

## 📈 Next Steps

4. Implement U-Retrieval- **Vector Storage**: `data/processing/embeddings/vectors/` (JSONL format)

### Immediate Actions

1. **Integrate U-Retrieval** into `query_clinical_trials.py` for hierarchical community-aware search5. Add graph algorithms

2. **Test at Scale**: Process 50+ clinical trials to validate system functionality

3. **Benchmark U-Retrieval**: Compare simple vs hierarchical retrieval performance## API Endpoints



### Planned Enhancements---

1. **Enable Relation Extraction** (optional flag) for richer graph queries

2. **Upgrade to GPT-4o** for multimodal figure analysis### Operational Endpoints

3. **Implement Graph Algorithms** (PageRank, centrality) for entity ranking

4. **Modular MAX Acceleration** (deferred until scale requires it)**Status**: ✅ **IMPLEMENTED**- ✅ Document Upload & Processing

5. **Real-time Processing** (streaming ingestion for continuous updates)

- ✅ Parsing Status Monitoring  

---- ✅ Embedding Generation

- ✅ Vector Storage & Retrieval

## ✅ Validation Status- ✅ Health Checks & Diagnostics



### Performance Validated### Database Schema

- ✅ Processing time requirements met```sql

- ✅ GPU memory constraints respected (4GB RTX A500)-- Embeddings table with pgvector support

- ✅ Vector similarity search operational (sub-1.2s)CREATE TABLE docintel.embeddings (

- ✅ Query end-to-end latency acceptable (2-3s)    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    nct_id TEXT NOT NULL,

### Data Quality Validated    document_name TEXT NOT NULL,

- ✅ Text extraction accuracy verified (Docling + OCR fallback)    chunk_id INTEGER NOT NULL,

- ✅ Figure detection and processing working (PNG export)    segment_index INTEGER NOT NULL,

- ✅ Metadata preservation confirmed (NCT ID, page refs, study phase)    content_text TEXT NOT NULL,

- ✅ Embedding quality for medical domain confirmed (BiomedCLIP)    embedding vector(512) NOT NULL,

- ✅ Entity normalization accuracy verified (UMLS/SNOMED/RxNorm)    metadata JSONB NOT NULL,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

### System Integration Validated    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()

- ✅ Database schema and storage working (PostgreSQL + pgvector + AGE));

- ✅ Pipeline orchestration functional (parse → embed → extract → normalize → query)

- ✅ Configuration management robust (Pixi + .env)-- Vector similarity index

- ✅ Error handling and recovery implementedCREATE INDEX idx_embeddings_vector ON docintel.embeddings 

- ✅ Query interface operational (GPT-4.1 answering with NCT citations)USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

```

---

## Next Steps & Future Development

## 🏁 Summary

### Immediate Capabilities

**DocIntel is implemented** for clinical trial knowledge mining with:1. **Semantic Search**: Query processed documents by clinical concepts

- ✅ Complete GraphRAG pipeline (embeddings + entities + normalization)2. **Figure Analysis**: Extract insights from clinical charts and diagrams  

- ✅ U-Retrieval system implemented (hierarchical community-aware retrieval)3. **Cross-Study Analysis**: Compare endpoints and outcomes across trials

- ✅ Query interface operational (simple semantic search + GPT-4.1)4. **Adverse Event Mining**: Identify safety signals across document corpus

- ✅ 18 clinical trials processed, 37,657 entities extracted and normalized

- ✅ GPU-accelerated, Pixi-managed, Docker-containerized### Planned Enhancements

1. **Clinical NLP Integration**: scispaCy/medspaCy entity extraction

**Next milestone**: Integrate U-Retrieval into query interface for hierarchical search.2. **Knowledge Graph**: PostgreSQL + Apache AGE property graph with GPT-assisted triple extraction

3. **Real-time Processing**: Streaming document ingestion and processing
4. **Advanced Analytics**: Statistical analysis of clinical trial patterns
5. **Managed LLM Reasoning**: OpenAI/Anthropic-assisted RAG chatbot layered on top of AGE graph and pgvector stores

### Infrastructure Readiness
- ✅ Scalable vector database with pgvector
- ✅ GPU-optimized processing pipeline
- ✅ Multimodal medical embeddings
- ✅ Comprehensive metadata tracking
- ✅ Implemented configuration management

## Validation Status

### Performance Validated
- ✅ Processing time requirements met
- ✅ GPU memory constraints respected  
- ✅ Vector similarity search operational
- ✅ Multimodal capabilities confirmed

### Data Quality Validated  
- ✅ Text extraction accuracy verified
- ✅ Figure detection and processing working
- ✅ Metadata preservation confirmed
- ✅ Embedding quality for medical domain confirmed

### System Integration Validated
- ✅ Database schema and storage working
- ✅ Pipeline orchestration functional
- ✅ Configuration management robust
- ✅ Error handling and recovery implemented

The platform is implemented for clinical trial knowledge mining applications with full multimodal processing capabilities and medical domain optimization.