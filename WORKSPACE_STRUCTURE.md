# DocIntel Workspace Structure

## 📁 Directory Organization

### **Root Directory** (Clean - Essential Files Only)
- `README.md` - Main project documentation
- `pixi.toml` / `pixi.lock` - Pixi dependency management
- `pyproject.toml` - Python project configuration
- `pytest.ini` - Test configuration
- `query_clinical_trials.py` - **Main CLI tool for querying clinical trials** ⭐

### **Core Application Code**
- `src/` - All production source code
  - `docintel/` - Main package
    - `embeddings/` - BiomedCLIP embedding client
    - `knowledge_graph/` - Entity extraction & graph construction
    - `config.py` - Configuration management
    - `parse.py` - Document parsing with Docling
    - `extract.py` - Entity extraction pipeline
    - `graph.py` - Graph building orchestration

### **Scripts & Tools**
- `scripts/` - Production utility scripts
  - `normalize_entities.py` - Entity normalization to UMLS/LOINC
  - `analyze_real_data.py` - Data analysis utilities
  - Other operational scripts

### **Tests**
- `tests/` - Test suite
  - `archive/` - Old test scripts and debugging tools
    - `test_*.py` - Various test files
    - `analyze_database_completeness.py` - DB validation
    - `benchmark_embeddings.py` - Performance tests
    - `check_*.py` - Verification scripts
    - `debug_*.py` - Debug utilities

### **Data & Processing**
- `data/` - Data storage
  - `ingestion/` - Raw PDF/DOCX files
  - `processing/` - Processed outputs
    - `text/` - Extracted text by NCT
    - `chunks/` - Text chunks for embedding
    - `markdown/` - Markdown conversions
    - `figures/` - Extracted figures (PNG)
    - `tables/` - Extracted tables
    - `html/` - HTML outputs
    - `structured/` - Structured extractions
  - `vocabulary_cache/` - Clinical vocabulary caches
  - `vocabulary_sources/` - UMLS/SNOMED/RxNorm sources

### **Models**
- `models/` - ML model cache
  - `biomedclip/` - BiomedCLIP weights
  - `models--ibm-granite--granite-docling-258M/` - Docling model
  - `models--sentence-transformers--*/` - Other embedding models

### **Database**
- `db/` - Database schemas and migrations
  - `migrations/` - SQL migration scripts
- `schema/` - Database schema definitions
  - `add_source_chunk_id_migration.sql` - Entity linkage schema

### **Documentation**
- `docs/` - Project documentation
  - `Clinical_Trial_Knowledge_Mining_TRD_Modular.md` - Technical requirements
  - `clinical-trial-mining-prd (1).md` - Product requirements
  - `current_architecture_status.md` - Architecture overview
  - `user_guide.md` - User documentation
  - Various analysis and design docs

### **Configuration**
- `config/` - Configuration files
  - Environment-specific settings
  - Model configurations

### **Logs & Debug**
- `logs/` - All log files and debug output
  - `*.log` - Execution logs
  - `debug/` - Debug artifacts
    - `debug_ingestion/` - Ingestion debugging
    - `debug_processing/` - Processing debugging
  - Historical extraction logs

### **Output**
- `output/` - Generated outputs and reports
  - `reports/` - Analysis reports and results
    - `query_result.json` - Latest query results
    - `embedding_benchmark_results.json` - Performance metrics
    - `DEMO_REMOVAL_SUCCESS_REPORT.md` - Cleanup reports
  - `extractions/` - Extraction results

---

## 🚀 Main Usage

### Query Clinical Trials
```bash
pixi run -- python query_clinical_trials.py "What is niraparib?"
```

### Entity Extraction
```bash
pixi run -- python -m docintel.extract --fast --skip-relations
```

### Entity Normalization
```bash
pixi run -- python scripts/normalize_entities.py
```

### Run Tests
```bash
pixi run -- pytest tests/
```

---

## 📊 Database

**PostgreSQL** with extensions:
- `pgvector` - Vector similarity search (3,735 embeddings)
- `Apache AGE` - Graph database (37,657 entities)

**Key Tables:**
- `docintel.embeddings` - BiomedCLIP embeddings (text/tables/figures)
- `docintel.entities` - Clinical entities with normalization
- `docintel.meta_graphs` - Knowledge graph structures
- `docintel.relations` - Entity relationships
- `docintel.vocabulary_*` - Clinical vocabularies (UMLS/LOINC/RxNorm/SNOMED)

---

## 🧹 Cleanup Notes

**Archived to `tests/archive/`:**
- Old test scripts (`test_*.py`)
- Debug utilities (`debug_*.py`, `check_*.py`)
- Analysis scripts (`analyze_*.py`, `benchmark_*.py`)

**Moved to `logs/`:**
- All `.log` files
- Debug directories

**Moved to `output/reports/`:**
- Result files (`.json`)
- Status reports (`.md`)

**Root directory now contains only:**
- Essential config files
- Main query interface
- Directory structure

---

## 🎯 System Status

✅ **Operational Components:**
- BiomedCLIP embeddings (3,735 total)
- Entity extraction (37,657 entities)
- Entity normalization (100% coverage)
- Clinical vocabularies (3.2M terms)
- Semantic search (pgvector)
- Knowledge graph (Apache AGE)
- LLM integration (Azure GPT-4.1)

✅ **Working Pipeline:**
1. Query → Embedding generation
2. Semantic search → Relevant chunks
3. Entity retrieval → Clinical concepts
4. LLM synthesis → Structured answer
5. Citation → NCT IDs

---

Last updated: October 2, 2025
