# DocIntel Quick Start Guide

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shyamsridhar123/ClinicalTrial-KnowledgeMining)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/README.md)
[![Development Status](https://img.shields.io/badge/status-active%20development-yellow.svg)](https://github.com/shyamsridhar123/ClinicalTrial-KnowledgeMining)

## 🎯 Main Interface: Query Clinical Trials

### Basic Usage
```bash
pixi run -- python query_clinical_trials.py "Your question here"
```

### Example Queries
```bash
# Drug information
pixi run -- python query_clinical_trials.py "What is niraparib?"

# Treatment protocols
pixi run -- python query_clinical_trials.py "Tell me about atezolizumab treatment"

# Study design
pixi run -- python query_clinical_trials.py "What are the patient inclusion criteria?"

# Statistical analysis
pixi run -- python query_clinical_trials.py "What statistical methods were used?"

# Safety data
pixi run -- python query_clinical_trials.py "What are the most common adverse events?"

# Efficacy endpoints
pixi run -- python query_clinical_trials.py "What are the primary endpoints?"
```

### Output
- **Console**: Formatted answer with sources and citations
- **File**: `output/reports/query_result.json` - Full result with metadata

---

## 📊 System Status (Verified Oct 5, 2025)

- **15 NCT Studies** indexed and searchable
- **3,735 Embeddings** (text/tables/figures)
- **37,657 Entities** with UMLS/SNOMED/RxNorm normalization
- **5,266 Relations** for graph-aware retrieval
- **3.2M Vocabulary Terms** cached locally

**Query Pipeline:**
```
User Query → BiomedCLIP → pgvector Search → U-Retrieval (graph expansion) 
→ Context Assembly → GPT-4.1 → Answer + Citations
```

---

## 🛠️ Advanced Operations

### Entity Extraction (from new documents)
```bash
# Fast mode (spaCy/medspaCy only)
pixi run -- python -m docintel.extract --fast --skip-relations

# Full mode (with LLM extraction)
pixi run -- python -m docintel.extract
```

### Entity Normalization
```bash
pixi run -- python scripts/normalize_entities.py
```

### Document Parsing
```bash
pixi run -- env PYTHONPATH=src DOCINTEL_STORAGE_ROOT=./data/ingestion \
  DOCINTEL_PROCESSED_STORAGE_ROOT=./data/processing \
  python -m docintel.parse --max-workers=1 --force-reparse
```

### Embedding Generation
```bash
pixi run -- python -m docintel.embeddings.phase
```

---

## 📁 Key Files & Directories

### Essential
- `query_clinical_trials.py` - **Main query interface** ⭐
- `WORKSPACE_STRUCTURE.md` - Full directory documentation
- `.env` - Configuration (API keys, database connection)

### Source Code
- `src/docintel/` - Main application package
  - `embeddings/client.py` - BiomedCLIP embedding client
  - `extract.py` - Entity extraction pipeline
  - `graph.py` - Knowledge graph builder
  - `parse.py` - Document parser (Docling)
  - `config.py` - Settings management

### Data
- `data/processing/text/` - Extracted text by NCT
- `data/processing/chunks/` - Text chunks for embeddings
- `data/processing/figures/` - Extracted images (PNG)
- `data/processing/tables/` - Extracted tables

### Models
- `models/biomedclip/` - BiomedCLIP weights (512-dim embeddings)
- `models/models--ibm-granite--granite-docling-258M/` - Docling parser

### Output
- `output/reports/query_result.json` - Latest query results
- `logs/*.log` - Execution logs

---

## 🔍 Database Queries (Direct SQL)

### Check embeddings
```sql
SELECT COUNT(*), artefact_type 
FROM docintel.embeddings 
GROUP BY artefact_type;
```

### Check entities
```sql
SELECT COUNT(*), entity_type 
FROM docintel.entities 
GROUP BY entity_type 
ORDER BY count DESC;
```

### Check entity normalization
```sql
SELECT 
    COUNT(*) as total,
    COUNT(normalized_id) as normalized,
    COUNT(DISTINCT normalized_source) as vocabs
FROM docintel.entities;
```

### Semantic search example
```sql
SELECT 
    chunk_id, 
    nct_id, 
    1 - (embedding <=> '[your_embedding_vector]'::vector) as similarity
FROM docintel.embeddings
ORDER BY similarity DESC
LIMIT 5;
```

---

## 🧪 Testing

### Run all tests
```bash
pixi run -- pytest tests/
```

### Specific test categories
```bash
pixi run -- pytest tests/test_embeddings.py
pixi run -- pytest tests/test_extract.py
pixi run -- pytest tests/test_parsing.py
```

### Archive (old debugging scripts)
```bash
ls tests/archive/  # Contains historical test/debug scripts
```

---

## 🐛 Troubleshooting

### GPU not detected
```bash
pixi run -- nvidia-smi
pixi run -- python -c "import torch; print(torch.cuda.is_available())"
```

### Database connection issues
```bash
# Check connection string in .env
grep DOCINTEL_VECTOR_DB_DSN .env

# Test connection
pixi run -- python -c "import psycopg; psycopg.connect('postgresql://dbuser:dbpass123@localhost:5432/docintel')"
```

### Embedding model issues
```bash
# Check model cache
ls models/biomedclip/

# Test embedding generation
pixi run -- python -c "
from docintel.embeddings.client import EmbeddingClient
from docintel.config import EmbeddingSettings
import asyncio

async def test():
    client = EmbeddingClient(EmbeddingSettings())
    result = await client.embed_texts(['test text'])
    print(f'Embedding dimension: {len(result[0].embedding)}')

asyncio.run(test())
"
```

### Missing dependencies
```bash
pixi install
```

---

## 📈 System Metrics

### Current Data (Verified Oct 5, 2025)
- **15 NCT Studies** fully indexed
- **3,735 Embeddings** (text chunks, tables, figures, captions)
- **37,657 Entities** (medications, conditions, procedures, adverse events)
- **5,266 Relations** for knowledge graph traversal
- **3.2M Vocabulary Terms** (UMLS, LOINC, RxNorm, SNOMED-CT)

### Performance
- **Query latency**: ~3-6 seconds (embedding + U-Retrieval + GPT-4.1)
- **Semantic search**: <100ms (pgvector)
- **Graph expansion**: ~100ms (1-hop relations)

---

## 🎓 Key Concepts

### GraphRAG
Combines **semantic search** (embeddings + vector DB) with **knowledge graphs** (entities + relations) to provide rich context for LLM generation.

### Entity Linking
`source_chunk_id` column bridges embeddings and entities, enabling:
1. Find relevant chunks via semantic search
2. Retrieve entities from those chunks
3. Access normalized clinical concepts
4. Build rich context for LLM

### Multimodal Search
System embeds and searches across:
- **Text chunks** (protocol sections, SAPs)
- **Tables** (safety data, demographics)
- **Figures** (survival curves, forest plots)
- **Captions** (figure descriptions)

### Clinical Normalization
Every entity mapped to standard vocabularies:
- **UMLS** - Unified Medical Language System
- **LOINC** - Lab tests & observations
- **RxNorm** - Medications
- **SNOMED CT** - Clinical terms

---

## 📞 Support

See `WORKSPACE_STRUCTURE.md` for detailed directory documentation.

Check `docs/` for technical requirements and architecture details.

Review `logs/` for execution history and debug information.

---

## 📚 Documentation

- **System Architecture:** `docs/SYSTEM_ARCHITECTURE.md`
- **Modular MAX/Mojo Status:** `docs/MODULAR_MAX_STATUS.md` ⚠️ **NOT OPERATIONAL**
- **Query System:** `docs/QUERY_ARCHITECTURE.md`
- **U-Retrieval:** `docs/URETRIEVAL_ARCHITECTURE.md`
- **Entity Normalization:** `docs/ENTITY_NORMALIZATION_GUIDE.md`
- **Query Rewriting:** `docs/QUERY_REWRITING_GUIDE.md`
- **CLI Reference:** `CLI_GUIDE.md`

**Note:** The TRD mentions Modular MAX/Mojo but these are NOT operational. See MODULAR_MAX_STATUS.md for details.

---

**Last Updated:** October 5, 2025
