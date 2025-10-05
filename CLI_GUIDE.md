# DocIntel CLI User Guide

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shyamsridhar123/ClinicalTrial-KnowledgeMining)
[![CLI](https://img.shields.io/badge/interface-CLI-blue.svg)](CLI_GUIDE.md)
[![Interactive](https://img.shields.io/badge/mode-interactive-brightgreen.svg)](CLI_GUIDE.md)

## Overview
The DocIntel CLI provides a comprehensive interactive menu for managing the entire clinical trial knowledge mining pipeline.

## Quick Start

```bash
# Activate the Pixi environment and run the CLI
pixi run python -m docintel.cli

# Or enter the shell first
pixi shell
python -m docintel.cli
```

## Menu Options

### 📥 INGESTION & PROCESSING

**1. Download clinical trial documents**
- Downloads clinical trial documents from ClinicalTrials.gov
- Prompts for: max_studies (default: 25)
- Output: Stores documents in `data/ingestion/`

**2. Parse documents (Docling)**
- Extracts text, tables, and figures using IBM Granite Docling
- GPU-accelerated parsing with automatic model caching
- Prompts for: force_reparse (yes/no), max_workers (optional)
- Output: Processed documents in `data/processing/`

**3. Generate embeddings (BiomedCLIP)**
- Creates semantic embeddings using BiomedCLIP
- Stores in PostgreSQL with pgvector
- Prompts for: force_reembed (yes/no), batch_size (optional)
- Output: Embeddings in `chunks` table

### 🧠 KNOWLEDGE EXTRACTION

**4. Extract entities & relations (GPT-4.1 + medspaCy)**
- Extracts clinical entities (medications, adverse events, conditions)
- Applies context detection (negation, historical, hypothetical, uncertain, family)
- Uses GPT-4.1 for extraction + medspaCy for clinical context
- **Context-aware**: Entities tagged with clinical flags BEFORE relation extraction
- Prompts for: force_reextract (yes/no), batch_size (optional)
- Output: Entities and relations in `entities` and `relations` tables

**5. Build knowledge graph (AGE)**
- Constructs Apache AGE graph from extracted entities and relations
- Creates clinical_graph with nodes and edges
- Output: Knowledge graph in PostgreSQL AGE extension

**6. View graph statistics**
- Shows entity counts by type
- Displays relation counts
- Shows graph size and coverage metrics

### 🔍 QUERY & SEARCH

**7. Semantic search (Ask questions)** ⭐ NEW
- **Context-aware Q&A** with GPT-4.1
- **Intelligent query rewriting**: Automatically expands short queries like "What is X?" for better results
- Uses U-Retrieval (hierarchical graph-aware retrieval)
- **Automatically shows clinical context flags** in results:
  - ❌ NEGATED - "no evidence of X"
  - 📅 HISTORICAL - past conditions
  - 🤔 HYPOTHETICAL - potential scenarios
  - ❓ UNCERTAIN - "possible", "may be"
  - 👨‍👩‍👧 FAMILY - family history
- Prompts for: question, max_results (default: 5)
- **Prevents GPT hallucinations** by showing context annotations

**Example queries:**
- "What adverse events occurred with niraparib?"
- "How should olaparib be administered?"
- "What is the recommended dose for PARP inhibitors?"

**8. Advanced graph query (Cypher)**
- Execute custom Cypher queries on the knowledge graph
- Direct access to Apache AGE
- Prompts for: Cypher query string
- Output: Query results (up to 20 shown)

**Example Cypher queries:**
```cypher
MATCH (e:entity {entity_type: 'medication'})-[r:treats]->(c:entity {entity_type: 'condition'}) 
RETURN e.text, c.text, r LIMIT 10

MATCH (e:entity {entity_type: 'adverse_event'}) 
WHERE NOT e.is_negated 
RETURN e.text, e.mentions 
ORDER BY e.mentions DESC LIMIT 20
```

**9. Test context-aware extraction** ⭐ NEW
- Runs comprehensive test suite for context detection
- Tests negation, historical, hypothetical, family history, uncertain scenarios
- Shows pass/fail for each test case
- Current test pass rate: **83% (5/6 tests passing)**
- Validates that pipeline correctly handles clinical context

### 🔧 UTILITIES

**10. Run full pipeline**
- Executes steps 1→2→3→4→5 in sequence
- Hands-off end-to-end processing
- Stops on errors with detailed traceback

**11. Show system status**
- GPU availability and CUDA version
- Model cache status (Granite Docling, BiomedCLIP)
- Database connectivity
- Configuration settings

**12. Show last run summaries**
- Displays results from all operations run in current session
- JSON-formatted reports with metrics

**0. Exit**
- Exits the CLI

## Pipeline Architecture

### Context-Aware Extraction Flow
```
1. Ingestion → Download documents from ClinicalTrials.gov
2. Parsing → Extract text/tables/figures with Docling (GPU-accelerated)
3. Embedding → Generate BiomedCLIP semantic embeddings
4. Extraction:
   a. GPT-4.1 entity extraction (with medication disambiguation rules)
   b. medspaCy context detection (BEFORE relation extraction) ← KEY
   c. GPT-4.1 relation extraction (WITH context flags attached)
5. Graph Construction → Build AGE knowledge graph
6. Query → U-Retrieval + context-aware prompting
```

### Key Improvements (Latest)
- **Context detection moved BEFORE relation extraction** (Step 4b before 4c)
- **Entity type filtering**: CONTEXT_IMMUNE_TYPES prevents "Patient" from getting clinical flags
- **50% overlap requirement** for stricter entity-context matching
- **Enhanced GPT prompts** with medication pattern recognition (-ib, -mab, -tinib)
- **Query system shows context flags** to prevent hallucinations

## Configuration

The CLI uses settings from `.env` and `pixi.toml`:

### Environment Variables (.env)
```bash
# Azure OpenAI (for GPT-4.1 extraction)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-1106-preview

# PostgreSQL Vector Database
DOCINTEL_VECTOR_DB_DSN=postgresql://user:pass@localhost:5432/docintel

# Storage paths
DOCINTEL_STORAGE_ROOT=./data/ingestion
DOCINTEL_PROCESSED_STORAGE_ROOT=./data/processing

# Model cache
MODULAR_CACHE_DIR=./models
```

### GPU Requirements
- NVIDIA GPU with CUDA 11.x or 12.x
- Windows 10/11 host with WSL2 Ubuntu 22.04
- NVIDIA CUDA Toolkit and nvidia-container-toolkit
- Docker Desktop with GPU passthrough enabled

## Troubleshooting

### "Import could not be resolved" errors
These are IDE lint warnings and can be ignored. The CLI will work correctly when run with `pixi run`.

### GPU not detected
```bash
# Check GPU access
pixi run -- nvidia-smi

# Verify CUDA in Python
pixi run -- python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Database connection issues
```bash
# Check PostgreSQL container status
docker ps | grep postgres

# Verify database connection
pixi run -- python -c "import psycopg; psycopg.connect('$DOCINTEL_VECTOR_DB_DSN')"
```

### Query returning "no results"
- Ensure you've run the full pipeline first (option 10)
- Check that embeddings were generated (option 3)
- Verify knowledge graph was built (option 5)
- Try option 6 to view graph statistics

### Context test failures
- Expected pass rate: 83% (5/6 tests)
- One known minor failure: family history test (entity name precision)
- Run option 9 to see detailed test results

## Performance Targets

- **3000-page document**: ≤10 minutes processing time (goal)
- **Throughput**: ≥150 concurrent documents
- **GPU utilization**: 85-95%
- **Query latency**: ≤1.2s for 95% of requests
- **Entity extraction precision**: ≥95%
- **Adverse event recall**: ≥90%

## Security & Compliance

- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **PHI/PII**: Automatic masking and anonymization
- **RBAC**: Role-based access controls (when deployed)
- **Audit trails**: All operations logged for ≥7 years
- **Standards**: HIPAA, GDPR, FDA 21 CFR Part 11, GxP, SOC 2 Type II

## Support

For issues or questions:
1. Check `docs/` directory for detailed architecture documents
2. Review `WORKSPACE_STRUCTURE.md` for file organization
3. See `docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md` for technical requirements
4. Run option 11 for system diagnostics

## Example Session

```bash
$ pixi run python -m docintel.cli

================================================================================
                  Clinical Trial Knowledge Mining Toolkit
================================================================================

📥 INGESTION & PROCESSING:
  1.  Download clinical trial documents from ClinicalTrials.gov
  2.  Parse documents (extract text, tables, figures with Docling)
  3.  Generate embeddings (BiomedCLIP + pgvector)

🧠 KNOWLEDGE EXTRACTION:
  4.  Extract entities & relations (GPT-4.1 + medspaCy + context-aware NLP)
  5.  Build knowledge graph (PostgreSQL + AGE graph database)
  6.  View graph statistics (nodes, edges, entity types)

🔍 QUERY & SEARCH:
  7.  Semantic search (Ask questions about clinical trials)
  8.  Advanced graph query (Cypher + U-Retrieval)
  9.  Test context-aware extraction (negation, historical, etc.)

🔧 UTILITIES:
  10. Run full pipeline (1→2→3→4→5)
  11. Show system status & configuration
  12. Show last run summaries
  
  0.  Exit
================================================================================

Enter option: 10

# Pipeline runs all steps...

Enter option: 7
🔍 Semantic Search (Context-Aware Q&A)
==================================================
Ask questions about clinical trials. Context flags (negation,
historical, hypothetical, etc.) are automatically applied.

Your question: What adverse events occurred with niraparib?
Maximum results [5]: 10

➡️  Searching for: What adverse events occurred with niraparib?
   Max results: 10

==================================================
ANSWER:
==================================================
Based on the retrieved clinical trial data, niraparib was associated with 
the following adverse events: thrombocytopenia, anemia, and fatigue. 
Note that some mentions indicate "no evidence of hepatotoxicity" 
(marked as NEGATED), meaning hepatotoxicity was NOT observed.

==================================================
RELEVANT ENTITIES:
==================================================
  1. niraparib (medication) ✓ Active
  2. thrombocytopenia (adverse_event) ✓ Active
  3. anemia (adverse_event) ✓ Active
  4. fatigue (adverse_event) ✓ Active
  5. hepatotoxicity (adverse_event) ❌NEGATED
  ...

Retrieved 8 relevant text chunks

✓ Query complete.

Enter option: 0
👋 Bye!
```

## What's New (Latest Version)

### Intelligent Query Rewriting ⭐ NEW
- **Automatic expansion** of short queries like "What is niraparib?" → "Define niraparib. Niraparib mechanism..."
- **Fixes poor semantic discrimination** (all chunks getting identical 0.663 scores)
- **Improves answer quality** for definitional questions
- See `docs/query_rewriting_guide.md` for details

### Context-Aware Query System ⭐
- **Visual context annotations** in query results (❌📅🤔❓👨‍👩‍👧)
- **Prevents GPT hallucinations** by showing clinical context to LLM
- **U-Retrieval integration** for hierarchical graph-aware search
- **Test suite** with 83% pass rate validating context handling

### Enhanced Entity Extraction
- **Medication disambiguation** with pattern recognition (-ib, -mab, -tinib)
- **Context detection BEFORE relation extraction** for accuracy
- **Entity type filtering** (CONTEXT_IMMUNE_TYPES) prevents false positives
- **50% overlap threshold** for stricter context matching

### CLI Improvements
- **Restructured menu** with logical grouping (Ingestion, Extraction, Query, Utilities)
- **Semantic search option** (7) with full context-aware Q&A
- **Context test option** (9) for validation
- **Better error handling** with detailed tracebacks
