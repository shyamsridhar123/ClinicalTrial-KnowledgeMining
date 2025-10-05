# Query System Architecture

**Last Updated:** October 5, 2025

## Overview

DocIntel's query system combines semantic search (BiomedCLIP embeddings) with knowledge graph expansion (U-Retrieval) to answer clinical trial questions using GPT-4.1.

**Query Flow:** User Question → Query Rewriting → Embedding → U-Retrieval → Context Assembly → GPT-4.1 Answer

---

## Components

### 1. Query Rewriting

**Purpose:** Improve semantic matching for short queries

**Implementation:** `src/docintel/query/query_rewriter.py`

**How it works:**
- Detects patterns: "What is X?", "How does X work?"
- Expands to multiple phrasings
- Example: "What is niraparib?" → "Define niraparib. Niraparib mechanism of action. Niraparib description."

**Trigger:** Queries ≤10 words with question patterns

**Performance:** <1ms overhead

---

### 2. Embedding Generation

**Model:** BiomedCLIP-PubMedBERT
- 512-dimensional vectors
- Multimodal: text + images
- Location: `src/docintel/embeddings/client.py`

---

### 3. U-Retrieval

**Purpose:** Hierarchical retrieval combining semantic + graph

**Implementation:** `src/docintel/knowledge_graph/u_retrieval.py`

**Process:**

1. **Semantic Search** (pgvector)
   - Query embedding → top-k chunks by cosine similarity
   - Default k=10

2. **Entity Extraction**
   - Extract entities from retrieved chunks
   - Include: medications, conditions, procedures, adverse events

3. **Graph Expansion** (1-hop)
   - Find related entities via `relations` table
   - Subject-predicate-object triples

4. **Context Assembly**
   - Retrieve `chunk_text` from database
   - Attach entity metadata + context flags

**Output:** Ranked chunks with entity annotations

---

### 4. Answer Generation

**LLM:** Azure OpenAI GPT-4.1

**Prompt Structure:**
```
System: You are a clinical trial expert...
Context: [Retrieved chunks with metadata]
User: [Original question]
```

**Output:** Structured answer with citations

---

## Data Flow

```
User Query
  ↓
Query Rewriter (if needed)
  ↓
BiomedCLIP Embedding (512-dim)
  ↓
pgvector Similarity Search
  ↓
Top-k Chunks (k=10)
  ↓
Extract Entities from Chunks
  ↓
Graph Expansion (1-hop relations)
  ↓
Assemble Context (chunk_text + entities)
  ↓
GPT-4.1 Generation
  ↓
Answer + Citations
```

**Performance:** ~3-6 seconds end-to-end

---

## Database Queries

### Semantic Search
```sql
SELECT chunk_id, chunk_text, nct_id,
       1 - (embedding <=> query_embedding) AS similarity
FROM docintel.embeddings
ORDER BY similarity DESC
LIMIT 10;
```

### Entity Lookup
```sql
SELECT entity_text, entity_type, normalized_id, context_flags
FROM docintel.entities
WHERE source_chunk_id = ANY($1);
```

### Graph Expansion
```sql
SELECT subject_id, predicate, object_id
FROM docintel.relations
WHERE subject_id = ANY($1) OR object_id = ANY($1);
```

---

## Configuration

**Environment Variables:**
```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1

# Database
DOCINTEL_VECTOR_DB_DSN=postgresql://dbuser:dbpass123@localhost:5432/docintel

# Query Parameters (defaults shown)
DOCINTEL_RETRIEVAL_TOP_K=10
DOCINTEL_TEMPERATURE=0.1
DOCINTEL_MAX_TOKENS=2048
```

---

## Usage

### CLI
```bash
pixi run python -m docintel.cli
# Select option 7: Semantic Query
```

### Programmatic
```python
from query_clinical_trials import ClinicalTrialQA

qa = ClinicalTrialQA()
result = qa.query("What is niraparib used for?")
print(result)
```

**Note:** Always run via `pixi` to preserve dependencies.

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Query Rewriting | <1ms | Pattern matching |
| Embedding | ~200ms | BiomedCLIP GPU |
| Semantic Search | ~50ms | pgvector IVFFLAT |
| Graph Expansion | ~100ms | 1-hop relations |
| Context Assembly | ~150ms | Database reads |
| GPT-4.1 Generation | ~3-5s | Azure API |
| **Total** | **~3-6s** | End-to-end |

**Dataset:** 3,735 embeddings, 37,657 entities, 15 NCTs

---

## Key Features

### Context-Aware Entities
Prevents hallucinations by marking clinical context:
- **Negation:** "no evidence of toxicity"
- **Historical:** Past medical history
- **Hypothetical:** Protocol eligibility criteria

### Citation Tracking
Every answer includes:
- NCT ID
- Chunk reference
- Page number (if available)

### Query Diagnostics
CLI shows:
- Original query
- Rewritten query (if modified)
- Relevance scores
- Retrieved chunk IDs

---

## Limitations

### Current
- 1-hop graph expansion only (no multi-hop reasoning)
- No temporal reasoning (trial phases, timelines)
- No cross-trial comparison
- No adverse event causality inference

### Future
- Multi-hop graph queries
- Temporal awareness
- Trial-to-trial comparison
- GPT-4o vision for figures/charts

---

## Related Documentation

- **Query Rewriting:** `docs/query_rewriting_guide.md`
- **U-Retrieval:** `docs/uretrieval_architecture.md`
- **System Architecture:** `docs/SYSTEM_ARCHITECTURE.md`
- **CLI Guide:** `CLI_GUIDE.md`

---

## Debugging

### Check Query Rewriting
```bash
pixi run python -c "from src.docintel.query import QueryRewriter; r = QueryRewriter(); print(r.rewrite('What is niraparib?'))"
```

### Verify Embeddings
```bash
pixi run python -c "from src.docintel.embeddings.client import EmbeddingClient; c = EmbeddingClient(); print(c.get_embedding('test text').shape)"
```

### Test Database Connection
```bash
pixi run -- psql $DOCINTEL_VECTOR_DB_DSN -c "SELECT COUNT(*) FROM docintel.embeddings;"
```

---

**Maintained by:** Clinical Trial Knowledge Mining Team
