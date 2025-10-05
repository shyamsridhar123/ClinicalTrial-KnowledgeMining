# U-Retrieval Architecture

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shyamsridhar123/ClinicalTrial-KnowledgeMining)
[![U-Retrieval](https://img.shields.io/badge/component-U--Retrieval-blue.svg)](URETRIEVAL_ARCHITECTURE.md)
[![Status](https://img.shields.io/badge/status-active%20development-yellow.svg)](URETRIEVAL_ARCHITECTURE.md)

**Last Updated:** October 5, 2025  
**Status:** Active Development

## Overview

U-Retrieval (Unified Retrieval) combines semantic search with knowledge graph expansion to improve clinical trial question-answering recall and accuracy.

**Core Concept:** Start with semantic search, then expand results using entity relationships to discover clinically relevant information that may not have high vector similarity.

---

## How It Works

### 3-Step Process

```
1. Semantic Search (pgvector)
   Query embedding → Top-K chunks by cosine similarity
   
2. Entity Extraction
   Extract entities from retrieved chunks
   
3. Graph Expansion (1-hop)
   Follow entity relationships → Discover related entities
```

### Data Flow

```
User Query: "What are the primary endpoints?"
    ↓
BiomedCLIP Embedding (512-dim)
    ↓
pgvector Search → Top 10 chunks
    ↓
Extract Entities → 39 entities from chunks
    ↓
Graph Expansion → Find related entities via relations table
    ↓
+11 entities discovered (28% increase)
    ↓
Assemble Context → 50 total entities with metadata
    ↓
GPT-4.1 Generation → Comprehensive answer
```

---

## Performance

**Validated Results (Oct 2025):**

| Metric | Semantic Only | U-Retrieval | Improvement |
|--------|--------------|-------------|-------------|
| Entities Retrieved | 39 | 50 | +28% |
| Graph-Expanded | 0 | 11 (22%) | New |
| Answer Quality | Generic | Comprehensive | ✅ |
| Processing Time | 450ms | 3.5s | +3s |

**Trade-off:** 3 seconds additional latency for 28% more relevant entities and significantly better answers.

---

## Implementation

### Location
`src/docintel/knowledge_graph/u_retrieval.py`

### Usage in Query Pipeline

```python
from docintel.knowledge_graph.u_retrieval import ClinicalURetrieval

# Initialize
retrieval = ClinicalURetrieval(
    db_dsn=DOCINTEL_VECTOR_DB_DSN,
    embedding_client=embedding_client
)

# Search
results = await retrieval.u_retrieval_search(
    query="What are the primary endpoints?",
    top_k=50,
    max_hops=1
)

# Results contain:
# - entities: List of SearchResult objects
# - metadata: source_chunk_id, hop_distance, relevance_score
```

### Graph Expansion

**Database Query (1-hop):**
```sql
-- Find related entities
SELECT DISTINCT 
    CASE 
        WHEN r.subject_id = e.entity_id THEN r.object_id
        ELSE r.subject_id
    END AS related_entity_id,
    r.predicate AS relationship_type
FROM docintel.relations r
JOIN docintel.entities e ON (e.entity_id = r.subject_id OR e.entity_id = r.object_id)
WHERE e.entity_id = ANY($1);
```

**Relevance Scoring:**
- Direct matches (0-hop): 1.0
- 1-hop expansion: 0.4
- 2-hop expansion: 0.25
- 3-hop expansion: 0.15

---

## Configuration

**Environment Variables:**
```bash
# U-Retrieval parameters
DOCINTEL_RETRIEVAL_TOP_K=50        # Number of entities to retrieve
DOCINTEL_MAX_HOPS=1                # Graph traversal depth (1-3)
DOCINTEL_GRAPH_EXPANSION=true     # Enable graph expansion
```

**Code Configuration:**
```python
# In query_clinical_trials.py
retrieval_config = {
    "top_k": 50,           # Return up to 50 entities
    "max_hops": 1,         # 1-hop graph expansion
    "use_graph": True      # Enable graph traversal
}
```

---

## Key Features

### 1. Hierarchical Search
Combines multiple retrieval strategies:
- **Level 1:** Vector similarity (BiomedCLIP)
- **Level 2:** Graph traversal (entity relations)
- **Level 3:** Entity normalization (UMLS/SNOMED)

### 2. Entity-Centric Context
Groups results by source chunk:
- Prioritizes information-dense chunks
- Preserves clinical context
- Links entities to original documents

### 3. Hop-Based Scoring
Penalizes distant relationships:
- Direct matches most relevant
- 1-hop relations moderately relevant
- 2-3 hop relations least relevant

---

## Integration Points

### Query Pipeline
`query_clinical_trials.py` → `retrieve_context()` method uses U-Retrieval automatically

### CLI
Option 7 (Semantic Query) inherits U-Retrieval automatically

### Entity Extraction
Extracts entities from top chunks using `source_chunk_id` linkage

### Context Assembly
Groups entities by chunk, sorts by entity density, selects top 10 chunks

---

## Example Output

**Query:** "What are the primary endpoints?"

**U-Retrieval Results:**
```
Direct Matches (39 entities):
- Progression-free survival (PFS) [SNOMED: 277933002]
- Overall survival (OS) [SNOMED: 444717001]
- Objective response rate (ORR) [SNOMED: 268527003]
- ...

Graph-Expanded (11 entities):
- Tumor assessment [via: endpoint → measurement]
- RECIST criteria [via: endpoint → evaluation_method]
- Overall response duration [via: ORR → related_measure]
- ...
```

**Answer Quality:**
- ✅ Comprehensive: All primary and secondary endpoints listed
- ✅ Specific: Exact endpoint names from trials
- ✅ Contextual: How endpoints are measured
- ✅ Citations: NCT IDs and chunk references

---

## Limitations

### Current
- **1-hop expansion only:** No multi-hop reasoning
- **No temporal awareness:** Can't reason about trial phases/timelines
- **No cross-trial comparison:** Each query limited to single trial context
- **Latency cost:** +3 seconds vs pure semantic search

### Future Improvements
- Multi-hop graph queries (2-3 hops)
- Temporal reasoning for trial progression
- Cross-trial entity linking and comparison
- Adaptive hop depth based on query complexity

---

## Troubleshooting

### No Graph Expansion Happening

**Check:**
1. Entities table populated? `SELECT COUNT(*) FROM docintel.entities;`
2. Relations table populated? `SELECT COUNT(*) FROM docintel.relations;`
3. Graph expansion enabled? `DOCINTEL_GRAPH_EXPANSION=true`
4. Entities have `source_chunk_id`? Check metadata field

### Low Entity Counts

**Fix:**
1. Increase `top_k`: Default 50 → Try 100
2. Increase `max_hops`: Default 1 → Try 2
3. Check entity extraction logs for errors

### Slow Performance

**Optimize:**
1. Reduce `max_hops`: 2-3 → 1
2. Reduce `top_k`: 100 → 50
3. Add database indexes on `source_chunk_id` and relation columns
4. Check query plan: `EXPLAIN ANALYZE` on graph expansion query

---

## Technical Details

**Algorithm:** Hierarchical graph-aware retrieval with hop-based scoring

**Database:** PostgreSQL with pgvector extension

**Graph Storage:** Relations table (subject-predicate-object triples)

**Embedding Model:** BiomedCLIP-PubMedBERT (512-dim)

**LLM:** Azure OpenAI GPT-4.1

---

## Related Documentation

- **Query System:** `docs/QUERY_ARCHITECTURE.md`
- **Entity Extraction:** `docs/Entity_Normalization_Guide.md`
- **System Overview:** `docs/SYSTEM_ARCHITECTURE.md`

---

**Maintained by:** Clinical Trial Knowledge Mining Team
