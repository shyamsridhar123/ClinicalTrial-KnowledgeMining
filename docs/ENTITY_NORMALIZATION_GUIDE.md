# Entity Normalization Guide

**Last Updated:** October 5, 2025

## Overview

The entity normalization system links extracted clinical entities to standardized medical vocabularies (UMLS, SNOMED-CT, RxNorm, ICD-10, LOINC) for consistent identification and interoperability.

**Purpose:** Map "diabetes" → UMLS:C0011847, SNOMED:73211009, ICD-10:E11

---

## Supported Vocabularies

| Vocabulary | Purpose | Example |
|------------|---------|---------|
| **UMLS** | Comprehensive medical terminology | Diabetes Mellitus (C0011847) |
| **SNOMED-CT** | Clinical terminology standard | Hypertensive disorder (38341003) |
| **RxNorm** | Medication terminology | Metformin (6809) |
| **ICD-10** | Disease classification | Type 2 Diabetes (E11) |
| **LOINC** | Laboratory tests | HbA1c (4548-4) |

---

## Entity Type Mapping

| Entity Type | Primary Vocabularies |
|-------------|---------------------|
| Drug/Medication | RxNorm, UMLS |
| Disease/Disorder | SNOMED-CT, UMLS, ICD-10 |
| Symptom/Sign | SNOMED-CT, UMLS |
| Procedure | SNOMED-CT, UMLS |
| Lab Test | LOINC, UMLS |

---

## How It Works

### Normalization Pipeline

```
Extracted Entity: "niraparib"
    ↓
Vocabulary Lookup (RxNorm + UMLS)
    ↓
Fuzzy Matching (if no exact match)
    ↓
Confidence Scoring
    ↓
Best Match: RxNorm:1658090 (confidence: 0.95)
    ↓
Store: entity_text="niraparib", normalized_id="RXNORM:1658090"
```

### Implementation

**Location:** `scripts/normalize_entities.py`

**Database:**
- **Input:** `docintel.entities` table (37,657 entities)
- **Output:** `normalized_id` column populated
- **Cache:** `data/vocabulary_cache/` (3.2M terms, SQLite)

**Matching Strategy:**
1. Exact string match (case-insensitive)
2. Fuzzy match if exact fails (Levenshtein distance)
3. Multi-vocabulary lookup (entity type determines vocabularies)
4. Confidence scoring (0.0-1.0)

---

## Running Normalization

### Setup Vocabularies

**Download vocabulary sources:**
```bash
# RxNorm
wget https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_<version>.zip
# Extract to data/vocabulary_sources/rxnorm/

# UMLS, SNOMED, ICD-10, LOINC
# Obtain from respective sources, extract to data/vocabulary_sources/
```

**Ingest vocabularies into cache:**
```bash
pixi run -- python scripts/ingest_vocabularies.py
```

This creates `data/vocabulary_cache/*.db` SQLite files with indexed terms.

### Normalize Entities

**Run normalization:**
```bash
pixi run -- python scripts/normalize_entities.py \
    --batch-size 100 \
    --confidence-threshold 0.7
```

**Parameters:**
- `--batch-size`: Entities processed per batch (default: 100)
- `--confidence-threshold`: Minimum match confidence (default: 0.7)
- `--force-renormalize`: Re-normalize already normalized entities

**Monitoring:**
```bash
# Progress
tail -f logs/normalization_progress.log

# Stats
pixi run -- python scripts/monitor_normalization.py
```

---

## Database Schema

**entities table:**
```sql
CREATE TABLE docintel.entities (
    entity_id SERIAL PRIMARY KEY,
    entity_text TEXT NOT NULL,
    entity_type TEXT,
    normalized_id TEXT,           -- "RXNORM:1658090"
    normalization_confidence FLOAT, -- 0.0-1.0
    source_chunk_id INTEGER,
    context_flags JSONB
);
```

**Verification:**
```sql
-- Count normalized entities
SELECT 
    entity_type,
    COUNT(*) AS total,
    COUNT(normalized_id) AS normalized,
    ROUND(COUNT(normalized_id)::NUMERIC / COUNT(*) * 100, 1) AS pct
FROM docintel.entities
GROUP BY entity_type
ORDER BY total DESC;
```

---

## Example Outputs

### Medication Normalization
```
Input: "niraparib"
Match: RxNorm:1658090
Confidence: 0.98
UMLS: C3501483
```

### Disease Normalization
```
Input: "esophageal cancer"
SNOMED: 94756003
ICD-10: C15
UMLS: C0014859
Confidence: 1.0 (exact match)
```

### Lab Test Normalization
```
Input: "HbA1c level"
LOINC: 4548-4
UMLS: C0019018
Confidence: 0.92
```

---

## Configuration

**Environment Variables:**
```bash
# Vocabulary cache location
DOCINTEL_VOCABULARY_CACHE=./data/vocabulary_cache

# Normalization settings
NORMALIZATION_CONFIDENCE_THRESHOLD=0.7
NORMALIZATION_BATCH_SIZE=100
NORMALIZATION_MAX_WORKERS=4
```

**Vocabulary Sources:**
- `data/vocabulary_sources/rxnorm/`
- `data/vocabulary_sources/umls/`
- `data/vocabulary_sources/snomed/`
- `data/vocabulary_sources/icd10/`
- `data/vocabulary_sources/loinc/`

---

## Performance

**Current State (verified Oct 5, 2025):**
- 37,657 entities extracted
- Normalization coverage: Check per entity type
- Cache size: ~3.2M terms indexed
- Lookup speed: <10ms per entity (cached)

**Optimization:**
- SQLite indexes on term columns
- Batch processing (100 entities/batch)
- Multi-worker parallel processing
- Fuzzy matching only when exact fails

---

## Troubleshooting

### Low Match Confidence

**Fix:**
1. Check entity text quality (typos, abbreviations)
2. Lower confidence threshold: 0.7 → 0.6
3. Add synonym mappings for common variations

### Slow Normalization

**Optimize:**
1. Increase batch size: 100 → 500
2. Check vocabulary cache indexes
3. Use more workers: 4 → 8

### Missing Vocabularies

**Verify:**
```bash
ls -lh data/vocabulary_cache/
# Should see: rxnorm.db, umls.db, snomed.db, icd10.db, loinc.db
```

**Rebuild:**
```bash
pixi run -- python scripts/ingest_vocabularies.py --force-rebuild
```

---

## Integration

### Query System
Normalized IDs used in U-Retrieval for entity expansion

### Knowledge Graph
Relations use normalized IDs for subject/object linking

### Export
FHIR/CDISC export uses standard vocabulary codes

---

## Related Documentation

- **System Architecture:** `docs/SYSTEM_ARCHITECTURE.md`
- **U-Retrieval:** `docs/URETRIEVAL_ARCHITECTURE.md`
- **Entity Extraction:** Handled by `src/docintel/extract.py`

---

**Maintained by:** Clinical Trial Knowledge Mining Team
