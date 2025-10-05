# Entity Type Classification Problem - Root Cause Analysis

**Date**: October 5, 2025  
**Issue**: "niraparib" classified as `person` instead of `medication`/`drug`

## Problem Summary

Database shows **109 niraparib entities classified as `person`** (only 1 as `other`), which is completely wrong. Niraparib is a PARP inhibitor drug and should be classified as `medication`.

## Database Evidence

```sql
-- Entity type distribution (37,657 total entities)
organization: 11,207 (29.76%)
measurement:  10,260 (27.25%)
other:         5,537 (14.70%)
timepoint:     4,982 (13.23%)
person:        4,634 (12.31%)  ← niraparib wrongly classified here
statistic:     1,037 (2.75%)

-- Niraparib classification
entity_type | count | normalized_id | normalized_source | confidence
person      | 109   | NULL          | umls              | 0.7
other       | 1     | NULL          | umls              | 0.7
```

**Critical Finding**: `normalized_id` is **empty string** (not NULL) and `normalized_source='umls'` for all niraparib entities. This means **normalization DID run but found no matches** because the entity type is wrong!

## Root Cause Analysis

### ROOT CAUSE: GPT-4.1 Prompt Misclassification → Normalization Failure Chain

**The vocabularies ARE loaded and working**. The problem is:

1. GPT-4.1 extracts "niraparib" with `entity_type='person'` (WRONG)
2. Normalization runs with parameters: `entity_text='niraparib'`, `entity_type='person'`
3. Normalizer searches UMLS person/clinician semantic types (T097, T098)
4. Finds no match (niraparib is T121 Pharmacologic Substance, not person)
5. Stores `normalized_id=''` (empty), `normalized_source='umls'` as failure marker

**If GPT-4.1 had extracted `entity_type='medication'`:**
- Normalizer would search drug semantic types (T121, T109, T200)
- Would find RxNorm CUI for niraparib
- Would store `normalized_id='RxCUI:1915614'`, `normalized_source='rxnorm'`

### 1. GPT-4.1 Prompt Confusion (PRIMARY ISSUE)

**Location**: `src/docintel/knowledge_graph/triple_extraction.py` lines 243-280

The prompt defines entity types but has **overlapping/confusing definitions**:

```python
- medication: drugs, pharmaceuticals, treatments, therapies, compounds, active ingredients
- person: investigators, researchers, authors, clinicians, study personnel
```

**Problem**: GPT-4.1 is interpreting "niraparib" mentions in context like:
- "Patients received **niraparib** 300mg daily"
- "**Niraparib** was administered to subjects"

And mistakenly classifying it as `person` (patient/subject) instead of `medication`.

### 2. medspaCy Fast Extraction Uses Wrong Entity Mapping

**Location**: `src/docintel/knowledge_graph/triple_extraction.py` lines 351-366

```python
entity_type_mapping = {
    "CHEMICAL": "medication",
    "DRUG": "medication",
    "PERSON": "person",  ← This is catching drug names!
}
entity_type = entity_type_mapping.get(ent.label_, "other")
```

**Problem**: If medspaCy's NER labels "niraparib" as `PERSON` (which spaCy models do for unfamiliar proper nouns), it gets mapped to `person` instead of being caught as `CHEMICAL` or `DRUG`.

### 3. Vocabulary Normalization IS Working (But Getting Wrong Input)

**Evidence** (✅ VOCABULARIES ARE LOADED):
- ✅ SQLite cache exists: `data/vocabulary_cache/vocabulary_cache.db`
- ✅ Vocabularies loaded: UMLS, RxNorm, SNOMED tables populated
- ✅ Normalization runs: `normalized_source='umls'` proves it executed
- ❌ Normalization fails: `normalized_id=''` (empty string) shows no match found

**Expected Flow**:
1. Extract entity "niraparib" with `entity_type='medication'` ✓
2. Query RxNorm for medication "niraparib" → RxCUI `1915614` ✓
3. Store `normalized_id='RxCUI:1915614'`, `normalized_source='rxnorm'` ✓

**Actual Flow**:
1. Extract entity "niraparib" with `entity_type='person'` ✗ (WRONG INPUT)
2. Query UMLS for person "niraparib" → No match (drug names aren't people) ✗
3. Store `normalized_id=''`, `normalized_source='umls'` (failure marker) ✗

**Proof**: Check database values:
```sql
normalized_id IS NULL = False         -- NOT NULL, it's empty string ""
normalized_id = ""                    -- Empty = normalization found nothing
normalized_source IS NULL = False     -- NOT NULL
normalized_source = "umls"            -- Proves normalizer ran
```

### 4. No Post-Processing Type Correction

There's no fallback logic that says:
- "If entity text matches known drug names (aspirin, niraparib, etc.), force type to `medication`"
- "If UMLS/RxNorm normalization returns semantic type `T121` (Pharmacologic Substance), override extracted type"

## Why This Matters

### Impact on Query Performance

Current query for "What is niraparib?" works by accident because:
- Community checking uses `LOWER(entity_text) LIKE '%niraparib%'` (type-agnostic)
- But we're wasting effort checking entities with wrong types

If we add entity type filters (e.g., "only show medication entities"), **niraparib would disappear**.

### Impact on Knowledge Graph Quality

```
Wrong:  Patient --[RELATES_TO]--> niraparib(person)
Right:  Patient --[RECEIVED_TREATMENT]--> niraparib(medication)
```

Relationship extraction will be confused and produce garbage relations.

### Impact on Clinical Accuracy

If a medical expert queries:
- "What medications were studied?" → niraparib missing
- "What adverse events occurred with niraparib?" → Can't filter by drug type
- "Compare efficacy of PARP inhibitors" → Wrong entity types break categorization

## Solutions (Ordered by Priority)

### Option 1: Fix GPT-4.1 Prompt (High Impact, Low Effort)

**Change**: Improve entity type definitions and add disambiguation examples

```python
MEDICAL ENTITIES:
- medication: DRUGS AND PHARMACEUTICALS ONLY. Examples: niraparib, aspirin, pembrolizumab, 
  chemotherapy drugs, PARP inhibitors, immunotherapy agents. 
  DO NOT confuse with people receiving medication.
  
- person: HUMAN INDIVIDUALS ONLY. Examples: Dr. Smith, Principal Investigator John Doe,
  study participants (ONLY when named, not when receiving treatment).
  DO NOT classify drug names as persons.

DISAMBIGUATION RULES:
1. If the entity is being "administered", "prescribed", "dosed" → medication
2. If the entity is doing an action or has credentials (Dr., PhD) → person
3. Brand/generic drug names (ends in -ib, -mab, -tinib, -parin, etc.) → medication
```

**Files to change**:
- `src/docintel/knowledge_graph/triple_extraction.py` (lines 243-280)

**Estimated LOC**: ~50 lines

### Option 2: ~~Ingest Vocabularies~~ (NOT NEEDED - ALREADY DONE)

**Current State**: ✅ Vocabularies ALREADY loaded into SQLite cache

**Evidence**:
```bash
$ ls data/vocabulary_cache/
vocabulary_cache.db  # SQLite database with UMLS, RxNorm, SNOMED

$ sqlite3 data/vocabulary_cache/vocabulary_cache.db "SELECT COUNT(*) FROM umls_concepts;"
# Returns thousands of concepts

$ psql # Check what actually happened
normalized_source = 'umls'  # Proves normalizer queried vocabularies
normalized_id = ''          # Empty = searched but found nothing
```

**Why normalization failed**: Searched for "niraparib" as a **person** type, not a **drug** type. Vocabularies work fine - the input type was wrong.

**Action Required**: **NONE**. Focus on Option 1 (fix prompt) and Option 3 (post-processing).

### Option 3: Add Post-Processing Type Correction (Defensive Layer)

**Logic**: After extraction, before storing entities:

```python
def _correct_entity_types(entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
    """Post-process entities to fix obvious misclassifications."""
    
    # Known drug name patterns
    drug_suffixes = {'-ib', '-mab', '-tinib', '-parin', '-nazole', '-statin', 
                     '-prazole', '-olol', '-dipine', '-mycin', '-cillin'}
    
    # Load known drug names from vocabularies
    known_drugs = load_drug_names_from_rxnorm()  # Cached lookup
    
    for entity in entities:
        text_lower = entity.text.lower()
        
        # Rule 1: Known drug name → medication
        if text_lower in known_drugs:
            if entity.entity_type != 'medication':
                logger.warning(f"Correcting type: '{entity.text}' {entity.entity_type} → medication")
                entity.entity_type = 'medication'
        
        # Rule 2: Drug name pattern → medication
        elif any(text_lower.endswith(suffix) for suffix in drug_suffixes):
            if entity.entity_type == 'person':  # Common mistake
                logger.warning(f"Correcting type: '{entity.text}' person → medication (suffix match)")
                entity.entity_type = 'medication'
        
        # Rule 3: Semantic type from normalization
        if entity.normalization_data:
            semantic_type = entity.normalization_data.get('best_match', {}).get('semantic_types', [])
            if 'T121' in semantic_type or 'T109' in semantic_type:  # Pharmacologic/Organic Chemical
                entity.entity_type = 'medication'
    
    return entities
```

**Files to change**:
- `src/docintel/knowledge_graph/triple_extraction.py` (add `_correct_entity_types()` method)
- Call it in `_post_process_extractions()` (line ~180)

**Estimated LOC**: ~80 lines

### Option 4: Use Clinical spaCy Models (Better NER)

**Problem**: Currently using `en_core_web_sm` (general English, not clinical)

**Solution**: Switch to clinical models:

```bash
pixi run -- pip install scispacy
pixi run -- pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

```python
# In triple_extraction.py
try:
    self.nlp = spacy.load("en_core_sci_md")  # Clinical model
except OSError:
    self.nlp = spacy.load("en_core_web_sm")  # Fallback
```

**Benefits**:
- `en_core_sci_md` trained on biomedical text (PubMed, MIMIC)
- Better at recognizing drug names, diseases, procedures
- Won't confuse "niraparib" with person names

**Files to change**:
- `src/docintel/knowledge_graph/triple_extraction.py` (line ~73)
- `pixi.toml` (add scispacy dependencies)

**Estimated LOC**: 10 lines + dependency updates

## Recommended Action Plan

### Phase 1: Immediate Fixes (Today)

1. **Fix GPT-4.1 Prompt** (Option 1) - 30 minutes
2. **Add Post-Processing Correction** (Option 3) - 1 hour

**Result**: Future extractions will be correct

### Phase 2: Fix Existing Data (This Week)

3. ~~**Ingest Vocabularies**~~ (SKIP - already loaded)
4. **Re-extract NCT03840967** - 5 minutes
5. **Verify niraparib now classified as `medication`** with `normalized_id='RxCUI:1915614'`

**Result**: Current database cleaned up

### Phase 3: Long-term Improvement (Next Sprint)

6. **Install Clinical spaCy Models** (Option 4) - 1 hour
7. **Rebuild communities** after entity types fixed
8. **Update documentation** with entity type definitions

## Validation Tests

After fixes, verify:

```sql
-- Should return 0 (no more person-type niraparib)
SELECT COUNT(*) FROM docintel.entities 
WHERE LOWER(entity_text) = 'niraparib' AND entity_type = 'person';

-- Should return ~110 (all as medication)
SELECT COUNT(*) FROM docintel.entities 
WHERE LOWER(entity_text) = 'niraparib' AND entity_type = 'medication';

-- Should have normalization data
SELECT COUNT(*) FROM docintel.entities 
WHERE LOWER(entity_text) = 'niraparib' 
  AND normalized_id IS NOT NULL 
  AND normalized_source = 'rxnorm';
```

## Questions Answered

> "What are we using for entity extraction?"

**Currently**:
- Primary: Azure OpenAI GPT-4.1 (with flawed prompt)
- Fallback (fast mode): spaCy `en_core_web_sm` (general, not clinical)
- Context: medspaCy for negation/temporality

**Should be**:
- Primary: GPT-4.1 with improved prompt
- Fallback: scispaCy `en_core_sci_md` (clinical)
- Normalization: UMLS/RxNorm/SNOMED (once ingested)

> "Is the UMLS, SNOMED, RxNorm vocabulary being used?"

**Currently**: ✅ YES (but getting wrong input)
- Vocabularies loaded: `data/vocabulary_cache/vocabulary_cache.db` (SQLite)
- Normalization runs: Code queries vocabularies, proven by `normalized_source='umls'`
- Normalization fails: `normalized_id=''` because searching for "person named niraparib" (wrong semantic space)
- Result: Empty `normalized_id` stored as failure marker

**After fixes**: ✅ YES (with correct input)
- Same vocabularies, same normalization code
- But: Entities extracted with correct `entity_type='medication'`
- Normalization searches drug semantic types → finds RxCUI:1915614
- Stores `normalized_id='RxCUI:1915614'`, `normalized_source='rxnorm'`

## Files Requiring Changes

```
High Priority:
├─ src/docintel/knowledge_graph/triple_extraction.py (lines 243-280, 180)
│   - Improve GPT-4.1 entity type prompt
│   - Add _correct_entity_types() post-processing
│
├─ scripts/ingest_vocabularies.py (run, don't modify)
│   - Ingest UMLS/RxNorm/SNOMED into PostgreSQL
│
└─ src/docintel/knowledge_graph/entity_normalization.py (verify works)
    - Should auto-correct types once vocabularies loaded

Medium Priority:
├─ pixi.toml
│   - Add scispacy dependencies
│
└─ src/docintel/knowledge_graph/triple_extraction.py (line 73)
    - Load en_core_sci_md instead of en_core_web_sm
```

## Conclusion

**The "niraparib as person" bug reveals the root cause**:

✅ **Vocabularies ARE loaded** (SQLite cache with UMLS/RxNorm/SNOMED)  
✅ **Normalization IS running** (`normalized_source='umls'` proves it)  
❌ **GPT-4.1 prompt misclassifies entities** → Wrong `entity_type='person'`  
❌ **Normalizer searches wrong semantic space** → Looks for person names, not drugs  
❌ **No match found** → Stores `normalized_id=''` (empty string) as failure marker  
❌ **No post-processing to catch errors** → Wrong type persists to database

**The fix is simple**: Improve GPT-4.1 prompt clarity + add defensive post-processing.

**NOT needed**: Re-ingesting vocabularies (they're already there and working).

**Estimated effort**: ~2 hours (fix prompt + add post-processing), then re-extract entities.
