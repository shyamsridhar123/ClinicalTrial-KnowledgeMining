# Query Rewriting Guide

## Overview

The DocIntel query system includes automatic query rewriting to improve semantic search results for short or ambiguous queries. This addresses a known limitation where simple queries produce embeddings with poor discrimination.

## Problem Statement

### The Issue

When using BiomedCLIP embeddings for semantic search:

- **Short queries like "What is niraparib?"** produce embeddings that don't discriminate well between chunks
- All retrieved chunks get similar low relevance scores (e.g., 0.663)
- The system cannot effectively rank chunks to prioritize definitional content
- Results may include administration instructions instead of drug definitions

### Example of the Problem

```bash
# Query: "What is niraparib?"
# All chunks receive identical scores:
[1] NCT: NCT03799627 | Relevance: 0.663
[2] NCT: NCT03799627 | Relevance: 0.663
[3] NCT: NCT02597946 | Relevance: 0.663
...all 0.663

# Answer: Only administration instructions, missing actual definition
```

### Why It Happens

1. The query "What is niraparib?" creates a low-dimensional semantic embedding
2. This embedding matches many chunks weakly but none strongly
3. All chunks containing "niraparib" get similar low scores
4. Without score discrimination, ranking fails
5. Retrieved chunks may not contain the actual definition

## Solution: Query Rewriting

### How It Works

The `QueryRewriter` class automatically detects and expands simple queries:

**Before Rewriting:**
```
"What is niraparib?"
```

**After Rewriting:**
```
"Define niraparib. Niraparib mechanism of action. Niraparib description. What is niraparib."
```

This expanded query produces a richer embedding that:
- Matches definitional content more strongly
- Produces varied relevance scores (0.664, 1.420, 1.231)
- Enables proper ranking of chunks
- Retrieves the actual drug definition

## Supported Query Patterns

### 1. Definitional Queries

**Pattern:** `What is/are X?`

**Condition:** Subject must be 3 words or less (single entity)

**Examples:**

| Original Query | Expanded Query |
|---------------|----------------|
| "What is niraparib?" | "Define niraparib. Niraparib mechanism of action. Niraparib description. What is niraparib." |
| "What are PARP inhibitors?" | "Define PARP inhibitors. PARP inhibitors mechanism of action. PARP inhibitors description. What are PARP inhibitors." |

**Not Rewritten (too complex):**
- "What is the relationship between niraparib and olaparib?" ❌
- "What is the difference between Phase I and Phase II?" ❌

### 2. Mechanism Queries

**Pattern:** `How does X work?`

**Condition:** Subject must be 3 words or less

**Examples:**

| Original Query | Expanded Query |
|---------------|----------------|
| "How does niraparib work?" | "Niraparib mechanism of action. Niraparib mode of action. How does niraparib work. Niraparib pharmacology." |

## Usage

### In query_clinical_trials.py (Automatic)

Query rewriting is **enabled by default**:

```bash
pixi run -- python query_clinical_trials.py "What is niraparib?"
```

Output shows when rewriting occurs:
```
📝 Query expanded for better results:
   Original: 'What is niraparib?'
   Expanded: 'Define niraparib. Niraparib mechanism of action...'
   Reason: Short definitional queries benefit from multiple phrasings
```

### Programmatic Usage

```python
from docintel.query import QueryRewriter

# Create rewriter
rewriter = QueryRewriter(enable_rewriting=True)

# Rewrite query
original = "What is niraparib?"
rewritten = rewriter.rewrite(original)

# Check if rewriting occurred
explanation = rewriter.explain_rewrite(original, rewritten)
if explanation:
    print(explanation)
```

### Disabling Rewriting (for debugging)

```python
# Disable in code
rewriter = QueryRewriter(enable_rewriting=False)

# Or pass through without rewriting
from docintel.query import rewrite_query
result = rewrite_query("What is niraparib?", enable=False)
```

## Results Comparison

### Before Query Rewriting

```bash
Query: "What is niraparib?"
Relevance scores: 0.663, 0.663, 0.663, 0.663, 0.663 (all identical)

Answer: "Niraparib is described as a medication that participants 
should take at approximately the same time each day..."
❌ Missing actual definition and mechanism
```

### After Query Rewriting

```bash
Query expanded: "Define niraparib. Niraparib mechanism of action..."
Relevance scores: 0.664, 1.420, 1.420, 1.231 (varied, good ranking)

Answer: "Niraparib is a synthetic, orally administered small molecule 
that functions as a poly(ADP-ribose) polymerase (PARP) inhibitor..."
✅ Complete definition with mechanism
```

## Implementation Details

### QueryRewriter Class

**Location:** `src/docintel/query/query_rewriter.py`

**Key Methods:**

```python
class QueryRewriter:
    def rewrite(self, query: str) -> str:
        """Main entry point - rewrites query if pattern matches"""
        
    def explain_rewrite(self, original: str, rewritten: str) -> Optional[str]:
        """Generate explanation if query was rewritten"""
        
    def _expand_definitional_query(self, subject: str, is_are: str) -> str:
        """Expand "What is X?" pattern"""
        
    def _expand_mechanism_query(self, subject: str) -> str:
        """Expand "How does X work?" pattern"""
```

### Integration Points

1. **query_clinical_trials.py**: Automatic rewriting in `retrieve_context()`
2. **CLI (src/docintel/cli.py)**: Rewriting happens transparently via query_clinical_trials module
3. **Future UI**: Will inherit rewriting automatically

## Configuration

### Regex Patterns

Defined in `QueryRewriter`:

```python
# Matches: "What is X?" or "What are X?"
WHAT_IS_PATTERN = re.compile(
    r"^what\s+(is|are)\s+(.+?)(?:\?|$)",
    re.IGNORECASE
)

# Matches: "How does X work?"
HOW_DOES_PATTERN = re.compile(
    r"^how\s+does\s+(.+?)\s+work(?:\?|$)",
    re.IGNORECASE
)
```

### Expansion Templates

**Definitional:**
```python
[
    f"Define {subject}.",
    f"{subject} mechanism of action.",
    f"{subject} description.",
    f"What {is_are} {subject}."
]
```

**Mechanism:**
```python
[
    f"{subject} mechanism of action.",
    f"{subject} mode of action.",
    f"How does {subject} work.",
    f"{subject} pharmacology."
]
```

## When to Disable

Consider disabling query rewriting:

1. **Debugging retrieval issues** - see raw query behavior
2. **Testing embedding models** - compare with/without rewriting
3. **Benchmarking** - measure impact on metrics
4. **Already-detailed queries** - user provided comprehensive question

## Future Enhancements

Potential improvements:

1. **Context-aware rewriting**: Detect query intent (safety, efficacy, dosing) and tailor expansions
2. **Query classification**: Use LLM to classify query type before rewriting
3. **User feedback**: Learn which rewrites improve results over time
4. **Domain-specific patterns**: Add patterns for clinical trial specific queries
5. **Multi-language**: Extend to non-English queries

## Testing

### Unit Tests

```bash
# Run query rewriter tests
pixi run pytest tests/test_query_rewriter.py -v
```

### Integration Tests

```bash
# Compare results with/without rewriting
pixi run python -c "
from docintel.query import QueryRewriter

rewriter = QueryRewriter()

# Test cases
queries = [
    'What is niraparib?',
    'What are adverse events?',
    'How does olaparib work?',
    'What is the relationship between X and Y?'  # Should NOT rewrite
]

for q in queries:
    rewritten = rewriter.rewrite(q)
    print(f'Original: {q}')
    print(f'Rewritten: {rewritten}')
    print(f'Changed: {q != rewritten}')
    print()
"
```

## Troubleshooting

### Issue: Query not being rewritten

**Check:**
1. Query matches supported patterns (case-insensitive)
2. Subject is ≤3 words
3. No exclusion keywords ("relationship", "difference", etc.)
4. Rewriting is enabled (`enable_rewriting=True`)

### Issue: Rewriting makes results worse

**Solutions:**
1. File issue with specific query and expected behavior
2. Temporarily disable for that query type
3. Adjust expansion templates in `query_rewriter.py`

## References

- **Implementation**: `src/docintel/query/query_rewriter.py`
- **Integration**: `query_clinical_trials.py` (line ~50-70)
- **Tests**: `tests/test_query_rewriter.py`
- **Issue tracking**: GitHub issue #XXX (query discrimination problem)

## Related Documentation

- [Query Architecture](query_architecture.md) - Overall query system design
- [U-Retrieval Architecture](uretrieval_architecture.md) - Graph-aware retrieval
- [Entity Extraction](Entity_Normalization_Guide.md) - How entities are extracted
- [Evaluation Metrics](Evaluation_Metrics_Guide.md) - Measuring query quality
