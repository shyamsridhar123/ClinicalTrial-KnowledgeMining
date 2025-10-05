# Model Router - Simplified Routing Logic

## Overview

The model router intelligently selects between GPT-4.1 and GPT-5-mini based on **query characteristics**, not token counts. This simplifies the logic and makes routing decisions more predictable.

## Routing Rules (Priority Order)

### 1. **Images Present → GPT-5-mini**
```python
if has_images:
    return GPT_5_MINI  # Multimodal capability
```
**When:** Query includes image attachments (base64 or file paths)  
**Why:** GPT-5-mini has multimodal vision capabilities  
**Example:** "Analyze this efficacy chart" + image attachment

---

### 2. **Visual Keywords → GPT-5-mini**
```python
visual_keywords = ["figure", "chart", "diagram", "image", "graph", 
                   "plot", "table", "visualization", "flowchart", 
                   "schematic", "survival curve", "kaplan-meier"]

if any(keyword in query.lower() for keyword in visual_keywords):
    return GPT_5_MINI  # Visual reasoning
```
**When:** Query mentions figures, charts, diagrams  
**Why:** Suggests visual analysis even if image not yet attached  
**Example:** "What does Figure 3 show in the efficacy section?"

---

### 3. **Reasoning Keywords → GPT-5-mini**
```python
reasoning_keywords = ["analyze", "compare", "reasoning", "step by step",
                      "explain why", "justify", "rationale", "evidence"]

if any(keyword in query.lower() for keyword in reasoning_keywords):
    return GPT_5_MINI  # Advanced reasoning
```
**When:** Query requires multi-step analysis or justification  
**Why:** GPT-5-mini has enhanced reasoning capabilities  
**Example:** "Analyze step by step why the trial failed to meet its endpoint"

---

### 4. **Default → GPT-4.1**
```python
return GPT_4_1  # Cost-optimized for simple text queries
```
**When:** None of the above conditions match  
**Why:** GPT-4.1 is cheaper and sufficient for simple text Q&A  
**Example:** "What is the primary endpoint of NCT02826161?"

---

## Model Comparison

| Feature | GPT-4.1 | GPT-5-mini |
|---------|---------|------------|
| **Context Window** | 1,047,576 tokens (~1M) | 400,000 tokens |
| **Vision (Images)** | ✅ Yes | ✅ Yes |
| **Reasoning** | Good | **Enhanced** |
| **Temperature Control** | ✅ Yes | ❌ No (fixed at 1.0) |
| **Cost** | Lower | Higher |
| **Best For** | Simple text Q&A, large documents | Multimodal, advanced reasoning |

---

## Implementation

### Basic Usage
```python
from docintel.query.model_router import get_router
from docintel.query.llm_client import get_llm_client

router = get_router()
client = get_llm_client()

# Route query
decision = router.route(
    query_text="What does Figure 3 show?",
    images=["figure3.png"]  # Optional
)

print(f"Model: {decision.model.value}")
print(f"Reason: {decision.reason}")

# Call selected model
response = client.query_with_images(
    text="Analyze this chart",
    images=["chart.png"],
    model=decision.model,
    max_tokens=2000
)

print(response.content)
```

### Force Specific Model
```python
# Override routing decision
decision = router.route(
    query_text="Simple query",
    force_model=ModelChoice.GPT_5_MINI
)
```

### Check Routing Statistics
```python
stats = router.get_routing_stats()

print(f"Total queries: {stats['total_queries']}")
print(f"GPT-4.1: {stats['gpt_4_1_percentage']:.1f}%")
print(f"GPT-5-mini: {stats['gpt_5_mini_percentage']:.1f}%")
print(f"Reasons: {stats['routing_reasons']}")
```

---

## API Parameter Handling

The LLM client automatically normalizes parameters:

### GPT-4.1
```python
response = client.query(
    messages=messages,
    model=ModelChoice.GPT_4_1,
    max_tokens=4096,        # Uses 'max_tokens'
    temperature=0.7          # Supports temperature
)
```

### GPT-5-mini
```python
response = client.query(
    messages=messages,
    model=ModelChoice.GPT_5_MINI,
    max_tokens=4096,        # Converted to 'max_completion_tokens'
    temperature=0.7          # Ignored (fixed at 1.0 for reasoning models)
)
```

---

## Expected Distribution

Based on typical clinical trial queries:

- **GPT-4.1:** ~60-70% (simple text queries, summaries)
- **GPT-5-mini:** ~30-40% (figures, advanced reasoning, comparisons)

This distribution optimizes for cost while preserving advanced capabilities where needed.

---

## Testing

Run the test suite:
```bash
pixi run python test_model_router.py
```

All 14 tests validate:
- ✅ Routing logic (simple text, images, keywords)
- ✅ Token counting utilities
- ✅ Edge cases (empty inputs, forced overrides)
- ✅ Statistics tracking

---

## Files

- **`src/docintel/query/model_router.py`** - Routing logic
- **`src/docintel/query/llm_client.py`** - Unified API client
- **`src/docintel/query/token_utils.py`** - Token counting (optional)
- **`test_model_router.py`** - Test suite
- **`examples/model_router_example.py`** - Usage examples

---

## Notes

- Token-based routing removed for simplicity
- Context window limits handled by Azure API (will return error if exceeded)
- Visual keywords include clinical trial-specific terms (Kaplan-Meier, survival curves)
- Reasoning keywords detect analytical queries
- All routing decisions logged for cost analysis
