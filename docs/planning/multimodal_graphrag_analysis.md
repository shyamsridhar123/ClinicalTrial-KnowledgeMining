# CRITICAL ANALYSIS: Tables & Figures in GraphRAG Application

## Current State (What You Actually Have)

### ✅ What's Embedded:
1. **3,214 text chunk embeddings** - Regular document text
2. **284 table embeddings** - Tables embedded as TEXT (not images)
3. **25 figure caption embeddings** - Only the caption TEXT (not the figure images)

### ⚠️ What's NOT Embedded:
1. **212 figure IMAGES** (PNG files in `data/processing/figures/`) - **NOT embedded**
2. **Table visual representations** - Only text structure embedded, not visual layout

### 🔍 Evidence:
```
Content types in embeddings database: None: 3523
```
This confirms: **NO image embeddings were created**, only text embeddings.

---

## The Fundamental Problem

### Your BiomedCLIP Setup:
- **Model**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **Capabilities**: 
  - ✅ Text encoder (PubMedBERT) - BEING USED
  - ✅ Vision encoder (ViT) - **NOT BEING USED**
  - ✅ Multimodal alignment - **NOT BEING LEVERAGED**

### What's Actually Happening:
```
Figure with chart → Docling extracts caption → Embed caption text → Store in pgvector
Table with data   → Docling extracts structure → Embed table text → Store in pgvector
```

### What SHOULD Happen for GraphRAG:
```
Figure with chart → Extract image file → Embed IMAGE with ViT encoder → Store in pgvector
Table with data   → Extract both text AND visual → Embed both → Link in graph
```

---

## GraphRAG Application Implications

### Scenario 1: User asks "Show me survival curves from breast cancer trials"

**Current Setup (Text-only embeddings):**
```
Query: "survival curves breast cancer"
  ↓
Semantic search on text embeddings
  ↓
Match: "figure_caption: Kaplan-Meier survival analysis for breast cancer cohort"
  ↓
Return: Caption text only
  ↓
LLM sees: "Figure 5 shows survival curves for breast cancer patients"
  ↓
Problem: LLM has NO ACCESS to the actual chart/image
  ↓
Result: LLM can only describe what the caption says, not analyze the actual data
```

**What You NEED (Multimodal embeddings):**
```
Query: "survival curves breast cancer"
  ↓
Semantic search on text + image embeddings
  ↓
Match: Image embedding of actual survival curve chart
  ↓
Retrieve: Image file path + caption + metadata
  ↓
Send to Vision LLM (GPT-4V/Gemini): Image + context
  ↓
LLM analyzes: Actual curve shape, median survival, confidence intervals, p-values
  ↓
Result: "The survival curve shows median survival of 18 months with HR=0.65, p<0.001"
```

### Scenario 2: User asks "What were the Grade 3+ adverse events?"

**Current Setup (Table text embeddings):**
```
Query: "Grade 3 adverse events"
  ↓
Match: Table embedding (whole table as text)
  ↓
Return: Markdown/JSON representation of table
  ↓
LLM reads: Structured data from 140K table_data entities
  ↓
Result: ✅ THIS WORKS - LLM can read structured table data
```

**Why it works for tables:**
- Tables are inherently structured data
- Text representation is complete (all cells, rows, headers)
- LLM doesn't need to "see" the table, just read the data
- Your 140K table entities capture all the details

---

## Architectural Decision Matrix

### Option 1: Keep Current Setup (Text-only)
**Works for:**
- ✅ Document text search
- ✅ Table data queries (via structured text)
- ✅ Finding relevant sections

**Fails for:**
- ❌ Analyzing charts/graphs visually
- ❌ Understanding complex diagrams
- ❌ Reading text from images in figures
- ❌ Analyzing visual patterns in data

**Use case coverage:** ~70% of clinical trial queries

### Option 2: Add Image Embeddings (Multimodal)
**Implementation:**
```python
# Embed figure images with BiomedCLIP vision encoder
for figure_png in figure_images:
    image_embedding = biomedclip.embed_images([figure_png])
    store_embedding(
        embedding=image_embedding,
        artefact_type="figure_image",
        source_path=figure_png,
        linked_caption_id=caption_chunk_id
    )
```

**Works for:**
- ✅ All text queries
- ✅ All table queries
- ✅ Visual analysis of charts/graphs
- ✅ Multimodal retrieval (find images similar to text query)
- ✅ Vision LLM can analyze actual figures

**Use case coverage:** ~95% of clinical trial queries

### Option 3: Hybrid Approach (Recommended)
**Strategy:**
1. Keep current text/table embeddings for structured queries
2. Add image embeddings for figures/charts
3. At retrieval time, return BOTH:
   - Text embedding match → Structured data
   - Image embedding match → Image file
4. Use appropriate LLM:
   - Text-only LLM (GPT-4) for structured data
   - Vision LLM (GPT-4V) when images retrieved

---

## LLM Inference Architecture

### Current Architecture (What you probably have):
```
User query
  ↓
Vector search → Find text embeddings
  ↓
Retrieve chunks + entities from graph
  ↓
Pass text to GPT-4
  ↓
Generate answer
```

**Limitation:** GPT-4 never sees the actual figures/charts

### Recommended Architecture (Multimodal RAG):
```
User query
  ↓
Vector search → Find text AND image embeddings
  ↓
Retrieve:
  - Text chunks
  - Table structured data (from 140K entities)
  - Figure image files (PNG)
  - Graph context (entities, relations)
  ↓
Intelligent routing:
  - If images retrieved → Use GPT-4V/Gemini with vision
  - If only text/tables → Use GPT-4 standard
  ↓
Context assembly:
  - Text: "Table 3 shows adverse events..."
  - Structured: {grade_3: 15%, grade_4: 3%}
  - Image: <figure_image.png>
  - Graph: [normalized entities: pembrolizumab (RxNorm:1234)]
  ↓
Generate answer with full context
```

---

## Specific Recommendations

### For Tables:
**Current setup is FINE** ✅
- You have 140K table entities with structured data
- Table text embeddings (284) for semantic search
- LLM can read all the cell values from entities
- No need for table images unless visual layout matters

**Action:** Nothing needed, your approach works

### For Figures/Charts:
**Current setup is INCOMPLETE** ⚠️
- You have 212 figure PNG files
- Only 25 caption text embeddings
- NO image embeddings of the actual figures

**Action Required:**
1. **Embed the 212 figure images** using BiomedCLIP vision encoder
2. Store image embeddings in pgvector with `artefact_type='figure_image'`
3. Link image embeddings to caption embeddings in graph
4. At retrieval time, return both caption AND image file path
5. Use vision-capable LLM (GPT-4V, Gemini) to analyze retrieved images

**Implementation snippet:**
```python
# Add to embedding phase
async def embed_figure_images(self, nct_id: str, document_name: str):
    figure_dir = self.processing_layout.figures / nct_id / document_name.replace('.json', '')
    figure_images = list(figure_dir.glob("*.png"))
    
    if not figure_images:
        return []
    
    # Use BiomedCLIP vision encoder
    image_embeddings = await self.client.embed_images(figure_images)
    
    records = []
    for img_path, emb_response in zip(figure_images, image_embeddings):
        records.append(EmbeddingRecord(
            chunk_id=f"figure-image-{img_path.stem}",
            embedding=emb_response.embedding,
            metadata={
                "artefact_type": "figure_image",
                "content_type": "image",
                "source_image_path": str(img_path),
                "nct_id": nct_id,
                "document_name": document_name,
            }
        ))
    
    return records
```

### For LLM Inference:
**Current setup needs VISION capability** ⚠️

**Action Required:**
1. Switch from GPT-4 to **GPT-4V** (or GPT-4o with vision)
2. Modify retrieval to include image file paths
3. Pass images to vision LLM along with text context

**Azure OpenAI Update:**
```python
# Your current: GPT-4.1
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1

# Change to: GPT-4V or GPT-4o (vision-enabled)
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o  # Has vision + multimodal
```

---

## Bottom Line Answers

### "Will my LLM inference the table or figure details?"

**Tables:** ✅ YES
- LLM receives structured data from 140K table entities
- Can read all cell values, headers, data
- Text representation is complete

**Figures:** ❌ NO (currently)
- LLM only sees caption text ("Figure 5 shows survival curves")
- LLM does NOT see the actual chart/graph
- Cannot analyze curve shapes, read values from axes, etc.

### "Should we consider something else at the fundamental level?"

**YES - Add image embeddings for figures:**
1. You have BiomedCLIP with vision encoder - USE IT
2. You have 212 figure PNG files - EMBED THEM
3. Update to vision-capable LLM (GPT-4o/GPT-4V)
4. Modify retrieval to return image paths
5. Pass images to vision LLM for analysis

**This is NOT a huge architectural change:**
- Your graph structure is fine
- Your table approach is fine
- Just add: Image embedding generation + Vision LLM inference

---

## Priority Action Items

1. **HIGH PRIORITY:** Add figure image embeddings
   - Run embedding phase on 212 figure PNG files
   - Store in pgvector with `content_type='image'`
   - Estimated: 2-3 hours work

2. **HIGH PRIORITY:** Update LLM to vision-capable model
   - Change deployment to GPT-4o or GPT-4V
   - Modify inference code to pass images
   - Estimated: 1-2 hours work

3. **MEDIUM PRIORITY:** Update retrieval logic
   - Return image file paths for figure matches
   - Assemble multimodal context (text + images)
   - Estimated: 2-3 hours work

**Total effort:** 1-2 days to full multimodal GraphRAG

---

## Test Queries to Validate

### After implementing image embeddings:

**Query 1:** "Show me Kaplan-Meier survival curves"
- Should retrieve: Figure images of survival curves
- LLM should: Analyze actual curves, read median survival, p-values

**Query 2:** "What do the dose-response charts show?"
- Should retrieve: Chart images showing dose-response
- LLM should: Describe trends, read specific values from graph

**Query 3:** "Grade 3+ adverse events percentage"
- Should retrieve: Table structured data
- LLM should: Read exact percentages from table entities

**All three should work seamlessly in a single GraphRAG application.**
