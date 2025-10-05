# Documentation Directory

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shyamsridhar123/ClinicalTrial-KnowledgeMining)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen.svg)](README.md)
[![Last Updated](https://img.shields.io/badge/updated-Oct%202025-blue.svg)](README.md)

**Last Updated:** October 5, 2025

## 📘 Start Here

## 📚 Current Documentation



**New to DocIntel?**

1. 📖 [QUICKSTART.md](../QUICKSTART.md) - Get up and running in 5 minutes

2. 📖 [CLI_GUIDE.md](../CLI_GUIDE.md) - Interactive command-line interface## 📘 Start Here### Core Reference

3. 📘 [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) - System overview

- **`Clinical_Trial_Knowledge_Mining_TRD_Modular.md`** - Technical requirements document

---

**New to DocIntel?**- **`clinical-trial-mining-prd (1).md`** - Product requirements document

## 🏗️ Architecture Documentation

1. 📖 [QUICKSTART.md](../QUICKSTART.md) - Get up and running in 5 minutes- **`current_architecture_status.md`** - **Current system status & implementation** ⭐

### Core System

- **[SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)** - Complete system overview, components, data flow ⭐2. 📖 [CLI_GUIDE.md](../CLI_GUIDE.md) - Interactive command-line interface

- **[MODULAR_MAX_STATUS.md](./MODULAR_MAX_STATUS.md)** - Modular MAX/Mojo status (**NOT OPERATIONAL**) ⚠️

- **[Clinical_Trial_Knowledge_Mining_TRD_Modular.md](./Clinical_Trial_Knowledge_Mining_TRD_Modular.md)** - Technical requirements (**ASPIRATIONAL**, not current implementation)3. 📘 [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) - System overview### Implementation Guides



### Query System- **`Entity_Normalization_Guide.md`** - UMLS/LOINC/RxNorm normalization details

- **[QUERY_ARCHITECTURE.md](./QUERY_ARCHITECTURE.md)** - Query pipeline, semantic search, answer generation

- **[QUERY_REWRITING_GUIDE.md](./QUERY_REWRITING_GUIDE.md)** - Automatic query expansion for better results---- **`docling_parsing_architecture.md`** - Document parsing with Docling

- **[URETRIEVAL_ARCHITECTURE.md](./URETRIEVAL_ARCHITECTURE.md)** - Hierarchical graph-aware retrieval

- **`performance_optimizations_implemented.md`** - Optimizations applied

### Processing Pipeline

- **[docling_parsing_architecture.md](./docling_parsing_architecture.md)** - Document parsing with Granite Docling## 🏗️ Architecture Documentation

- **[ENTITY_NORMALIZATION_GUIDE.md](./ENTITY_NORMALIZATION_GUIDE.md)** - Entity linking to UMLS/SNOMED/RxNorm

### Evaluation

---

### Core System- **`Evaluation_Metrics_Guide.md`** - Metrics for system validation

## 🎯 Quick References

- **[SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)** - Complete system overview, components, data flow ⭐- **`user_guide.md`** - End-user documentation

- **[QUERY_REWRITING_QUICKREF.md](./QUERY_REWRITING_QUICKREF.md)** - Query rewriting patterns cheat sheet

- **[CLI_QUICKREF.md](../CLI_QUICKREF.md)** - CLI commands quick reference- **[Clinical_Trial_Knowledge_Mining_TRD_Modular.md](./Clinical_Trial_Knowledge_Mining_TRD_Modular.md)** - Technical requirements document (authoritative reference)

- **[WORKSPACE_STRUCTURE.md](../WORKSPACE_STRUCTURE.md)** - Directory layout

---

---

### Query System

## 📊 Current System State (Verified Oct 5, 2025)

- **[QUERY_ARCHITECTURE.md](./QUERY_ARCHITECTURE.md)** - Query pipeline, semantic search, answer generation## 📁 Archived Documentation

| Component | Count | Status |

|-----------|-------|--------|- **[QUERY_REWRITING_GUIDE.md](./QUERY_REWRITING_GUIDE.md)** - Automatic query expansion for better results

| NCT Studies | 15 | ✅ |

| Embeddings | 3,735 | ✅ |- **[URETRIEVAL_ARCHITECTURE.md](./URETRIEVAL_ARCHITECTURE.md)** - Hierarchical graph-aware retrieval### `archive/` - Historical Analysis

| Entities | 37,657 | ✅ |

| Relations | 5,266 | ✅ |Contains outdated status reports and analysis:

| Vocabulary Terms | 3.2M | ✅ |

### Processing Pipeline- `BRUTAL_HONEST_ANALYSIS.md` - Early system assessment

**Technology Stack:**

- Parsing: Granite Docling SDK + PyTorch CUDA- **[docling_parsing_architecture.md](./docling_parsing_architecture.md)** - Document parsing with Granite Docling- `honest_performance_summary.md` - Historical performance notes

- Embeddings: BiomedCLIP (Hugging Face)

- NLP: medspaCy + scispaCy- **[ENTITY_NORMALIZATION_GUIDE.md](./ENTITY_NORMALIZATION_GUIDE.md)** - Entity linking to UMLS/SNOMED/RxNorm- `Knowledge_Graph_Analysis_Report.md` - Early graph analysis

- LLM: Azure OpenAI GPT-4.1

- Database: PostgreSQL + pgvector

- Environment: Pixi (dependency management)

---### `planning/` - Future Work

**NOT USING:** Modular MAX, Mojo kernels, Mammoth orchestration

Contains design docs for features **not yet implemented**:

---

## 🎯 Quick References- `Community_Detection_Guide.md` - Graph community detection (future)

## 📁 Archive

- `Modular_AI_Acceleration_Integration_Analysis.md` - MAX integration (deferred)

Historical documentation (verbose, obsolete, or superseded versions):

- `archive/current_architecture_status_BROKEN.md` - Duplicated headers, wrong dates- **[QUERY_REWRITING_QUICKREF.md](./QUERY_REWRITING_QUICKREF.md)** - Query rewriting patterns cheat sheet- `u_retrieval_implementation_plan.md` - Hierarchical retrieval (future)

- `archive/query_architecture_VERBOSE.md` - 985 lines → 250 lines (concise version available)

- `archive/uretrieval_architecture_VERBOSE.md` - 1523 lines → 350 lines (concise version available)- **[CLI_QUICKREF.md](../CLI_QUICKREF.md)** - CLI commands quick reference- `medical_graph_rag_analysis.md` - Advanced GraphRAG (future)

- `archive/query_rewriting_guide_VERBOSE.md` - 317 lines → 180 lines (concise version available)

- `archive/Entity_Normalization_Guide_VERBOSE.md` - 337 lines → 200 lines (concise version available)- **[WORKSPACE_STRUCTURE.md](../WORKSPACE_STRUCTURE.md)** - Directory layout- `multimodal_graphrag_analysis.md` - Multimodal enhancements (future)

- `archive/user_guide_REDUNDANT.md` - Duplicate of README + QUICKSTART

- `archive/query_rewriting_implementation_summary_OBSOLETE.md` - Temporary status doc- `Semantic_Chunking_Enhancement.md` - Chunking improvements (future)

- `archive/performance_optimizations_implemented_OBSOLETE.md` - Historical status

- `archive/step4_implementation_complete_OBSOLETE.md` - Completed task doc---



**Note:** Archive contains superseded documentation. Always refer to current versions in main docs/ directory.---



---## 📊 Current System State (Verified Oct 5, 2025)



## 🔧 Technical Deep Dives## 🎯 Quick Reference



### Entity Processing| Component | Count | Status |

- [ENTITY_TYPE_PROBLEM_ANALYSIS.md](./ENTITY_TYPE_PROBLEM_ANALYSIS.md) - Entity type categorization analysis

- [Evaluation_Metrics_Guide.md](./Evaluation_Metrics_Guide.md) - Performance evaluation metrics|-----------|-------|--------|### What's Working Now?



### Recommendations| NCT Studies | 15 | ✅ |See: **`current_architecture_status.md`**

- [EVIDENCE_BASED_RECOMMENDATION.md](./EVIDENCE_BASED_RECOMMENDATION.md) - Evidence-based design decisions

| Embeddings | 3,735 | ✅ |

---

| Entities | 37,657 | ✅ |### How to Use the System?

## 📝 Planning & Future Work

| Relations | 5,266 | ✅ |See: **`../QUICKSTART.md`** (root directory)

Check `planning/` directory for development plans and feature roadmaps.

| Vocabulary Terms | 3.2M | ✅ |

**Note:** Planning docs describe future features, not current implementation. See SYSTEM_ARCHITECTURE.md for actual system state.

### What's the Project Structure?

---

---See: **`../WORKSPACE_STRUCTURE.md`** (root directory)

## ⚠️ Important Notes



### Modular MAX/Mojo Status

**NOT OPERATIONAL** - See [MODULAR_MAX_STATUS.md](./MODULAR_MAX_STATUS.md) for details.## 📁 Archive### How Does Normalization Work?



The TRD mentions Modular MAX, Mojo kernels, and Mammoth orchestration, but these are **aspirational specifications**, not current implementation.See: **`Entity_Normalization_Guide.md`**



**Actual stack:** PyTorch CUDA + Docling SDK + medspaCy + PostgreSQLHistorical documentation (verbose, obsolete, or superseded versions):



### TRD vs Reality- `archive/current_architecture_status_BROKEN.md` - Duplicated headers, wrong dates### What's the Technical Spec?

The Technical Requirements Document (TRD) describes an idealized future system. For actual working architecture, always refer to:

- [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) - Current implementation- `archive/query_architecture_VERBOSE.md` - 985 lines → 250 lines (concise version available)See: **`Clinical_Trial_Knowledge_Mining_TRD_Modular.md`**

- [MODULAR_MAX_STATUS.md](./MODULAR_MAX_STATUS.md) - What's NOT being used

- `archive/uretrieval_architecture_VERBOSE.md` - 1523 lines → 350 lines (concise version available)

---

- `archive/query_rewriting_guide_VERBOSE.md` - 317 lines → 180 lines (concise version available)---

## 🤝 Contributing to Documentation

- `archive/Entity_Normalization_Guide_VERBOSE.md` - 337 lines → 200 lines (concise version available)

When updating documentation:

- `archive/user_guide_REDUNDANT.md` - Duplicate of README + QUICKSTART## 📝 Documentation Status

### Quality Standards

1. **Be concise** - No unnecessary code examples (only critical snippets)- `archive/query_rewriting_implementation_summary_OBSOLETE.md` - Temporary status doc

2. **Be accurate** - Verify all claims against actual database/code

3. **Be clear** - Crystal clear architecture descriptions, no ambiguity- `archive/performance_optimizations_implemented_OBSOLETE.md` - Historical status✅ **Current & Accurate**:

4. **Reference sources** - Link to TRD, vendor docs, cite actual documentation

5. **Use tables** - Present data clearly and scannable- `archive/step4_implementation_complete_OBSOLETE.md` - Completed task doc- current_architecture_status.md

6. **Include dates** - Last updated timestamps on every doc

7. **Keep it current** - Archive obsolete versions to `archive/`- Entity_Normalization_Guide.md



### Verification Requirements**Note:** Archive contains superseded documentation. Always refer to current versions in main docs/ directory.- docling_parsing_architecture.md

- **Database counts** → Use PostgreSQL MCP tools to verify

- **Performance claims** → Back with actual measurements from logs

- **API details** → Verify against source code, never guess

- **Technology claims** → Verify what's actually installed and running---⚠️ **Partially Outdated** (predates recent fixes):



### Archiving Guidelines- Clinical_Trial_Knowledge_Mining_TRD_Modular.md (Modular integration not implemented)

When creating new version of existing doc:

1. Move old version to `archive/` with suffix: `_VERBOSE`, `_OBSOLETE`, `_BROKEN`, or `_REDUNDANT`## 🔧 Technical Deep Dives- user_guide.md (some features not yet available)

2. Create deprecation notice in old location (if keeping redirect)

3. Update README.md to point to new version- performance_optimizations_implemented.md (check current_architecture_status.md for reality)

4. Update all cross-references in other docs

### Entity Processing

---

- [ENTITY_TYPE_PROBLEM_ANALYSIS.md](./ENTITY_TYPE_PROBLEM_ANALYSIS.md) - Entity type categorization analysis❌ **Future/Planning Only** (in `planning/`):

## 📞 Support

- [Evaluation_Metrics_Guide.md](./Evaluation_Metrics_Guide.md) - Performance evaluation metrics- All docs in planning/ describe unimplemented features

For questions about:

- **Architecture:** See [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)

- **Usage:** See [QUICKSTART.md](../QUICKSTART.md) and [CLI_GUIDE.md](../CLI_GUIDE.md)

- **Development:** See actual implementation, NOT the aspirational TRD### Recommendations---

- **Troubleshooting:** Check relevant architecture docs

- **Modular MAX/Mojo:** See [MODULAR_MAX_STATUS.md](./MODULAR_MAX_STATUS.md) - NOT OPERATIONAL- [EVIDENCE_BASED_RECOMMENDATION.md](./EVIDENCE_BASED_RECOMMENDATION.md) - Evidence-based design decisions



---Last updated: October 2, 2025



**Maintained by:** Clinical Trial Knowledge Mining Team  ---

**Repository:** https://github.com/shyamsridhar123/ClinicalTrial-KnowledgeMining

## 📝 Planning & Future Work

Check `planning/` directory for development plans and feature roadmaps.

**Note:** Planning docs describe future features, not current implementation. See SYSTEM_ARCHITECTURE.md for actual system state.

---

## 🤝 Contributing to Documentation

When updating documentation:

### Quality Standards
1. **Be concise** - No unnecessary code examples (only critical snippets)
2. **Be accurate** - Verify all claims against actual database/code
3. **Be clear** - Crystal clear architecture descriptions, no ambiguity
4. **Reference sources** - Link to TRD, vendor docs, cite https://docs.modular.com/
5. **Use tables** - Present data clearly and scannable
6. **Include dates** - Last updated timestamps on every doc
7. **Keep it current** - Archive obsolete versions to `archive/`

### Verification Requirements
- **Database counts** → Use PostgreSQL MCP tools to verify
- **Performance claims** → Back with actual measurements from logs
- **API details** → Verify against source code, never guess
- **Modular features** → Always cite official Modular documentation

### Archiving Guidelines
When creating new version of existing doc:
1. Move old version to `archive/` with suffix: `_VERBOSE`, `_OBSOLETE`, `_BROKEN`, or `_REDUNDANT`
2. Create deprecation notice in old location (if keeping redirect)
3. Update README.md to point to new version
4. Update all cross-references in other docs

---

## 📞 Support

For questions about:
- **Architecture:** See [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)
- **Usage:** See [QUICKSTART.md](../QUICKSTART.md) and [CLI_GUIDE.md](../CLI_GUIDE.md)
- **Development:** See [Clinical_Trial_Knowledge_Mining_TRD_Modular.md](./Clinical_Trial_Knowledge_Mining_TRD_Modular.md)
- **Troubleshooting:** Check relevant architecture docs

---

**Maintained by:** Clinical Trial Knowledge Mining Team  
**Repository:** https://github.com/shyamsridhar123/ClinicalTrial-KnowledgeMining
