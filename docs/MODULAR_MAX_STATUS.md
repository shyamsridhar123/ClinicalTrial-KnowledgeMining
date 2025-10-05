# Modular MAX/Mojo Status

**Last Updated:** October 5, 2025  
**Status:** NOT OPERATIONAL

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shyamsridhar123/ClinicalTrial-KnowledgeMining)
[![MAX Status](https://img.shields.io/badge/Modular%20MAX-NOT%20OPERATIONAL-red.svg)](MODULAR_MAX_STATUS.md)
[![Alternative](https://img.shields.io/badge/using-PyTorch%20CUDA-green.svg)](MODULAR_MAX_STATUS.md)

---

## Executive Summary

**Modular MAX, Mojo kernels, and Mammoth orchestration are NOT currently used in the DocIntel system.**

The TRD (Technical Requirements Document) was written with aspirational goals for Modular acceleration, but the production system uses alternative proven technologies.

---

## Current Production Stack

| Component | TRD Specified | Actually Using | Status |
|-----------|---------------|----------------|--------|
| **Document Parsing** | Modular MAX + Granite Docling | **Docling SDK directly** | ✅ Working |
| **GPU Acceleration** | Mojo kernels | **PyTorch CUDA** | ✅ Working |
| **Embeddings** | MAX-served models | **Direct BiomedCLIP** | ✅ Working |
| **Orchestration** | Mammoth | **Docker Compose** | ✅ Working |
| **NLP Kernels** | Mojo custom kernels | **medspaCy + scispaCy** | ✅ Working |

---

## Why MAX/Mojo Not Used

### 1. Docling SDK Path Faster
> "The parsing CLI invokes Granite Docling directly through the SDK rather than Modular MAX. This path proved faster and more reliable for document-heavy workloads."
> 
> — README.md

**Reality:** Direct SDK integration eliminated HTTP overhead and provided more granular control.

### 2. Complexity vs Benefit
- MAX requires additional server process (`max serve`)
- OpenAI-compatible wrapper adds latency
- Direct SDK integration simpler to debug and monitor

### 3. Maturity of Alternatives
- PyTorch CUDA acceleration mature and well-documented
- medspaCy/scispaCy proven for clinical NLP
- Docker Compose sufficient for current scale (15 NCTs)

---

## What IS Using Modular Technologies

### Pixi Package Manager ✅
**Still using Pixi** for environment management:
```bash
pixi run -- python -m docintel.cli
pixi shell
```

**Why:** Reproducible environments, per Modular best practices.

**Note:** Pixi is a standalone tool, doesn't require MAX/Mojo runtime.

---

## TRD vs Reality

### TRD Claims (Aspirational)
- ❌ "Serve Granite Docling via `max serve`" — Not implemented
- ❌ "Mojo kernels for UMLS matching" — Using Python/NumPy instead
- ❌ "Mammoth autoscaling" — Using Docker for orchestration
- ❌ "Sub-10-minute processing for 3000-page documents" — Not achieved yet (performance varies)

### Actual Implementation
- ✅ Direct Docling SDK with GPU acceleration
- ✅ PyTorch CUDA for matrix operations
- ✅ BiomedCLIP embeddings via Hugging Face Transformers
- ✅ PostgreSQL + pgvector for storage
- ✅ medspaCy/scispaCy for clinical NLP

---

## Files Containing Outdated MAX/Mojo References

### Documentation
- `docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md` — Aspirational spec, not current implementation
- `docs/docling_parsing_architecture.md` — References MAX but notes "optional"
- `docs/Evaluation_Metrics_Guide.md` — Mentions MAX in performance sections
- `docs/clinical-trial-mining-prd (1).md` — Product requirements (aspirational)

### Code (Unused/Historical)
- `src/docintel/validate_chat.py` — MAX endpoint validator (unused)
- `src/docintel/parsing/client.py` — Comments mention MAX (code uses SDK)
- `pixi.lock` — Mojo packages installed but unused

### Main Documentation
- `README.md` — States "rather than Modular MAX" correctly
- `CLI_GUIDE.md` — Troubleshooting section mentions MAX (outdated)

---

## Recommended Actions

### For Users
**Ignore all references to:**
- Modular MAX / `max serve`
- Mojo kernels
- Mammoth orchestration

**Use instead:**
- Direct CLI commands via `pixi run`
- Docker for database services
- Docling SDK for parsing

### For Developers
**When updating docs:**
1. Remove MAX/Mojo from operational architecture
2. Mark TRD as "aspirational/future work"
3. Document actual tech stack (PyTorch, Docling SDK, etc.)

**When writing new code:**
- Don't add MAX/Mojo dependencies
- Use PyTorch for GPU acceleration
- Use Docling SDK directly

---

## Future Considerations

### If MAX/Mojo Considered Again

**Evaluate:**
- Is MAX deployment simpler than current SDK approach?
- Does Mojo provide measurable performance benefit over PyTorch?
- Is the added complexity justified by scale requirements?

**Requirements for adoption:**
- Benchmark MAX vs SDK on realistic workloads
- Document deployment and troubleshooting procedures
- Train team on Mojo development and debugging
- Verify regulatory compliance (FDA 21 CFR Part 11, GxP)

**Current verdict:** Defer until system scales beyond 50+ NCTs and current performance becomes bottleneck.

---

## Summary

✅ **Pixi** - Using for environment management  
❌ **Modular MAX** - Not operational, not in production  
❌ **Mojo kernels** - Not implemented, using Python/PyTorch  
❌ **Mammoth** - Not implemented, using Docker  

**Production reality:** PyTorch + Docling SDK + PostgreSQL + Docker

**TRD status:** Aspirational specification, not current implementation

---

**Maintained by:** Clinical Trial Knowledge Mining Team  
**Questions?** See actual architecture in `SYSTEM_ARCHITECTURE.md`
