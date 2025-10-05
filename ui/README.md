# DocIntel Gradio UI

Web-based interface for the Clinical Trial Knowledge Mining Platform.

## Features

### 📊 System Health
- Real-time health checks for all components
- GPU, Azure OpenAI, PostgreSQL, Model Cache status
- Visual color-coded status indicators

### 📥 Data Pipeline
- **Ingestion**: Download clinical trial documents from ClinicalTrials.gov
- **Parsing**: Extract text, tables, figures with Docling
- **Embeddings**: Generate BiomedCLIP semantic embeddings

### 🔍 Query & Analysis
- **Semantic Search**: Ask questions in plain English with context-aware answers
- **Graph Statistics**: View entity and relation counts
- **Context Tests**: Validate extraction accuracy (83% pass rate)

### 🎨 Context Flag Annotations
All query results show clinical context flags:
- ❌ **NEGATED**: "no evidence of hepatotoxicity"
- 📅 **HISTORICAL**: past conditions
- 🤔 **HYPOTHETICAL**: "if condition worsens"
- ❓ **UNCERTAIN**: "possible adverse event"
- 👨‍👩‍👧 **FAMILY**: family history
- ✓ **Active**: actual finding

## Quick Start

### Launch the UI

```bash
cd /home/shyamsridhar/code/docintel
pixi run python ui/app.py
```

Then open your browser to: **http://localhost:7860**

### Using the UI

1. **Check System Health** (Tab 1)
   - Click "Run Health Check"
   - Verify all components are green ✅

2. **Run Pipeline** (Tabs 2-4)
   - Ingestion → Parsing → Embeddings
   - Or use CLI Option 10 for full pipeline

3. **Ask Questions** (Tab 5)
   - Enter: "What adverse events occurred with niraparib?"
   - Get context-aware answers with source annotations

4. **Validate Accuracy** (Tab 7)
   - Run context tests
   - Verify 83%+ pass rate

## Architecture

The UI is a thin wrapper around the existing CLI functionality:

```
ui/app.py
  ↓
docintel.ingest, docintel.parse, docintel.embed
  ↓
docintel.health.HealthCheckRunner
  ↓
query_clinical_trials.ClinicalTrialQA
  ↓
PostgreSQL + Apache AGE + pgvector
```

## Configuration

Uses the same `.env` file as the CLI:

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1

# PostgreSQL
DOCINTEL_VECTOR_DB_DSN=postgresql://user:pass@localhost:5432/docintel

# Storage
DOCINTEL_STORAGE_ROOT=./data/ingestion
DOCINTEL_PROCESSED_STORAGE_ROOT=./data/processing
```

## Deployment

### Local Development
```bash
pixi run python ui/app.py
```

### Production (with SSL)
```bash
pixi run python ui/app.py --server-name 0.0.0.0 --server-port 443 --ssl-certfile cert.pem --ssl-keyfile key.pem
```

### Docker
```bash
# Build
docker build -t docintel-ui -f ui/Dockerfile .

# Run
docker run -p 7860:7860 --env-file .env docintel-ui
```

## Troubleshooting

### UI won't start
- Check pixi environment: `pixi run python --version`
- Install gradio: `pixi add gradio`

### Health check shows errors
- Run CLI Option 11 to diagnose
- Check `.env` configuration
- Verify PostgreSQL is running

### Queries return no results
- Ensure pipeline has run (Tabs 2-4)
- Check Tab 6 for entity/relation counts
- Verify embeddings were generated

### Context tests failing
- Expected: 83% pass rate (5/6 tests)
- One known minor failure (family history entity name)

## Features vs CLI

| Feature | CLI | Gradio UI |
|---------|-----|-----------|
| System health check | ✅ Option 11 | ✅ Tab 1 |
| Data ingestion | ✅ Option 1 | ✅ Tab 2 |
| Document parsing | ✅ Option 2 | ✅ Tab 3 |
| Embedding generation | ✅ Option 3 | ✅ Tab 4 |
| Semantic search | ✅ Option 7 | ✅ Tab 5 |
| Graph statistics | ✅ Option 6 | ✅ Tab 6 |
| Context tests | ✅ Option 9 | ✅ Tab 7 |
| Full pipeline | ✅ Option 10 | ⚠️ Use CLI |
| Entity extraction | ✅ Option 4 | ⚠️ Use CLI |
| Graph construction | ✅ Option 5 | ⚠️ Use CLI |
| Advanced Cypher | ✅ Option 8 | ⚠️ Use CLI |

## Performance

- **Health Check**: ~3 seconds
- **Semantic Search**: ~1-2 seconds per query
- **Ingestion**: Varies by document count
- **Parsing**: GPU-accelerated (≤10 min for 3000 pages)

## Security

- No authentication by default (add with `auth` parameter)
- Runs on localhost by default
- Use SSL for production deployments
- Never expose API keys in UI

## Contributing

To add new features:

1. Add function in `ui/app.py`
2. Create new tab in Gradio interface
3. Wire up button/input handlers
4. Test with `pixi run python ui/app.py`

## Support

- **CLI Documentation**: `CLI_GUIDE.md`
- **Architecture**: `docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md`
- **Issues**: Check health check tab first
