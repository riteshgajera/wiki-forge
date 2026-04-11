# 🧠 Wiki Forge — Local-First Multi-Agent Knowledge Base

> Inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
> The wiki is a **persistent, compounding artifact** — not a RAG index.

A production-grade, autonomous LLM-powered system that compiles raw documents into a structured, self-maintaining Obsidian-compatible markdown wiki.

**The key insight:** When you add a new source, the LLM doesn't just create one new page — it reads the existing wiki and updates 10–15 pages: enriching entity records, adding cross-references, flagging contradictions, and strengthening the knowledge graph. Knowledge accumulates. Nothing needs to be re-derived on every query.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourname/wiki-forge
cd wiki-forge
pip install -e ".[dev]"
cp .env.example .env          # edit LLM settings
```

### 2. Configure your LLM

**Local (Ollama — recommended, free, private):**
```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2
ollama pull nomic-embed-text
# .env already defaults to ollama — nothing else needed
```

**OpenAI API:**
```env
LLM__PROVIDER=openai
LLM__MODEL=gpt-4o-mini
LLM__OPENAI_API_KEY=sk-...
```

### 3. Add documents and compile

```bash
# Drop any documents into /raw (PDF, Markdown, text, images)
cp my-papers/*.pdf   raw/papers/
cp my-notes/*.md     raw/articles/

# Step 1: Scan and fingerprint new files
kb ingest

# Step 2: Run the full agent pipeline → generates /wiki
kb compile

# Step 3: Check quality
kb lint
```

### 4. Search and query

```bash
# Keyword + semantic hybrid search
kb query "attention mechanism"

# AI-synthesised answer
kb query "How does BERT differ from the original Transformer?" --generate

# File the answer back into the wiki (it compounds)
kb query "Compare BERT and GPT architectures" --generate --file
```

### 5. Open in Obsidian

Set your Obsidian vault to `./wiki`. The wiki uses Obsidian-compatible:
- `[[wikilinks]]` for cross-references
- YAML frontmatter for metadata and Dataview queries
- Tags for filtering
- Backlinks populated automatically

### 6. Start the web UI

```bash
uvicorn web.main:app --reload
# → http://localhost:8000
```

---

## The Three Layers

```
/raw/          Raw sources (immutable — LLM reads, never writes)
  articles/    Markdown articles, blog posts
  papers/      Research PDFs
  images/      Diagrams, screenshots (with OCR)
  repos/        Code repositories

/wiki/         LLM-generated wiki (LLM writes, you read)
  WIKI.md      Governing schema — read this first every session
  index.md     Content catalog with L0 summaries (for fast retrieval)
  log.md       Append-only operation timeline
  wip.md       Work in progress / session continuity
  concepts/    Topic and concept pages
  entities/    People, models, organizations, datasets
  summaries/   One page per ingested source
  analysis/    Filed query answers (they compound too)
  _meta/       sources.md registry, backups

/raw/ → Pipeline → /wiki/   (one-way: raw stays immutable)
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `kb ingest` | Scan `/raw`, fingerprint new/changed files, queue them |
| `kb ingest --watch` | Watch for new files continuously |
| `kb compile` | Run full agent pipeline, write wiki |
| `kb compile --guided` | One source at a time (interactive, Karpathy-style) |
| `kb compile --force` | Reprocess all documents including already-done |
| `kb compile --interactive` | Prompt for human approval on low-confidence outputs |
| `kb query "question"` | Hybrid BM25 + vector search |
| `kb query "..." --generate` | AI-synthesised answer from wiki context |
| `kb query "..." --file` | File answer back to `wiki/analysis/` |
| `kb query "..." --mode bm25` | Keyword-only search |
| `kb lint` | Quality-check wiki: orphans, stale pages, broken links |
| `kb lint --output report.md` | Save lint report to file |
| `kb status` | Show processing stats (pending/done/failed/review) |
| `kb status --detailed` | Per-document breakdown |
| `kb session` | Show session context: recent log + wiki state |
| `kb session --claude-md` | Write `wiki/CLAUDE.md` for LLM agent startup |

---

## Agent Pipeline

Each document flows through these agents in order:

```
Ingestor → Summarizer → ConceptExtractor → Linker
         → IntegrationAgent (updates 10-15 existing pages)
         → EntityAgent (creates/updates entity pages)
         → IndexBuilder → LintingQA
```

### Confidence gate
- Score ≥ 0.85: auto-approved, written to wiki
- Score 0.70–0.85: written with `status: review` frontmatter
- Score < 0.70: queued for human review (`kb compile --interactive`)

### Incremental processing
Files are SHA-256 fingerprinted. Unchanged files are skipped automatically. Only new or modified sources trigger the pipeline.

---

## Web UI

Start with `uvicorn web.main:app --reload` and open `http://localhost:8000`.

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Stats, recent log, quick search, compile trigger |
| Articles | `/articles` | Browse all wiki pages, filter by type |
| Article | `/articles/{subdir}/{slug}` | Read + raw view of any wiki page |
| Search | `/search` | Full-text keyword search |
| Run Pipeline | `/run` | Trigger compile, view status, CLI reference |
| API Docs | `/docs` | FastAPI OpenAPI documentation |
| Health | `/health` | JSON health check |

### API endpoints

```
GET  /api/articles/                    List all articles
GET  /api/articles/{subdir}/{slug}     Get article content
GET  /api/search/?q=query&k=5         Full-text search
POST /api/agents/compile               Trigger background compile
GET  /api/agents/jobs/{job_id}         Poll job status
GET  /api/agents/status                DB stats
```

---

## Wiki Page Structure

Every wiki page has YAML frontmatter with depth tiers:

```yaml
---
title: "Attention Mechanism"
type: concept
tags: [ml, nlp, attention]
created: 2026-04-10T08:00:00Z
updated: 2026-04-10T08:00:00Z
source: raw/papers/attention_is_all_you_need.md
confidence: 0.91
depth: L1
status: active
l0_summary: "Computes weighted sums over value vectors using query-key compatibility."
related: [transformer-architecture, multi-head-attention]
contradicts: []
---
```

**Depth levels:**
- **L0** — `l0_summary` field only (one sentence). Always loaded for index scanning.
- **L1** — Full page body. Standard for most queries.
- **L2** — Deep reference with all sources and raw quotes. For research deep-dives.

---

## Docker

```bash
docker compose up -d          # starts kb + ollama
docker compose exec kb kb ingest
docker compose exec kb kb compile
# Web UI available at http://localhost:8000
```

To pull the model inside the container:
```bash
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
```

---

## Project Structure

```
wiki-forge/
├── kb/
│   ├── agents/
│   │   ├── base.py                  BaseAgent: retry, validation, confidence scoring
│   │   ├── summarizer.py            Structured summaries from document chunks
│   │   ├── concept_extractor.py     Concepts, entities, relationships
│   │   ├── linker.py                [[wikilinks]] and backlinks
│   │   ├── integration_agent.py     ★ Updates existing pages on new source ingest
│   │   ├── entity_agent.py          ★ Persistent entity pages (people/models/orgs)
│   │   ├── index_builder.py         TOC, map-of-content files
│   │   ├── linting.py               Quality scoring and improvement suggestions
│   │   └── research.py              Gap detection, enrichment suggestions
│   ├── pipelines/
│   │   ├── engine.py                DAG pipeline with topological sort
│   │   └── orchestrator.py          ★ Full Karpathy-pattern compile cycle
│   ├── search/
│   │   └── hybrid_search.py         ★ BM25 + FAISS + Reciprocal Rank Fusion
│   ├── services/
│   │   ├── llm/
│   │   │   ├── base.py              LLMProvider interface
│   │   │   ├── openai_provider.py   OpenAI implementation
│   │   │   ├── ollama_provider.py   Ollama local implementation
│   │   │   └── factory.py           Config-driven provider factory
│   │   └── vector/
│   │       └── faiss_store.py       FAISS vector store + embedding pipeline
│   ├── storage/
│   │   ├── metadata_store.py        SQLite document state tracking
│   │   ├── wiki_manager.py          Read/write/list/backup wiki articles
│   │   └── wiki_logger.py           ★ log.md, index.md, wip.md, sources.md
│   ├── ingestion/
│   │   ├── ingestion_engine.py      Scan, hash, dedup, queue
│   │   └── parsers/
│   │       └── document_parsers.py  Text, PDF (PyMuPDF), Image (+OCR)
│   ├── tools/
│   │   └── plugin_registry.py       Dynamic agent registration
│   └── utils/
│       ├── config.py                Pydantic-settings configuration
│       ├── logging.py               Structlog setup
│       └── helpers.py               Slugify, chunk, hash, retry, JSON parse
├── cli/
│   ├── main.py                      Typer app root
│   └── commands/
│       ├── ingest.py                kb ingest
│       ├── compile.py               kb compile
│       ├── query.py                 kb query
│       ├── lint.py                  kb lint
│       ├── status.py                kb status
│       └── session.py               kb session
├── web/
│   ├── main.py                      FastAPI app (all page + API routes)
│   ├── routers/
│   │   ├── articles.py              /api/articles
│   │   ├── search.py                /api/search
│   │   └── agents.py                /api/agents
│   └── templates/
│       ├── base.html                Shared layout + CSS
│       ├── index.html               Dashboard
│       ├── articles.html            Article browser
│       ├── article.html             Article viewer
│       ├── search.html              Search page
│       └── run.html                 Pipeline controls
├── tests/
│   ├── unit/
│   │   ├── test_storage_and_helpers.py   29 tests
│   │   ├── test_agents.py                31 tests
│   │   └── test_production_upgrades.py   35 tests
│   └── integration/
│       └── test_e2e_pipeline.py          10 tests (95 total)
├── config/
│   └── default.yaml                 Default configuration
├── wiki/                            Generated wiki (gitignored)
│   └── WIKI.md                      Governing schema — read this first
├── raw/                             Input documents (gitignored)
├── data/                            SQLite DB + FAISS vectors (gitignored)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Adding Custom Agents

```python
# my_agent.py
from kb.agents.base import AgentInput, BaseAgent

class MyAgent(BaseAgent):
    name = "my_agent"

    def _execute(self, inp: AgentInput) -> dict:
        result = self._call_llm_json("your prompt", "system prompt")
        return result  # must include "confidence" key

    def validate(self, result: dict) -> bool:
        return "confidence" in result and "my_field" in result

# Register it
from kb.tools.plugin_registry import register_agent
register_agent("my_agent", MyAgent)
```

---

## Configuration

All settings can be overridden via environment variables or `.env`:

```env
# LLM
LLM__PROVIDER=ollama          # ollama | openai
LLM__MODEL=llama3.2
LLM__OPENAI_API_KEY=sk-...
LLM__BASE_URL=http://localhost:11434

# Paths
RAW_DIR=./raw
WIKI_DIR=./wiki
DB_PATH=./data/kb.db
VECTOR_STORE_PATH=./data/vectors

# Processing thresholds
PROCESSING__CONFIDENCE_THRESHOLD=0.70    # below = human review
PROCESSING__AUTO_APPROVE_THRESHOLD=0.85  # above = auto-write

# Logging
LOG_LEVEL=INFO                # DEBUG | INFO | WARNING
LOG_FORMAT=json               # json | console
```

---

## Development

```bash
# Run all 95 tests
pytest tests/ -v

# End-to-end demo (no LLM needed — uses mock)
python production_demo.py

# Lint and format
ruff check kb/ cli/ web/
black kb/ cli/ web/

# Type check
mypy kb/

# Start web UI in dev mode
uvicorn web.main:app --reload --port 8000
```

---

## Scaling Path

| Phase | Scale | Configuration |
|-------|-------|---------------|
| MVP | 1–100 docs | Default (SQLite + BM25). No vector DB needed. |
| v1 | 100–5k docs | Enable FAISS: `kb compile` builds vectors automatically. |
| v2 | 5k–50k docs | Switch to Chroma. Add Celery workers. DuckDB analytics. |
| v3 | 50k+ | Kafka ingest. Multi-node LLM (vLLM). Qdrant vector DB. |

---

## Troubleshooting

**`kb compile` produces no articles:**
```bash
kb status          # check if documents are in "review" state
kb status --detailed
# If confidence is low, use:
kb compile --interactive    # approve/reject each output
```

**Web UI shows empty wiki:**
```bash
kb ingest          # make sure files are queued
kb compile         # then compile
# Refresh http://localhost:8000
```

**Ollama connection errors:**
```bash
curl http://localhost:11434/api/tags    # check Ollama is running
ollama list                             # check models are pulled
ollama pull llama3.2                    # pull if missing
```

**Broken wikilinks in the web UI:**
All `[[wikilinks]]` now route correctly:
- `[[slug]]` → `/articles/concepts/slug`
- `[[summaries/slug|Title]]` → `/articles/summaries/slug`
- `[[entities/person]]` → `/articles/entities/person`

---

## License

MIT — see `LICENSE` for details.

## Acknowledgements

- [Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — LLM Wiki pattern inspiration
- [Ollama](https://ollama.com) — local LLM serving
- [FastAPI](https://fastapi.tiangolo.com) — web framework
- [HTMX](https://htmx.org) — lightweight interactivity
