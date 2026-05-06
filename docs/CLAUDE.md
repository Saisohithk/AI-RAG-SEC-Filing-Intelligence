# CLAUDE.md — AI Coding Assistant Context & Instructions

> Read this file before editing any code in this repository.
> These instructions override default assistant behavior.

---

## 1. Project Identity

**Name:** SEC Filing Intelligence — RAG System  
**Purpose:** Retrieval-Augmented Generation over SEC EDGAR 10-K/10-Q filings. Users ask natural-language questions; the system retrieves relevant filing chunks and generates grounded answers with citations.  
**Python:** 3.11+ (3.13 used in development)  
**Package manager:** `uv` (preferred) or `pip`. Use `uv pip install -r requirements.txt`.

---

## 2. Tech Stack & Exact Versions

| Layer | Library | Version |
|---|---|---|
| API Framework | `fastapi` | 0.115.4 |
| ASGI Server | `uvicorn` | 0.32.1 |
| Frontend | `streamlit` | 1.40.1 |
| Schema / Config | `pydantic` / `pydantic-settings` | 2.12.5 / 2.13.1 |
| LangChain | `langchain` + ecosystem | 1.2.x |
| Vector DB | `chromadb` (local) / `pinecone-client` (cloud) | 1.5.5 / 5.0.1 |
| Embeddings | `sentence-transformers` | 3.3.1 |
| Reranking | `sentence-transformers` CrossEncoder | BAAI/bge-reranker-base |
| BM25 | `rank-bm25` | 0.2.2 |
| LLM Providers | `langchain-groq`, `langchain-google-genai`, `langchain-ollama` | latest pinned |
| Evaluation | `ragas` | 0.4.3 |
| Observability | `langsmith` | 0.7.25 |
| Visualisation | `plotly` | 6.7.0 |
| Testing | `pytest` | 8.3.3 |
| HTTP client | `httpx` | 0.28.1 |

**Do not upgrade versions without updating `requirements.txt` and running the full test suite.**

---

## 3. Directory Structure

```
sec-rag/
├── src/
│   ├── api/
│   │   ├── app.py          ← FastAPI app, all route definitions
│   │   └── main.py         ← uvicorn entrypoint
│   ├── core/
│   │   ├── config.py       ← pydantic-settings Settings class
│   │   ├── schemas.py      ← ALL shared Pydantic models
│   │   └── logging_config.py ← @log_call, @handle_errors decorators
│   ├── ingestion/
│   │   ├── downloader.py   ← SEC EDGAR fetch
│   │   ├── chunker.py      ← RecursiveCharacterTextSplitter wrapper
│   │   └── embedder.py     ← Jina / LocalEmbedder with fallback
│   ├── retrieval/
│   │   ├── hybrid_search.py ← BM25 + vector + RRF fusion
│   │   └── reranker.py     ← CrossEncoder top-20 → top-5
│   ├── generation/
│   │   ├── llm_router.py   ← Ollama / Groq / Gemini dispatcher
│   │   └── prompt_templates.py ← All prompt strings (no inline prompts elsewhere)
│   └── ui/
│       ├── chat.py         ← Streamlit chat rendering
│       ├── charts.py       ← Plotly source donut + reranker bars
│       └── sidebar.py      ← Settings panel
├── tests/
│   ├── conftest.py         ← Shared fixtures (sample_documents, mock_embedder)
│   ├── unit/               ← One file per src/ module
│   └── integration/        ← test_api.py via httpx.AsyncClient
├── evals/
│   ├── run_ragas.py        ← RAGAS evaluation runner
│   └── generate_testset.py ← Test question generation
├── docs/
│   ├── PLAN.md             ← Engineering roadmap (SMART tasks)
│   └── SKILLS.md           ← Skills and competencies map
├── data/                   ← Raw downloaded filings (gitignored)
├── chroma_db/              ← Local ChromaDB persistence (gitignored)
├── Makefile                ← make install | test | lint | docker-up
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

### Data Flow

```
User question
    │
    ▼
POST /query  (src/api/app.py)
    │
    ├─► Query Rewriting (src/generation/llm_router.py + prompt_templates.py)
    │       └─► 3 query variants [PLANNED — CS-5]
    │
    ├─► Hybrid Search (src/retrieval/hybrid_search.py)
    │       ├─► BM25 sparse retrieval (rank-bm25)
    │       ├─► Vector retrieval (ChromaDB / Pinecone)
    │       └─► RRF fusion → top-20 candidates
    │
    ├─► Reranking (src/retrieval/reranker.py)
    │       └─► CrossEncoder → top-5 with scores
    │
    ├─► LLM Generation (src/generation/llm_router.py)
    │       └─► Groq / Gemini / Ollama → answer string
    │
    └─► QueryResponse (src/core/schemas.py)
            └─► JSON to Streamlit UI
```

---

## 4. Naming Conventions

| Scope | Convention | Example |
|---|---|---|
| Files | `snake_case.py` | `hybrid_search.py` |
| Classes | `PascalCase` | `HybridSearcher`, `LLMRouter` |
| Functions / methods | `snake_case` | `search()`, `rerank()` |
| Pydantic models | `PascalCase` | `QueryRequest`, `Source` |
| Constants | `UPPER_SNAKE_CASE` | `QUERY_REWRITE_PROMPT` |
| Environment variables | `UPPER_SNAKE_CASE` | `LLM_PROVIDER`, `API_KEY` |
| Test files | `test_<module>.py` | `test_hybrid_search.py` |
| Test functions | `test_<what>_<condition>()` | `test_rrf_score_empty_corpus()` |

---

## 5. Development Constraints

### Must Do
- **All shared types live in `src/core/schemas.py`.** Never define a Pydantic model inline in a route or module.
- **All prompt strings live in `src/generation/prompt_templates.py`.** No inline f-string prompts elsewhere.
- **All config is read via `settings.*`.** Never read `os.environ` directly.
- **New API endpoints require an integration test** in `tests/integration/test_api.py`.
- **Decorate all external calls** with `@log_call` from `src/core/logging_config.py`.
- **Use `httpx.AsyncClient`** for HTTP calls; never `requests` in async contexts.
- **Type-annotate every function signature.** Return type included.

### Must Not Do
- Do not introduce external CSS frameworks (no Tailwind, Bootstrap). Streamlit native + `st.markdown` with inline styles only.
- Do not use `dict[str, Any]` at module boundaries — use typed Pydantic models.
- Do not call `os.environ` directly anywhere in `src/`.
- Do not add `print()` statements — use `logging.getLogger(__name__)`.
- Do not use `FastAPI BackgroundTasks` for new async work — use the RQ job queue (see I-5 in PLAN.md).
- Do not use class-based views or `APIRouter` unless a module exceeds 300 lines.
- Do not write multi-line docstrings — a single summary line maximum.
- Do not commit `data/`, `chroma_db/`, `*.pkl`, or `*.pyc` files.

### Testing Rules
- Tests must not make real network calls. Mock `httpx`, `chromadb`, and `langchain` at boundaries.
- Use `monkeypatch` or `unittest.mock.MagicMock`; never `pytest-mock`.
- Fixtures shared across suites belong in `tests/conftest.py`.
- Coverage target: `src/core/` at 100%; all other modules ≥70%.

---

## 6. Environment Variables Reference

See `.env.example` for the full list. Critical variables:

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `groq` \| `gemini` \| `ollama` |
| `GROQ_API_KEY` | If Groq | Groq cloud API key |
| `GOOGLE_API_KEY` | If Gemini | Google AI Studio key |
| `JINA_API_KEY` | Optional | Falls back to LocalEmbedder |
| `PINECONE_API_KEY` | If Pinecone | Cloud vector store key |
| `VECTOR_STORE` | Yes | `chroma` \| `pinecone` |
| `LANGCHAIN_API_KEY` | Optional | LangSmith tracing |
| `API_KEY` | v2.0+ | Authenticates `/query` callers |

---

## 7. Running Locally

```bash
make install          # uv pip install -r requirements.txt
make dev              # uvicorn src.api.main:app --reload  (port 8000)
make ui               # streamlit run main.py              (port 8501)
make test             # pytest tests/ -v --cov=src
make lint             # ruff check . && mypy src/
make docker-up        # docker-compose up --build
```

---

## 8. Services & Architecture (v2.0)

All planned items from `docs/PLAN.md` are implemented. The stack at v2.0:

| Component | Location | Notes |
|---|---|---|
| API auth | `verify_api_key` dependency in `app.py` | `X-API-Key` header; no-op when `API_KEY=""` |
| Rate limiting | `slowapi` in `app.py` | 60/min on `/query`; 30/min on `/query/stream` |
| Redis cache | `app.py` lifespan | 1-hour TTL; `sha256(question:provider:top_k)` key |
| BM25 persistence | `HybridSearcher.save_index()` / `load_index()` | `joblib` + atomic rename to `data/bm25_index.pkl` |
| Query rewriting | `_get_query_variants()` in `app.py` | Calls `QUERY_REWRITE_PROMPT`; fallback to original |
| Metadata filtering | `_build_filters()` in `app.py` | `ticker` + `year` fields on `QueryRequest` |
| Async job queue | RQ + `scripts/worker.py` | Falls back to `BackgroundTasks` when Redis absent |
| Job status polling | `GET /ingest/{job_id}` | Requires Redis |
| CI/CD | `.github/workflows/ci.yml` | lint + test + docker; PR gate |
| Nightly eval gate | `.github/workflows/nightly_ragas.yml` | faithfulness ≥ 0.85, relevancy ≥ 0.80 |
| K8s manifests | `k8s/` | Deployment + HPA + PVC + ConfigMap/Secret |
| E2E tests | `tests/e2e/test_pipeline.py` | `pytest -m e2e`; excluded from normal CI |
| Reranker unit tests | `tests/unit/test_reranker.py` | Full coverage with mock CrossEncoder |

## 9. Running Tests

```bash
make test                           # unit + integration (default, excludes e2e)
pytest tests/unit/ -v               # unit only
pytest tests/integration/ -v        # integration only
pytest tests/e2e/ -m e2e -v        # e2e (requires docker compose up)
pytest --cov=src --cov-report=html  # with HTML coverage report
```
