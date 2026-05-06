# PLAN.md — SEC Filing Intelligence: Engineering Roadmap

> **Version:** v2.0  
> **Last Updated:** April 2026  
> **Owner:** Engineering  
> **Status Key:** ✅ Done | 🔄 In Progress | 📋 Planned | ❌ Blocked

---

## FOUNDATION

### F-1 — Pydantic v2 Type-Safe Schema Layer ✅
**Deliverable:** `src/core/schemas.py` defines `QueryRequest`, `QueryResponse`, `Source`, `ChatMessage`, `LatencyBreakdown`, `IngestJobResponse` with validators at every layer boundary.  
**Measure:** Zero `dict[str, Any]` exchanges across module boundaries; validator edge cases at 100% unit coverage.

### F-2 — Environment Configuration ✅
**Deliverable:** `pydantic-settings` loads `.env`; type errors surface at startup. `Settings.__repr__` masks `API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY` in logs.  
**Measure:** repr output verified in startup logs; masked values confirmed with `****` pattern.

### F-3 — Structured Logging ✅
**Deliverable:** `src/core/logging_config.py` with `@log_call` and `@handle_errors` decorators applied across all API and UI boundaries.  
**Measure:** Every external call emits a structured JSON log line with `duration_ms`, `status`, and `caller`.

### F-4 — Dependency & Reproducibility ✅
**Deliverable:** `requirements.txt` with exact version pins including `slowapi`, `redis`, `rq`, `pytest-cov`, `mypy`, `ruff`; `Makefile`; `.env.example` documents every variable.  
**Measure:** `make install && make test` succeeds on a clean clone with no manual steps.

### F-5 — CI/CD Pipeline ✅
**Deliverable:** `.github/workflows/ci.yml` — three jobs: `lint` (ruff + mypy), `test` (pytest ≥70% coverage + codecov), `docker` (compose build). `all-checks-pass` gate required before merge.  
**Measure:** Every PR gated; `main` branch always green and deployable.

### F-6 — Persistent BM25 Index ✅
**Deliverable:** `HybridSearcher.save_index()` / `load_index()` using `joblib` with atomic rename (`.tmp` → final). Saved after every successful ingest; loaded at API startup if file exists.  
**Measure:** Cold-start time drops from ~20s to <2s; `data/bm25_index.pkl` path configurable via `settings.BM25_INDEX_PATH`.

---

## CORE SERVICES

### CS-1 — SEC EDGAR Ingestion Pipeline ✅
**Deliverable:** `src/ingestion/{downloader,chunker,embedder}.py` — 15 filings (5 tickers × 3 years) → 4,779 vectors indexed in ChromaDB.  
**Measure:** Correct metadata (`ticker`, `filing_date`, `chunk_id`) on all vectors.

### CS-2 — Hybrid Retrieval (BM25 + Vector + RRF) ✅
**Deliverable:** `src/retrieval/hybrid_search.py` — RRF fusion with optional metadata filters (`ticker`, `filing_date`).  
**Measure:** Unit tests cover RRF math, empty-corpus guard, BM25-not-built guard, and filter application.

### CS-3 — Cross-Encoder Reranking ✅
**Deliverable:** `src/retrieval/reranker.py` — `BAAI/bge-reranker-base` top-20 → top-5. Score attached to metadata.  
**Measure:** Full unit test suite covering ordering, top_k limiting, score attachment, metadata preservation, and pair construction.

### CS-4 — LLM Router (Multi-Provider) ✅
**Deliverable:** `src/generation/llm_router.py` — Ollama / Groq / Gemini with streaming. `get_streaming_llm("ollama")` passes `streaming=True`.  
**Measure:** Unit tests cover all three providers, `ValueError` on missing API key, case-insensitive provider names.

### CS-5 — Query Rewriting (Multi-Variant) ✅
**Deliverable:** `_get_query_variants()` in `app.py` — calls `QUERY_REWRITE_PROMPT`, parses JSON, returns 3 variants + original. Candidates deduplicated by `chunk_id` before reranking. Gracefully falls back to original query on LLM/JSON error.  
**Measure:** All `/query` and `/query/stream` calls now route through multi-variant retrieval.

### CS-6 — RAGAS Evaluation Pipeline ✅
**Deliverable:** `evals/run_ragas.py` — faithfulness, answer relevancy, context recall, context precision.  
**Measure:** Results stored in `evals/results/`; baseline scores documented.

### CS-7 — RAGAS Regression CI Gate ✅
**Deliverable:** `.github/workflows/nightly_ragas.yml` — runs daily at 02:00 UTC; fails if `faithfulness < 0.85` or `answer_relevancy < 0.80`; uploads results artifact.  
**Measure:** Zero silent quality regressions between releases.

---

## INTEGRATIONS

### I-1 — LangSmith Tracing ✅
**Deliverable:** All LLM calls traced via `langsmith`; project name via `LANGCHAIN_PROJECT`.  
**Measure:** 100% of `/query` and `/query/stream` visible in LangSmith dashboard.

### I-2 — API Key Authentication ✅
**Deliverable:** `verify_api_key` FastAPI dependency — validates `X-API-Key` header against `settings.API_KEY`. Returns HTTP 401 on mismatch. No-op when `API_KEY` is empty (local dev).  
**Measure:** Integration tests assert 401 on wrong key and 200 on correct key. `/health` remains public.

### I-3 — Rate Limiting ✅
**Deliverable:** `slowapi` middleware — 60 req/min on `/query`; 30 req/min on `/query/stream`; keyed by remote IP. Returns HTTP 429.  
**Measure:** E2E test confirms 429 on burst of 70 requests to `/query`.

### I-4 — CORS Hardening ✅
**Deliverable:** `ALLOWED_ORIGINS` setting (comma-separated); loaded into `CORSMiddleware`. Defaults to `"*"` for local dev; set to exact frontend URL in production via `.env`.  
**Measure:** CORS preflight from unlisted origin returns HTTP 403 in production config.

### I-5 — Async Ingestion Queue ✅
**Deliverable:** `/ingest` enqueues to RQ (`Queue("ingestion")`) when `REDIS_URL` is set; falls back to `BackgroundTasks` otherwise. `GET /ingest/{job_id}` polls job status. `scripts/worker.py` is the standalone RQ worker process.  
**Measure:** Ingestion job survives API restart when Redis is configured; status queryable after server bounce.

### I-6 — Redis Query Cache ✅
**Deliverable:** `/query` checks Redis for `sha256(question:provider:top_k)` key before pipeline execution (TTL 1 hour). Cache write on miss. Gracefully degrades when Redis unavailable.  
**Measure:** Repeated identical queries return in <50ms when cached; `/health` reports `redis_connected`.

### I-7 — Metadata Filtering ✅
**Deliverable:** `QueryRequest` accepts optional `ticker: str | None` (pattern `^[A-Z]{1,5}$`) and `year: int | None` (2000–2030). `_build_filters()` constructs filter dict passed to `searcher.search()`.  
**Measure:** Single-ticker queries filter at the RRF fusion stage; integration test confirms filter applied.

---

## DEPLOYMENT

### D-1 — Docker Compose (Local) ✅
**Deliverable:** `docker-compose.yml` — four services: `redis` (7-alpine), `api`, `worker`, `streamlit`. Health-check dependencies enforced. `data/` and `chroma_db/` volumes mounted.  
**Measure:** `docker compose up` starts all four services; `/health` returns 200 with `redis_connected: true`.

### D-2 — Testing Pyramid (Unit + Integration) ✅
**Deliverable:** Unit tests for all six `src/` modules (`schemas`, `chunker`, `hybrid_search`, `llm_router`, `reranker`); integration tests for all API endpoints including auth cases.  
**Measure:** `pytest tests/unit/ tests/integration/ --cov=src --cov-fail-under=70` passes.

### D-3 — End-to-End Smoke Test ✅
**Deliverable:** `tests/e2e/test_pipeline.py` — covers health, ingest, query, metadata filter, stats, and rate-limit trigger. Requires running docker-compose stack; excluded from normal `pytest` via `pytest.ini`.  
**Measure:** `pytest tests/e2e/ -m e2e -v` passes against live stack; runtime < 3 minutes.

### D-4 — Kubernetes Deployment ✅
**Deliverable:** `k8s/` manifests — `api-deployment.yaml` (Deployment + Service + HPA 3–10 replicas), `worker-deployment.yaml` (2 replicas), `ui-deployment.yaml` (2 replicas + LoadBalancer), `configmap.yaml` (ConfigMap + Secret + PVC for BM25 index).  
**Measure:** `kubectl apply -k k8s/` deploys the full stack; HPA scales on CPU 70%.

### D-5 — Pinecone Serverless Migration ✅
**Deliverable:** Code natively supports ChromaDB (local dev via `USE_LOCAL_CHROMA=true`) and Pinecone Serverless (production via `USE_LOCAL_CHROMA=false` + `PINECONE_API_KEY`). K8s ConfigMap defaults to Pinecone.  
**Measure:** Same API contract regardless of vector store backend; backend selected at startup via settings.

---

## Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|---|---|---|---|
| MVP | ChromaDB local + Pinecone cloud dual support | Zero-config local dev; production cloud scale | Weaviate (Docker complexity), Qdrant (less mature SDK) |
| MVP | BGE open-source reranker | No API cost; CPU-only; best OSS cross-encoder quality | Cohere Rerank (paid), ColBERT (complex to host) |
| MVP | FastAPI BackgroundTasks for ingestion | No extra infra for MVP; explicit Phase 2 upgrade path | RQ (Phase 2), Celery (Phase 3) |
| MVP | Jina + LocalEmbedder fallback | Zero-friction onboarding; dev works offline without any API key | OpenAI Ada (paid), Cohere Embed (no free tier) |
| v1.0 | Pydantic v2 schemas across all layer boundaries | Silent `dict[str, Any]` bugs caught at construction time | TypedDict (no validation), dataclasses (no JSON round-trip) |
| v1.0 | `@st.cache_resource` for APIClient | Single connection pool reused across all Streamlit reruns | New client per request (TCP overhead) |
| v1.0 | Plotly v6 for charts | Full dark-theme control; hover tooltips; donut/bar chart types | st.bar_chart (no theming), Altair (verbose config) |
| v2.0 | uv for dependency management | 10-100× faster than pip; lockfile reproducibility | Poetry (slower), pip-tools (manual) |
| v2.0 | slowapi for rate limiting | FastAPI-native decorator syntax; Redis backend support | Manual middleware (more code), nginx rate limiting (infra dep) |
| v2.0 | joblib over pickle for BM25 | Atomic write prevents corruption; numpy-aware; version-tolerant | pickle (unsafe from untrusted sources; no atomic write) |
| v2.0 | RQ over Celery for ingestion queue | Simpler setup; single Redis dependency already present; sufficient for single-server | Celery (heavier; requires separate broker + result backend config) |
| v2.0 | Query rewriting with graceful fallback | +8% recall gain; zero cost if LLM fails to return valid JSON | Always-on (brittle), disabled (leaves recall on table) |
