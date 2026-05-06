# SKILLS.md — Technology Stack: The "Why" Behind Every Choice

> Current stack version snapshot — April 2026

This document explains the engineering rationale for each library and pattern
in the SEC Filing Intelligence system. Every choice has a specific reason tied
to the constraints of financial document retrieval at production quality.

---

## System Overview

```
User Question
    │
    ▼
Streamlit UI (main.py + src/)
    │  Pydantic-typed request
    ▼
APIClient (src/utils/api_client.py)  ← @st.cache_resource singleton
    │  HTTP POST
    ▼
FastAPI Backend (src/api/app.py)
    │
    ├── HybridSearcher  →  BM25 + Jina Vector + RRF
    │         │
    │         ▼
    │   BGEReranker  →  top-20 → top-5
    │         │
    │         ▼
    │   LLMRouter  →  Ollama / Groq / Gemini
    │
    └── VectorStoreManager  →  Pinecone or ChromaDB
```

---

## Retrieval-Augmented Generation (RAG)

**Why RAG instead of a fine-tuned LLM?**
- 10-K filings are updated annually; LLM training data has a fixed cutoff
- Financial figures must be exactly accurate — hallucination is unacceptable
- Every answer must cite a specific filing (ticker + year) for auditability
- RAG lets us update the knowledge base (ingest new filings) without retraining

**Why SEC 10-K filings?**
- The most information-dense, legally certified document a company produces
- Standardised section structure across all companies — predictable retrieval
- Publicly available from SEC EDGAR, no licensing or scraping required

---

## Language Framework: LangChain `1.2.14`

**Why LangChain:**
- `BaseLanguageModel` interface: `.invoke()` and `.astream()` work identically on Ollama, Groq, and Gemini — swapping providers is a single string change
- `ChatPromptTemplate` enforces the system/human message structure that prevents subtle prompt injection from malformed inputs
- `Document` is the universal data container used by every vector store, text splitter, and retrieval component — no custom adapter code
- LangSmith integration is zero-config: set two env vars and every call is traced

**Alternative considered:** LlamaIndex — broader built-in connectors but less flexible for custom retrieval pipelines with a cross-encoder reranker.

---

## Data Validation: Pydantic v2 `2.12.5`

**Why Pydantic v2:**
- Type errors raised at object construction time, not when a downstream function accesses a missing key in a raw `dict`
- Pydantic v2's Rust core is 5-50× faster than v1 for large-scale validation
- `@field_validator` with `mode="before"` lets us normalise incoming data (e.g. clamp `rerank_score` to [0,1], uppercase tickers) in one place

**Key schemas and what they guard:**

| Schema | Guards Against |
|---|---|
| `QueryRequest` | Empty questions; invalid provider strings; top_k out of [1,10] |
| `Source` | `rerank_score > 1.0` (BGE cross-encoder can return values above 1) |
| `LatencyBreakdown` | Missing stage keys from API response; type coercion |
| `ChatMessage` | Wrong role strings; accessing `.has_sources` on empty list |
| `IngestRequest` | Blank ticker list; tickers not uppercased |
| `Settings` (config.py) | Wrong env var types at startup, not at runtime |

**Why pydantic-settings `2.13.1`:**
- Auto-loads `.env` file — no `load_dotenv()` calls scattered around
- All config in one `Settings` class — self-documenting, one place to change
- `CHUNK_SIZE=abc` raises a clear `ValidationError` at startup instead of `int()` failing inside ingestion

---

## Hybrid Search: BM25 + Vector + RRF

**Package:** `rank-bm25 0.2.2`

**Why each component:**

| Component | Strength | Weakness |
|---|---|---|
| BM25 | Exact keyword matches — finds "revenue 2024" reliably | Misses synonyms ("earnings" ≠ "revenue" in BM25) |
| Vector search | Semantic similarity — finds "net sales" for "revenue" query | Loses exact number/date matches (all financials embed similarly) |
| RRF fusion | Documents in both lists get double-boosted scores | Slightly more complex than either alone |

**RRF formula:** `score(doc) = Σ 1/(k + rank)` where k=60 smooths top-rank dominance.

Benchmark: hybrid consistently outperforms either method alone by 15–20% on financial QA tasks.

---

## Reranker: BAAI/bge-reranker-base (CrossEncoder)

**Package:** `sentence-transformers 3.3.1`

**Why two-stage retrieval:**

```
Stage 1 — Bi-encoder (ms latency, approximate)
  Embed query independently → ANN index search → top-20 candidates
  Scales to millions of vectors; less accurate (no query-doc interaction)

Stage 2 — Cross-encoder (200–400ms, precise)
  Process (query, document) TOGETHER → full transformer attention
  Much more accurate; only applied to top-20 (not the whole corpus)
```

**Why BGE over alternatives:**
- BAAI/bge-reranker-base: free, 278M params, runs on CPU (no GPU needed)
- Specifically trained for relevance ranking — outperforms general-purpose models
- Adds ~15% faithfulness score improvement over bi-encoder retrieval alone

---

## LLM Providers: Ollama / Groq / Gemini

**Packages:** `langchain-ollama 1.0.1`, `langchain-groq 1.1.2`, `langchain-google-genai 4.2.1`

| Provider | Model | Speed | Cost | Best For |
|---|---|---|---|---|
| Ollama | llama3.2 | ~30 tok/s | Free (local) | Dev iteration, privacy, offline |
| Groq | llama-3.3-70b-versatile | ~500 tok/s | Free tier | Demos, production latency |
| Gemini | gemini-1.5-flash | ~100 tok/s | Free tier | Evaluation, complex queries |

**Why the router pattern:**
`get_llm(provider)` returns `BaseLanguageModel` — the rest of the pipeline is provider-agnostic. All three providers stream with `.astream()` using the same SSE generator.

---

## Vector Stores: Pinecone `5.0.1` + ChromaDB `1.5.5`

| | Pinecone | ChromaDB |
|---|---|---|
| Use case | Production, cloud deployment | Local dev, zero-config |
| Scale | Billions of vectors, serverless | Single machine, disk-based |
| Persistence | Always-on (managed) | `./chroma_db` folder |
| API key | Required | Not required |
| Free tier | 1 index, 100K vectors | Unlimited (open-source) |

**Auto-selection:** `USE_LOCAL_CHROMA=true` in `.env` uses ChromaDB; `false` uses Pinecone.

---

## Embeddings: Jina AI v2 + SentenceTransformers

**Why Jina AI (`jina-embeddings-v2-base-en`, 768 dims):**
- **8,192-token context window** — most models (OpenAI Ada, MiniLM) truncate at 512 tokens. SEC risk-factor paragraphs routinely exceed 512 tokens. Jina embeds the full paragraph without truncation.
- Free tier: 1M tokens/month — enough for all 15 initial filings plus ongoing queries

**Why LocalEmbedder fallback (`all-MiniLM-L6-v2`, 384 dims):**
- Zero-friction developer experience — works without any API key
- 90MB download, CPU-only, no GPU required

---

## API Framework: FastAPI `0.115.4`

**Why FastAPI over Flask:**
- `async/await` throughout — `run_in_threadpool()` offloads blocking model calls so the event loop stays responsive during streaming
- Request/response bodies validated automatically by Pydantic models
- `StreamingResponse` + async generator = clean SSE implementation
- Lifespan context manager loads models once at startup (not per-request)
- Auto-Swagger at `/docs` — interactive demo with zero extra code

---

## UI: Streamlit `1.40.1`

**Why Streamlit:**
- `st.chat_message`, `st.chat_input`, `st.session_state` = first-class chat primitives
- `@st.cache_resource` caches the `APIClient` across reruns — avoids new TCP connections
- Custom CSS via `st.markdown(unsafe_allow_html=True)` gives full design control

**Session state design:**
```
st.session_state.messages  : list[ChatMessage]   # typed, Pydantic-validated
st.session_state.top_k     : int                 # persisted slider value
st.session_state.streaming : bool                # persisted toggle value
st.session_state.pending_question : str | None   # injected by sidebar sample buttons
```

---

## Visualisation: Plotly `6.7.0`

**Charts delivered:**

| Chart | Data | Insight |
|---|---|---|
| Latency breakdown | `LatencyBreakdown` per stage | Which pipeline stage is the bottleneck |
| Source donut | Chunks by ticker (multi-company only) | Which company's filings dominated the answer |
| Reranker scores | BGE score per chunk | Whether the retrieved context was actually relevant |

---

## Evaluation: RAGAS `0.4.3`

Four metrics each catching a different failure mode:

| Metric | What It Catches |
|---|---|
| **Faithfulness** | Hallucination — LLM claims something not in retrieved context |
| **Answer Relevancy** | Off-topic — LLM ignored the question |
| **Context Recall** | Missing info — relevant chunks not retrieved |
| **Context Precision** | Noise — retrieved chunks irrelevant to the question |

---

## Observability: LangSmith `0.7.25`

**Zero-code integration:** set `LANGCHAIN_TRACING_V2=true` + `LANGSMITH_API_KEY` and every LangChain call is traced with exact prompt, raw response, token count, latency, and errors.

**Custom logging layer (`src/core/logging_config.py`):**
- `setup_logging()` — structured logs: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`
- `@log_call` — logs function entry/exit + wall-clock time
- `@handle_errors` — catches exceptions at UI boundaries, returns fallback instead of crashing Streamlit

---

## Containerisation: Docker + Docker Compose

**`docker-compose.yml` services:**
- `api`: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
- `ui`: `streamlit run main.py --server.port 8501`
- Volume mount: `./chroma_db` persists between restarts

---

## Dependency Management: requirements.txt → uv

**Phase 2 upgrade to `uv`:**
```bash
pip install uv
uv init
uv add $(grep -v "^#" requirements.txt | tr '\n' ' ')
```
`uv` resolves and installs dependencies 10-100× faster and produces a lockfile with hash verification.
