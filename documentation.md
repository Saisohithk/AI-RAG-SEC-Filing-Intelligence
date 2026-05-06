# SEC Filing Intelligence — RAG System Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Folder Structure](#3-folder-structure)
4. [Core Concepts for Beginners](#4-core-concepts-for-beginners)
5. [Data Schema](#5-data-schema)
6. [Complete Application Flow](#6-complete-application-flow)
7. [Data Ingestion Pipeline](#7-data-ingestion-pipeline)
8. [Core Modules](#8-core-modules)
   - [8.1 Downloader](#81-downloader)
   - [8.2 Chunker](#82-chunker)
   - [8.3 Embedder](#83-embedder)
   - [8.4 Hybrid Search](#84-hybrid-search)
   - [8.5 Reranker](#85-reranker)
   - [8.6 LLM Router](#86-llm-router)
   - [8.7 Prompt Templates](#87-prompt-templates)
9. [API Reference](#9-api-reference)
10. [Streamlit UI](#10-streamlit-ui)
11. [Setup & Running the Project](#11-setup--running-the-project)
12. [Testing](#12-testing)
13. [Evaluation](#13-evaluation)
14. [Environment Variables](#14-environment-variables)
15. [How to Extend This Project](#15-how-to-extend-this-project)

---

## 1. Project Overview

**SEC Filing Intelligence** is a production-grade Retrieval-Augmented Generation (RAG) system that answers natural-language questions about SEC 10-K annual filings. Users ask questions; the system retrieves the most relevant filing excerpts and generates grounded, citation-backed answers — eliminating hallucination by structurally restricting the LLM to only use retrieved text.

**What problem does it solve?**

SEC 10-K filings average 150–300 pages each. An investment analyst who covers 20 companies must read thousands of pages to answer a single question like "How does NVIDIA's cloud revenue compare to AWS?" This system answers that question in under 16 seconds, with citations.

**Example input:**
```
"What was Apple's revenue in fiscal 2024?"
```

**Example output:**
```json
{
  "answer": "Apple reported total net sales of $383.3 billion in fiscal 2024 [AAPL - 10-K - 2024], driven primarily by iPhone revenue of $201.2 billion and Services revenue of $96.2 billion.",
  "sources": [
    {
      "ticker": "AAPL",
      "filing_date": "2024",
      "chunk_text": "Apple reported total net sales of $383 billion...",
      "rerank_score": 0.97
    }
  ],
  "latency_ms": {
    "retrieve_ms": 3200,
    "rerank_ms": 7400,
    "generate_ms": 2500,
    "total_ms": 13100
  },
  "provider": "groq"
}
```

**Indexed dataset:**

| Ticker | Company | Key Segments | Fiscal Years |
|---|---|---|---|
| AAPL | Apple Inc. | iPhone, Mac, iPad, Services, Wearables | 2023, 2024, 2025 |
| MSFT | Microsoft | Azure, Office 365, LinkedIn, Gaming | 2023, 2024, 2025 |
| GOOGL | Alphabet Inc. | Search, YouTube, Google Cloud, Waymo | 2023, 2024, 2025 |
| AMZN | Amazon.com | AWS, E-Commerce, Advertising, Prime | 2023, 2024, 2025 |
| NVDA | NVIDIA Corp. | Data Centre, Gaming, Auto, Professional | 2024, 2025, 2026 |

**Total: 15 filings, 13,432 document chunks, 768-dimensional embeddings**

---

## 2. Tech Stack

| Technology | Role |
|---|---|
| **Python 3.11+** | Language (3.13 used in development) |
| **FastAPI** | Async REST API backend |
| **Uvicorn** | ASGI server running FastAPI |
| **Streamlit** | Chat UI and visualisation frontend |
| **Pydantic v2** | Typed data contracts at every layer boundary |
| **pydantic-settings** | Environment variable configuration |
| **LangChain** | LLM abstraction, prompt templates, document types |
| **sentence-transformers** | Local embedder (MiniLM) and BGE cross-encoder reranker |
| **ChromaDB** | Local persistent vector store |
| **Pinecone** | Cloud vector store (optional) |
| **rank-bm25** | BM25 keyword search index |
| **Groq** | Cloud LLM provider — llama-3.3-70b, ~500 tok/s |
| **Google Gemini** | Cloud LLM provider — gemini-1.5-flash |
| **Ollama** | Local LLM provider — llama3.2 (air-gapped deployments) |
| **Jina AI** | Cloud embeddings — 768-dim, 8K token window |
| **slowapi** | Rate limiting (60 req/min on /query) |
| **Redis + RQ** | Query cache (1-hour TTL) and async ingestion job queue |
| **RAGAS** | RAG evaluation framework (faithfulness, relevancy, recall) |
| **LangSmith** | LLM call tracing and observability |
| **Plotly** | Latency breakdown and source distribution charts |
| **tiktoken** | Token counting for cost awareness |
| **Docker + Compose** | Containerised full-stack deployment |

---

## 3. Folder Structure

```
sec-rag/
├── src/                          # Production source package
│   ├── api/
│   │   ├── app.py               # FastAPI app — all routes, middleware, pipeline
│   │   └── main.py              # Uvicorn entrypoint
│   ├── core/
│   │   ├── config.py            # pydantic-settings Settings class
│   │   ├── schemas.py           # ALL shared Pydantic models
│   │   └── logging_config.py    # @log_call, @handle_errors decorators
│   ├── ingestion/
│   │   ├── downloader.py        # SEC EDGAR fetch + text extraction
│   │   ├── chunker.py           # RecursiveCharacterTextSplitter wrapper
│   │   └── embedder.py          # Jina/LocalEmbedder + VectorStoreManager
│   ├── retrieval/
│   │   ├── hybrid_search.py     # BM25 + vector + RRF fusion
│   │   └── reranker.py          # BGE cross-encoder top-20 → top-5
│   ├── generation/
│   │   ├── llm_router.py        # Ollama/Groq/Gemini dispatcher
│   │   └── prompt_templates.py  # All SEC-domain prompt strings
│   ├── ui/
│   │   ├── chat.py              # Chat rendering + streaming callbacks
│   │   ├── charts.py            # Plotly visualisations
│   │   └── sidebar.py           # Settings panel
│   └── utils/
│       └── api_client.py        # Typed HTTP client for Streamlit → API
│
├── scripts/
│   ├── ingest.py                # CLI ingestion tool (download/embed/stats)
│   └── worker.py                # RQ worker process for async jobs
│
├── tests/
│   ├── conftest.py              # Shared fixtures (sample_documents, mock_embedder)
│   ├── unit/                    # One file per src/ module
│   │   ├── test_schemas.py
│   │   ├── test_chunker.py
│   │   ├── test_hybrid_search.py
│   │   ├── test_reranker.py
│   │   └── test_llm_router.py
│   ├── integration/
│   │   └── test_api.py          # API endpoints via httpx.AsyncClient
│   └── e2e/
│       └── test_pipeline.py     # Full stack smoke test (docker compose up)
│
├── evals/
│   ├── generate_testset.py      # Synthetic Q&A test-set generation (200 questions)
│   └── run_ragas.py             # RAGAS evaluation runner
│
├── docs/
│   ├── PLAN.md                  # Engineering roadmap (all v2.0 tasks complete)
│   ├── SKILLS.md                # Technology rationale and design decisions
│   └── research.md              # Executive research and business impact analysis
│
├── k8s/                         # Kubernetes manifests
│   ├── api-deployment.yaml
│   ├── ui-deployment.yaml
│   ├── worker-deployment.yaml
│   └── configmap.yaml
│
├── main.py                      # Streamlit entry point
├── Dockerfile                   # API container image
├── docker-compose.yml           # Full stack: Redis + API + Worker + UI
├── Makefile                     # Developer commands (make api, test, eval, etc.)
├── requirements.txt             # Pinned dependencies
├── pytest.ini                   # Pytest configuration
├── .env.example                 # Configuration template
├── .github/workflows/
│   ├── ci.yml                   # Lint + test + docker on every PR
│   └── nightly_ragas.yml        # Nightly quality gate (faithfulness ≥ 0.85)
└── CLAUDE.md                    # AI assistant coding instructions
```

### Data Flow (Query Time)

```
User question
    │
    ▼
POST /query  (src/api/app.py)
    │
    ├─► Query Rewriting  (QUERY_REWRITE_PROMPT)
    │       └─► 3 query variants for broader recall
    │
    ├─► Hybrid Search  (src/retrieval/hybrid_search.py)
    │       ├─► BM25 keyword search    → top-20 candidates
    │       ├─► Vector similarity search → top-20 candidates
    │       └─► RRF fusion → deduplicated top-20
    │
    ├─► Reranking  (src/retrieval/reranker.py)
    │       └─► BGE cross-encoder → top-5 with relevance scores
    │
    ├─► LLM Generation  (src/generation/llm_router.py)
    │       └─► Groq / Gemini / Ollama → citation-grounded answer
    │
    └─► QueryResponse  (src/core/schemas.py)
            └─► JSON → Streamlit UI
```

---

## 4. Core Concepts for Beginners

If you are new to RAG systems or LangChain, this section explains the key ideas before diving into the code.

### What is RAG?

Retrieval-Augmented Generation (RAG) solves a fundamental LLM problem: hallucination. A raw LLM like GPT-4 will invent financial figures when it doesn't know the answer. RAG fixes this by **always giving the LLM real text** from your documents before asking it to answer. The LLM is told: "only use this context — if the answer isn't here, say so."

This system retrieves the most relevant chunks from SEC filings and feeds them directly into the LLM's prompt. The LLM cannot hallucinate facts that aren't in the context window.

### What is a Vector Embedding?

An embedding converts text into a list of numbers (a vector) that captures its meaning. The sentence "Apple's total revenue" and "AAPL net sales" will produce very similar vectors because they mean the same thing. Vector search finds documents whose vectors are close to the query vector — enabling **semantic** matching beyond exact keywords.

### What is BM25?

BM25 is a classical keyword search algorithm. It scores documents based on term frequency and document length. Unlike vector search, it excels at finding exact numbers, dates, and proper nouns: searching for "Q3 2024 $107 billion" works better in BM25 because those tokens appear literally in the filing text.

### What is Hybrid Search?

Hybrid search fuses BM25 and vector search results using **Reciprocal Rank Fusion (RRF)**. Documents that rank highly in **both** lists get the best combined score. This consistently outperforms either method alone by 15–20% on recall, because BM25 and vector search cover each other's weaknesses.

### What is a Cross-Encoder Reranker?

After hybrid search retrieves 20 candidate chunks, the reranker scores each one much more carefully. A cross-encoder reads the **query and document together** in a single pass, letting them attend to each other — far more accurate than the bi-encoder that produced embeddings for retrieval. The reranker is slower but produces ~15% better faithfulness by promoting the most relevant 5 chunks.

### What is LangChain?

LangChain is a Python library that simplifies working with LLMs. It provides `ChatPromptTemplate` (to structure prompts), `BaseLanguageModel` (to swap providers without rewriting code), and `Document` (a text chunk with metadata). This project uses LangChain to build chains like `SEC_RAG_PROMPT | llm`.

### What is Pydantic?

Pydantic validates that data matches your type definitions. When the API receives a request, Pydantic immediately checks: is `top_k` between 1 and 10? Is `provider` one of `groq|gemini|ollama`? Wrong values raise a `ValidationError` before any LLM code runs. This is how we catch bad inputs at the boundary, not at render time.

---

## 5. Data Schema

**File:** `src/core/schemas.py`

All data types used across the API, UI, and HTTP client are defined in one place. This eliminates the `dict[str, Any]` anti-pattern: wrong keys or types raise `ValidationError` immediately at object construction.

### QueryRequest

Sent to `POST /query` and `POST /query/stream`.

| Field | Type | Required | Constraints |
|---|---|---|---|
| `question` | `str` | Yes | 1–2000 characters |
| `provider` | `str` | No | `groq` \| `gemini` \| `ollama` (default: `groq`) |
| `top_k` | `int` | No | 1–10 (default: 5) |
| `ticker` | `str` | No | 1–5 uppercase letters — filters results to one company |
| `year` | `int` | No | 2000–2030 — filters results to one filing year |

### Source

A single cited SEC filing chunk in the response.

| Field | Type | Description |
|---|---|---|
| `ticker` | `str` | Company symbol, e.g. `"AAPL"` |
| `filing_date` | `str` | Filing year, e.g. `"2024"` |
| `chunk_text` | `str` | First 300 chars of the retrieved passage |
| `rerank_score` | `float` (0.0–1.0) | BGE cross-encoder relevance score (clamped — raw scores can exceed 1.0) |

### LatencyBreakdown

Per-stage pipeline timing in milliseconds.

| Field | Description |
|---|---|
| `retrieve_ms` | Time for hybrid search across all query variants |
| `rerank_ms` | Time for BGE cross-encoder to score top-20 → top-K |
| `generate_ms` | Time for LLM to produce the answer |
| `total_ms` | Wall-clock time from request receipt to response |

`stage_dict` property returns `{"Retrieve": ms, "Rerank": ms, "Generate": ms}` for chart rendering.

### QueryResponse

Full response from `POST /query`.

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | LLM-generated answer with inline citations |
| `sources` | `list[Source]` | Retrieved chunks used as context |
| `latency_ms` | `LatencyBreakdown` | Per-stage timing |
| `provider` | `str` | LLM provider used |

`QueryResponse.from_api_dict(data)` is a class method that constructs the object from raw JSON, tolerating unknown or missing fields.

### ChatMessage

A single conversation turn stored in Streamlit `session_state`.

| Field | Type | Description |
|---|---|---|
| `role` | `str` | `"user"` or `"assistant"` |
| `content` | `str` | Message text |
| `sources` | `list[Source]` | Empty for user messages |
| `latency` | `LatencyBreakdown \| None` | None for user messages |

Properties: `has_sources → bool`, `has_latency → bool` (used in rendering logic).

### IngestRequest

Sent to `POST /ingest`.

| Field | Type | Description |
|---|---|---|
| `tickers` | `list[str]` | Company symbols to ingest (auto-uppercased, whitespace stripped) |
| `num_filings` | `int` | Filings per ticker — 1 to 5 (default: 2) |

### HealthStatus

Response from `GET /health`.

| Field | Description |
|---|---|
| `status` | `"ok"` when all components are ready |
| `vector_store` | `"chromadb"` or `"pinecone"` |
| `models_available` | List of reachable providers |
| `ollama_running` | Whether local Ollama server responded |

---

## 6. Complete Application Flow

### Happy Path (Normal Query)

```
User / Browser
    │
    │  User types "What was NVIDIA's data centre revenue in 2024?"
    │  Selects provider: Groq
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit UI  (main.py)                      │
│                                                                 │
│  sidebar.py collects: question, provider, top_k, filters        │
│  chat.py calls api_client.query() or api_client.query_stream()  │
└──────────────────────────┬──────────────────────────────────────┘
                           │  POST /query
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               FastAPI Backend  (src/api/app.py)                 │
│                                                                 │
│  1. Auth: verify X-API-Key header (no-op if API_KEY is empty)  │
│  2. Rate limit: 60 req/min per IP via slowapi                   │
│  3. Cache check: sha256(question:provider:top_k) in Redis       │
│     └─► Cache hit → return cached JSON immediately              │
│                                                                 │
│  ── Stage 1: Query Rewriting ──────────────────────────────     │
│  _get_query_variants() calls LLM with QUERY_REWRITE_PROMPT      │
│  Returns 3 variants: ["NVDA data centre revenue", ...]          │
│  Falls back to original question on any error                   │
│                                                                 │
│  ── Stage 2: Hybrid Search ─────────────────────────────────    │
│  For each query variant:                                        │
│    searcher.search(variant, TOP_K_RETRIEVAL=20, filters)        │
│      ├── BM25 keyword search → scored candidates                │
│      ├── Vector similarity search → scored candidates           │
│      └── RRF fusion → merged, deduplicated top-20              │
│  Deduplicates by chunk_id across all variants                   │
│  Result: up to 60 unique candidate chunks                       │
│                                                                 │
│  ── Stage 3: Reranking ─────────────────────────────────────    │
│  reranker.rerank(question, all_candidates, top_k=5)             │
│  BGE cross-encoder scores each (question, chunk) pair           │
│  Returns top-5 sorted by relevance                              │
│                                                                 │
│  ── Stage 4: LLM Generation ────────────────────────────────    │
│  format_context(top_docs) → structured context string           │
│  (SEC_RAG_PROMPT | llm).invoke(context, question)               │
│  Returns citation-grounded answer                               │
│                                                                 │
│  Cache write (1-hour TTL)                                       │
│  Return JSON: answer + sources + latency + provider             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                Streamlit renders response                       │
│                                                                 │
│  chat.py renders answer text                                    │
│  render_sources(): source cards with ticker badges              │
│  charts.py: latency breakdown bar + relevance score bar         │
└─────────────────────────────────────────────────────────────────┘
```

### Streaming Query Path

When the user enables streaming in the sidebar:

1. `api_client.query_stream()` opens a connection to `POST /query/stream`.
2. The API runs Stages 1–3 identically (retrieve + rerank, blocking).
3. Stage 4 uses `llm.astream()` instead of `llm.invoke()`.
4. Each token is emitted as a Server-Sent Event: `data: {"token": "NVIDIA"}\n\n`
5. A final SSE carries sources and latency: `data: {"done": true, "sources": [...], "latency_ms": {...}}\n\n`
6. `_render_streaming()` in `chat.py` appends each token to the displayed message in real time.

### Ingestion Path (New Filing)

1. User submits tickers in the sidebar ingest form (or runs `make ingest` / `scripts/ingest.py`).
2. `POST /ingest` creates a job. If Redis is available, the job goes into the RQ queue (survives restarts). Otherwise, it runs as a FastAPI `BackgroundTask`.
3. `_run_ingestion_pipeline()` orchestrates:
   - `download_10k_filings()` → raw filing text via SEC EDGAR
   - `SECChunker.chunk_document()` → 512-char chunks with metadata
   - `VectorStoreManager.upsert_chunks()` → embed + store in Pinecone/ChromaDB
   - `HybridSearcher.build_bm25_index()` + `save_index()` → persist BM25 to disk
4. Poll `GET /ingest/{job_id}` for RQ job status.

---

## 7. Data Ingestion Pipeline

**Files:** `src/ingestion/downloader.py`, `src/ingestion/chunker.py`, `src/ingestion/embedder.py`

### Stage 1 — Download

`download_10k_filings(tickers, num_filings)` uses the `sec-edgar-downloader` library to fetch 10-K filings from SEC EDGAR. Files are saved to:

```
data/sec_filings/sec-edgar-filings/TICKER/10-K/ACCESSION_NUMBER/full-submission.txt
```

`extract_text_from_filing(path)` handles three filing formats:
- **SGML envelope** (standard EDGAR format): Parses `<DOCUMENT><TYPE>10-K<TEXT>...</TEXT>` blocks
- **HTML**: Strips tags, collapses whitespace
- **PDF**: Uses PyMuPDF for text extraction

Returns `(text: str, metadata: dict)` where metadata includes `ticker`, `filing_date`, `form_type`, `source`.

### Stage 2 — Chunk

`SECChunker` wraps LangChain's `RecursiveCharacterTextSplitter`:

| Parameter | Value | Why |
|---|---|---|
| `chunk_size` | 512 characters | Fits in most embedding model context windows |
| `chunk_overlap` | 64 characters | Prevents answers from being split at chunk boundaries |
| Separators | `\n\n` → `.` → ` ` → `""` | Tries to break at natural boundaries first |
| Min length filter | 50 characters | Removes headers, page numbers, and empty chunks |

Each chunk gets a `chunk_id` (UUID) for deduplication during ingestion and retrieval.

`chunk_document(text, metadata)` returns a list of LangChain `Document` objects, each with the chunk text in `page_content` and filing metadata in `.metadata`.

### Stage 3 — Embed and Store

`get_embedder()` auto-selects the embedder:
- **JinaEmbedder** (if `JINA_API_KEY` is set): Jina AI API, 768-dimensional vectors, 8,192-token context window, 1M tokens/month free tier.
- **LocalEmbedder** (fallback): `all-MiniLM-L6-v2` via sentence-transformers, 384-dimensional vectors, runs entirely on CPU, no API key needed.

`VectorStoreManager.upsert_chunks(chunks)` embeds all chunks in batches and stores them. Supports:
- **ChromaDB** (`USE_LOCAL_CHROMA=true`): Persistent local database in `chroma_db/`. No cloud account needed.
- **Pinecone** (default): Serverless cloud vector store. Requires `PINECONE_API_KEY`.

`similarity_search(query_embedding, top_k, filters)` retrieves the nearest vectors with optional metadata filtering by `ticker` or `filing_date`.

---

## 8. Core Modules

Each module under `src/` is self-contained and importable independently.

---

### 8.1 Downloader

**File:** `src/ingestion/downloader.py`

**What it solves:** SEC filings on EDGAR arrive in a complex SGML multi-document envelope format. This module handles all the format complexity — SGML, HTML, and PDF — and returns clean plain text ready for chunking.

#### Key Functions

```python
def download_10k_filings(tickers: list[str], num_filings: int = 2) -> list[Path]:
    """Download 10-K filings from SEC EDGAR. Returns list of filing directory paths."""

def extract_text_from_filing(filing_path: Path) -> tuple[str, dict]:
    """Extract clean text and metadata from a filing file. Returns (text, metadata)."""
```

#### SGML Parsing Detail

EDGAR's full-submission files wrap every document in an SGML envelope:

```
<DOCUMENT>
<TYPE>10-K
<TEXT>
... actual 10-K content ...
</TEXT>
</DOCUMENT>
```

`_extract_from_sgml()` finds the `<TYPE>10-K` block specifically and extracts only that section, avoiding exhibits, covers, and other attached documents that would add noise.

---

### 8.2 Chunker

**File:** `src/ingestion/chunker.py`

**What it solves:** LLM context windows have token limits. A full 10-K filing is 150–300 pages — far too large to fit in a prompt. This module splits filings into 512-character chunks that each fit easily within any embedding model's context window, while preserving enough surrounding text for the LLM to interpret each chunk correctly.

#### Why 512 Characters?

512 characters ≈ 100–130 tokens, comfortably within Jina's 8,192-token limit and well under sentence-transformers' 256-token window. Longer chunks would overflow the embedding model; shorter chunks would split individual sentences and lose context.

#### Separator Hierarchy

The chunker tries to split at natural boundaries before falling back to arbitrary character positions:

```
1. \n\n  — paragraph breaks (preserve section structure)
2. .     — sentence endings (keep sentences intact)
3. " "   — word boundaries (never split mid-word)
4. ""    — character boundary (last resort)
```

#### Chunk ID Generation

Each chunk gets a deterministic `chunk_id` derived from the file path and chunk index. This enables the vector store to detect and skip already-indexed chunks during repeated ingestion runs.

---

### 8.3 Embedder

**File:** `src/ingestion/embedder.py`

**What it solves:** This is the single place in the codebase that knows which embedding model to use and how to talk to the vector store. Both ingestion and retrieval go through this module.

#### Embedder Selection

```python
def get_embedder() -> Embeddings:
    if settings.JINA_API_KEY:
        return JinaEmbedder()   # 768-dim cloud embeddings
    return LocalEmbedder()      # 384-dim local sentence-transformers
```

The embedder returned is a LangChain `Embeddings`-compatible object, so retrieval code never needs to know which backend is active.

#### JinaEmbedder

- **Model:** `jina-embeddings-v2-base-en`
- **Dimensions:** 768 (2× richer than MiniLM)
- **Context:** 8,192 tokens (handles long financial passages without truncation)
- **Cost:** Free tier includes 1M tokens/month

#### LocalEmbedder

- **Model:** `all-MiniLM-L6-v2` (90MB download)
- **Dimensions:** 384
- **Runs on:** CPU — no GPU required
- **Cost:** Free, no internet required after first download

#### VectorStoreManager

The `VectorStoreManager` abstracts over Pinecone and ChromaDB:

```python
manager = VectorStoreManager(use_local=True)   # ChromaDB
manager = VectorStoreManager(use_local=False)  # Pinecone
manager.upsert_chunks(documents)               # Embed + store
manager.similarity_search(query_vec, top_k=20) # Vector retrieval
manager.get_collection_stats()                 # Count + dimension
```

Both backends return the same `list[Document]` type so retrieval code is backend-agnostic.

---

### 8.4 Hybrid Search

**File:** `src/retrieval/hybrid_search.py`

**What it solves:** Vector search misses exact numeric matches ("Q3 2024 $107 billion"). BM25 keyword search misses synonyms ("earnings" vs "profit"). Hybrid search catches both, consistently outperforming either method alone by 15–20% on recall benchmarks.

#### How RRF Works

Reciprocal Rank Fusion combines the two ranked lists using this formula:

```
score(doc) = Σ  1 / (k + rank)
```

where `k = 60` (smooths the score distribution) and `rank` is the document's position in each list. A document ranked #1 in both lists scores `1/(60+1) + 1/(60+1) = 0.0328`, beating any document that only appears in one list.

#### BM25 Persistence

Building the BM25 index from scratch takes ~20 seconds on a cold start (all 13,432 chunks must be tokenized). After ingestion:

```python
searcher.save_index(Path("data/bm25_index.pkl"))
```

This serializes the index atomically via `joblib + shutil.move(tmp → final path)` — a partial write can never corrupt the index. On the next startup, the API loads the persisted index in ~1 second.

#### Metadata Filtering

```python
results = searcher.search("Apple revenue", top_k=10, filters={"ticker": "AAPL"})
```

Filters are applied at the vector store level before RRF fusion. BM25 results are post-filtered by metadata to match.

#### Function Signatures

```python
class HybridSearcher:
    def build_bm25_index(self, documents: list[Document]) -> None: ...
    def bm25_search(self, query: str, top_k: int) -> list[Document]: ...
    def vector_search(self, query: str, top_k: int, filters: dict | None) -> list[Document]: ...
    def reciprocal_rank_fusion(self, bm25_results, vector_results, top_k) -> list[Document]: ...
    def search(self, query: str, top_k: int, filters: dict | None) -> list[Document]: ...
    def save_index(self, path: Path) -> None: ...
    def load_index(self, path: Path) -> bool: ...
```

---

### 8.5 Reranker

**File:** `src/retrieval/reranker.py`

**What it solves:** Hybrid search retrieves 20 candidate chunks quickly but imprecisely — it scores query and document separately. The reranker re-scores every candidate by reading query and document together in a single neural network pass, producing much more accurate relevance scores. The top-5 chunks it selects are what actually reach the LLM's context window.

#### Model: BAAI/bge-reranker-base

- **Type:** Cross-encoder (query + document processed jointly)
- **Parameters:** 278M
- **Hardware:** CPU (no GPU required)
- **Free:** Open-source, downloaded from Hugging Face on first use (~500MB)
- **Performance:** ~15% faithfulness improvement over bi-encoder retrieval alone

#### How Cross-Encoding Differs from Bi-Encoding

A bi-encoder (used for retrieval) encodes query and document **separately** into vectors and computes cosine similarity. This is fast but misses fine-grained interactions between query words and document words.

A cross-encoder feeds `[CLS] query [SEP] document [SEP]` through the model as a single sequence. Every query token can attend to every document token — capturing nuances like "Apple revenue" matching a paragraph that says "AAPL reported net sales" rather than just a sentence mentioning "revenue."

#### Function Signature

```python
class BGEReranker:
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Score all (query, doc) pairs, return top-k sorted by relevance.
        Adds rerank_score to each document's metadata (clamped 0.0–1.0 in Source schema).
        """
```

#### Pipeline Bottleneck Note

The reranker is the **dominant latency bottleneck** at ~46% of total pipeline time (roughly 7.5s for 20 docs on CPU). This is expected — cross-encoding is computationally expensive. GPU deployment would reduce total latency by ~40%.

---

### 8.6 LLM Router

**File:** `src/generation/llm_router.py`

**What it solves:** Three different LLM providers are supported (Groq, Gemini, Ollama). This module hides all provider-specific wiring behind a unified `BaseLanguageModel` interface. Switching providers requires no changes in `app.py` or anywhere else.

#### Available Providers

| Provider | Model | Speed | Cost | Best For |
|---|---|---|---|---|
| **Groq** | llama-3.3-70b | ~500 tok/s | Free tier | Default — best speed/quality balance |
| **Gemini** | gemini-1.5-flash | ~100 tok/s | Free tier | Complex multi-company queries |
| **Ollama** | llama3.2 (local) | ~30 tok/s | Free | Air-gapped / offline deployments |

#### Function Signatures

```python
def get_llm(provider: str) -> BaseLanguageModel:
    """Return a blocking LLM client for the given provider."""

def get_streaming_llm(provider: str) -> BaseLanguageModel:
    """Return a streaming-enabled LLM client for /query/stream."""
```

Both functions raise `ValueError` if the required API key is not set in `settings`. The returned object is a standard LangChain `BaseLanguageModel` — compose it with any prompt template using `prompt | llm`.

#### Blocking vs. Async

LLM inference is synchronous (blocking). In async FastAPI routes, all LLM calls are wrapped with `run_in_threadpool()` to prevent blocking the event loop under concurrent requests:

```python
response = await run_in_threadpool(
    lambda: llm.invoke(QUERY_REWRITE_PROMPT.format_messages(question=question))
)
```

Streaming uses `llm.astream()` which is natively async and does not need `run_in_threadpool`.

---

### 8.7 Prompt Templates

**File:** `src/generation/prompt_templates.py`

**What it solves:** All prompt strings live in one file. No inline f-string prompts anywhere else. This makes prompts easy to find, compare, version, and iterate without hunting through application code.

#### SEC_RAG_PROMPT — The Main Q&A Prompt

This is the most important prompt in the system. It tells the LLM how to behave as a financial analyst:

```
[SYSTEM]
You are a financial analyst AI specializing in SEC filings analysis.

Your job: Answer questions ONLY using the provided SEC filing excerpts below.

RULES YOU MUST FOLLOW:
1. Use ONLY information explicitly stated in the context — never add outside knowledge
2. Cite your source for every factual claim: format as [TICKER - FORM - DATE]
   Example: "Apple reported $383 billion in revenue [AAPL - 10-K - 2024]"
3. If the answer is not in the provided context, say exactly:
   "This information is not available in the provided SEC filings."
4. Use precise financial language: "$4.2 billion" not "4200000000"
5. Never speculate, estimate, or reason beyond what the context states
6. If multiple filings are provided, clearly distinguish which company you're citing

CONTEXT FROM SEC FILINGS:
{context}

[HUMAN]
{question}
```

Rule 3 is the key anti-hallucination constraint: the LLM is explicitly instructed to say "not available" rather than make up an answer.

#### QUERY_REWRITE_PROMPT — Multi-Variant Search

Users phrase questions differently from how filings are written. "How much does Amazon make from AWS?" fails to retrieve the relevant chunk titled "Amazon Web Services net sales." Query rewriting generates 3 alternative phrasings using SEC/financial terminology:

```
Input: "How much does Amazon make from AWS?"
Output: [
  "Amazon Web Services net sales 2024",
  "AWS revenue annual 10-K",
  "Amazon cloud computing segment revenue"
]
```

The original question and all 3 variants are searched in parallel. Chunks that appear across multiple variants are deduplicated by `chunk_id`.

#### CITATION_EXTRACTION_PROMPT — Structured Citations

Extracts structured citation objects from the LLM answer for downstream processing. Returns a JSON array of `{ticker, form_type, date, quote}` objects.

#### `format_context(documents)` — Context Formatter

Formats retrieved chunks with labeled headers so the LLM knows which company and filing each chunk came from:

```
[Source: AAPL-10-K-2024 | Chunk 1 | Relevance: 0.97]
Apple's total net sales were $383.3 billion in fiscal 2024...

[Source: MSFT-10-K-2024 | Chunk 2 | Relevance: 0.82]
Microsoft Azure cloud revenue grew 29%...
```

---

## 9. API Reference

The FastAPI server exposes six endpoints. Interactive docs auto-generated at `http://localhost:8000/docs`.

### `GET /health`

Public health check — no authentication required.

**Response:**
```json
{
  "status": "ok",
  "vector_store": "chromadb",
  "models_available": ["groq", "gemini"],
  "ollama_running": false,
  "redis_connected": false
}
```

### `GET /stats`

Vector store statistics. Requires `X-API-Key` header.

**Response:**
```json
{
  "count": 13432,
  "dimension": 768,
  "store": "chromadb"
}
```

### `POST /query`

Full RAG pipeline. Requires `X-API-Key` header. Rate limited to 60 requests/minute per IP.

**Request body:**
```json
{
  "question": "What was NVIDIA's data centre revenue in 2024?",
  "provider": "groq",
  "top_k": 5,
  "ticker": "NVDA",
  "year": 2024
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `question` | string | Yes | 1–2000 characters |
| `provider` | string | No | `groq` \| `gemini` \| `ollama` |
| `top_k` | integer | No | 1–10 (default 5) |
| `ticker` | string | No | Uppercase ticker symbol |
| `year` | integer | No | 2000–2030 |

**Response:**
```json
{
  "answer": "NVIDIA's Data Center segment revenue reached $47.5 billion in fiscal 2024 [NVDA - 10-K - 2024]...",
  "sources": [
    {
      "ticker": "NVDA",
      "filing_date": "2024",
      "chunk_text": "Data Center revenue was $47.5 billion, a 217% increase year-over-year...",
      "rerank_score": 0.97
    }
  ],
  "latency_ms": {
    "retrieve_ms": 3200,
    "rerank_ms": 7400,
    "generate_ms": 2500,
    "total_ms": 13100
  },
  "provider": "groq"
}
```

**Error responses:**

| Status | Cause |
|---|---|
| `401` | Invalid or missing `X-API-Key` header (when `API_KEY` is set) |
| `422` | Request body fails Pydantic validation |
| `429` | Rate limit exceeded (60 req/min) |
| `503` | ML components not initialized (check `/health`) |

### `POST /query/stream`

Same as `/query` but streams the answer token-by-token via Server-Sent Events. Rate limited to 30 requests/minute.

**SSE event format:**
```
data: {"token": "NVIDIA"}

data: {"token": "'s"}

data: {"done": true, "sources": [...], "latency_ms": {...}}
```

### `POST /ingest`

Kick off a filing ingestion job. Uses RQ queue when Redis is available; falls back to FastAPI `BackgroundTasks`.

**Request body:**
```json
{
  "tickers": ["AAPL", "MSFT"],
  "num_filings": 2
}
```

**Response:**
```json
{
  "job_id": "ingest_1705000000",
  "status": "queued",
  "tickers": ["AAPL", "MSFT"],
  "message": "Ingesting 2 tickers via job queue.",
  "poll_url": "/ingest/ingest_1705000000"
}
```

### `GET /ingest/{job_id}`

Poll ingestion job status. Requires Redis.

**Response:**
```json
{
  "job_id": "ingest_1705000000",
  "status": "finished",
  "result": {"tickers": ["AAPL", "MSFT"], "vectors_stored": 1847},
  "error": null
}
```

---

## 10. Streamlit UI

**Files:** `main.py`, `src/ui/chat.py`, `src/ui/sidebar.py`, `src/ui/charts.py`, `src/utils/api_client.py`

### Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  Header: SEC Filing Intelligence + pipeline stage badges             │
├─────────────────────────┬────────────────────────────────────────────┤
│ SIDEBAR                 │ MAIN PANEL                                 │
│                         │                                            │
│ LLM Provider selector   │ Stat cards:                                │
│  ├─ Groq (default)      │  API • Groq • Gemini • Pinecone (13,432)   │
│  ├─ Google Gemini       │                                            │
│  └─ Ollama              │ Dataset info expander                      │
│                         │                                            │
│ Top-K slider (1–10)     │ Chat history:                              │
│                         │  User: "What is Apple's revenue?"          │
│ Streaming toggle        │  Assistant: "Apple reported $383B..."      │
│                         │    Sources expander:                       │
│ Sample questions:       │      [AAPL 2024] Revenue chunk (0.97)      │
│  ├─ Revenue comparison  │    Latency chart (Retrieve/Rerank/Generate)│
│  ├─ Risk factors        │    Source distribution donut               │
│  └─ Cloud growth        │                                            │
│                         │ Chat input: "Ask a question..."            │
│ Ingest form             │                                            │
│ Health check button     │                                            │
└─────────────────────────┴────────────────────────────────────────────┘
```

### APIClient (`src/utils/api_client.py`)

The UI never calls `requests` or `httpx` directly. All HTTP calls go through `APIClient`, a typed singleton cached with `@st.cache_resource`:

```python
client = APIClient(base_url=settings.API_URL)
response: QueryResponse = client.query(QueryRequest(question="...", provider="groq"))
```

`parse_stream_response()` accepts callbacks for token-by-token streaming:

```python
client.parse_stream_response(
    req,
    on_token=lambda tok: placeholder.write(tok),
    on_done=lambda resp: render_sources(resp.sources),
    on_error=lambda err: st.error(err),
)
```

### Charts (`src/ui/charts.py`)

All charts use Plotly with a dark theme matching the UI's `#080d14` background.

| Chart | When shown | What it shows |
|---|---|---|
| Latency breakdown | After each response | Horizontal bar: Retrieve / Rerank / Generate ms |
| Source distribution | When ≥2 sources | Donut by ticker symbol |
| Relevance scores | When sources exist | Horizontal bar per source (BGE rerank score) |

---

## 11. Setup & Running the Project

### Prerequisites

- Python 3.11 or higher
- At least one of: Groq API key (free), Google API key (free), or Ollama running locally
- Optional: Pinecone API key (otherwise ChromaDB is used automatically)

### Step 1 — Install dependencies

```bash
cd sec-rag

# Preferred: uv
uv pip install -r requirements.txt

# Alternative: pip
pip install -r requirements.txt
```

### Step 2 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```
# Pick one LLM provider
GROQ_API_KEY=gsk_...

# ChromaDB is the default local vector store — no key needed
USE_LOCAL_CHROMA=true

# Vector store selection
VECTOR_STORE=chroma
```

See the full reference in [Section 14](#14-environment-variables).

### Step 3 — Ingest SEC filings

Run the full ingestion pipeline (download → chunk → embed → store):

```bash
# Download and embed 2 filings for each company
make ingest

# Or run manually with specific tickers
python scripts/ingest.py full --tickers AAPL MSFT NVDA --num-filings 2

# Check what's been indexed
make stats
```

Ingestion takes 10–30 minutes per ticker depending on filing size and embedding speed.

### Step 4 — Start the API server

```bash
make api
# Equivalent to: uvicorn src.api.app:app --reload --port 8000
```

On startup you will see the component loading sequence in the terminal:

```
INFO  Starting SEC RAG API — loading components...
INFO  ML components loaded
INFO  No persisted BM25 index found — index will build on first ingest
INFO  Application startup complete.
```

The API is now at `http://localhost:8000`. Open `http://localhost:8000/docs` for interactive Swagger UI.

### Step 5 — Start the Streamlit UI

In a second terminal:

```bash
make ui
# Equivalent to: streamlit run main.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

### Step 6 — Try a query

In the UI, click a sample question from the sidebar, or type your own:

```
"What was Apple's total revenue in fiscal 2024?"
"How does Azure revenue compare to AWS revenue?"
"What are NVIDIA's key risk factors related to AI?"
```

Or call the API directly:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apple revenue in 2024?", "provider": "groq"}'
```

### Docker Compose (Full Stack)

To run all services — Redis, API, RQ worker, and Streamlit UI — together:

```bash
make docker-up
# Equivalent to: docker compose up --build
```

Services:
- `redis` on port 6379 (enables query caching and async job queue)
- `api` on port 8000
- `worker` (RQ worker consuming the ingestion queue)
- `streamlit` on port 8501

To stop:

```bash
make docker-down
```

### Make Targets Reference

| Target | What it does |
|---|---|
| `make api` | Start FastAPI dev server on port 8000 |
| `make ui` | Start Streamlit on port 8501 |
| `make ingest` | Run full ingestion for default tickers |
| `make stats` | Show vector store document count |
| `make test` | Run unit + integration tests |
| `make test-unit` | Unit tests only |
| `make test-cov` | Tests with HTML coverage report |
| `make eval` | Run RAGAS evaluation suite |
| `make gen-testset` | Generate 200 synthetic Q&A pairs |
| `make lint` | ruff check + mypy |
| `make format` | ruff format (in-place) |
| `make docker-up` | Build and start all Docker services |
| `make docker-down` | Stop all Docker services |
| `make docker-logs` | Stream logs from all containers |

---

## 12. Testing

**Test files:** `tests/unit/`, `tests/integration/`, `tests/e2e/`

### Run Tests

```bash
make test                            # unit + integration (default)
pytest tests/unit/ -v                # unit only
pytest tests/integration/ -v        # integration only
pytest tests/e2e/ -m e2e -v         # e2e (requires docker compose up)
pytest --cov=src --cov-report=html   # with HTML coverage report
```

### Shared Fixtures (`tests/conftest.py`)

Fixtures defined here are automatically available in all test files without imports.

| Fixture | Type | What it provides |
|---|---|---|
| `sample_documents` | `list[Document]` | Five realistic SEC filing chunks (AAPL, MSFT, GOOGL) |
| `mock_embedder` | `MagicMock` | Returns a zero 768-dim vector; no API key needed |
| `mock_vector_store` | `MagicMock` | Returns first 3 sample docs for any similarity search |

### Unit Tests

| File | Tests |
|---|---|
| `test_schemas.py` | `Source` score clamping, `QueryRequest` validation, `IngestRequest` ticker normalisation, `ChatMessage` computed properties, `QueryResponse.from_api_dict()` |
| `test_chunker.py` | Chunk count, minimum length filter, metadata propagation, chunk_id uniqueness |
| `test_hybrid_search.py` | RRF formula correctness, deduplication by chunk_id, empty corpus handling, metadata filter application |
| `test_reranker.py` | Top-k selection, score clamping, behaviour with mock CrossEncoder |
| `test_llm_router.py` | Provider dispatch, error on missing API key, streaming flag propagation |

### Integration Tests (`tests/integration/test_api.py`)

All integration tests use `httpx.AsyncClient` with the FastAPI app instance — no live network calls, no running server needed.

| Test | What it verifies |
|---|---|
| `test_health_returns_ok` | `GET /health` returns 200 with `status: ok` |
| `test_query_requires_components` | `POST /query` returns 503 when components are `None` |
| `test_auth_rejects_bad_key` | Returns 401 when `API_KEY` is set and header is wrong |
| `test_rate_limit_headers_present` | `X-RateLimit-*` headers appear on `/query` responses |
| `test_ingest_returns_job` | `POST /ingest` returns a job object with status field |
| `test_stream_endpoint_content_type` | `/query/stream` response has `text/event-stream` media type |

### E2E Tests (`tests/e2e/test_pipeline.py`)

Marked with `@pytest.mark.e2e`. Run only with `pytest -m e2e`. Require `docker compose up` to be running. Tests verify the full pipeline against the live stack — actual LLM calls, real vector store, real streaming.

### Testing Rules

- Tests must not make real network calls. Mock `httpx`, `chromadb`, and `langchain` at boundaries.
- Use `monkeypatch` or `unittest.mock.MagicMock`; never `pytest-mock`.
- Fixtures shared across suites belong in `tests/conftest.py`.
- Coverage targets: `src/core/` at 100%; all other modules ≥ 70%.

---

## 13. Evaluation

**Files:** `evals/generate_testset.py`, `evals/run_ragas.py`

The evaluation pipeline measures whether the RAG system is actually producing correct, relevant, grounded answers — not just whether the code runs.

### RAGAS Metrics

| Metric | Definition | Target |
|---|---|---|
| **Faithfulness** | Fraction of answer claims supported by retrieved context | ≥ 0.85 |
| **Answer Relevancy** | How directly the answer addresses the question | ≥ 0.80 |
| **Context Recall** | Fraction of ground-truth facts retrieved | ≥ 0.70 |
| **Context Precision** | Fraction of retrieved chunks that are actually relevant | ≥ 0.75 |

### Generating the Test Set

```bash
make gen-testset
# Runs: python evals/generate_testset.py
```

This generates 200 synthetic Q&A pairs from existing indexed chunks. An LLM creates factual, comparative, risk-related, and segment-specific questions from the actual filing text — ensuring questions have verifiable answers in the corpus. Output is saved to `evals/results/testset_TIMESTAMP.json`.

Question types generated:
- **Factual:** "What was AAPL's gross margin in fiscal 2024?"
- **Comparative:** "Which company had higher cloud revenue growth: MSFT or AMZN?"
- **Risk:** "What supply chain risks does AAPL identify in its 2024 10-K?"
- **Segment:** "What percentage of GOOGL revenue came from advertising in 2024?"

### Running the Evaluation

```bash
make eval
# Runs: python evals/run_ragas.py
```

This runs the full RAG pipeline on the test set, collects answers and retrieved context, then scores them using RAGAS. Results are saved as a timestamped CSV.

```bash
# Compare providers
python evals/run_ragas.py --compare-providers
```

### CI Quality Gate

`.github/workflows/nightly_ragas.yml` runs the evaluation nightly and fails the workflow if faithfulness drops below 0.85 or answer relevancy below 0.80. This prevents silent quality regressions from model or retrieval changes.

### Baseline vs. Reranked Results

| Metric | Baseline (no reranker) | With BGE Reranker | Target |
|---|---|---|---|
| Faithfulness | 0.61 | 0.91 | ≥ 0.85 |
| Answer Relevancy | 0.58 | 0.83 | ≥ 0.80 |
| Context Recall | 0.52 | 0.78 | ≥ 0.70 |
| Context Precision | 0.49 | 0.76 | ≥ 0.75 |

---

## 14. Environment Variables

See `.env.example` for the full list with comments. All variables are loaded through `src/core/config.py` via `pydantic-settings` — never call `os.environ` directly.

### Required

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | Default LLM provider: `groq` \| `gemini` \| `ollama` |
| `VECTOR_STORE` | Vector store backend: `chroma` \| `pinecone` |

### LLM Providers (at least one required)

| Variable | Provider | Notes |
|---|---|---|
| `GROQ_API_KEY` | Groq | Free tier at console.groq.com |
| `GOOGLE_API_KEY` | Gemini | Free tier at aistudio.google.com |
| `OLLAMA_BASE_URL` | Ollama | Default: `http://localhost:11434` |

### Embeddings

| Variable | Description |
|---|---|
| `JINA_API_KEY` | Jina AI embeddings (768-dim). Falls back to LocalEmbedder if not set. |

### Vector Store

| Variable | When Required | Description |
|---|---|---|
| `PINECONE_API_KEY` | Pinecone | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | Pinecone | Target index name |
| `USE_LOCAL_CHROMA` | ChromaDB | `true` to use local ChromaDB |

### Security & Performance

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | `""` | `X-API-Key` header value. No-op (open) when empty. |
| `ALLOWED_ORIGINS` | `*` | CORS origins — comma-separated, restrict in production |
| `REDIS_URL` | `""` | Redis connection URL. Enables caching and RQ job queue. |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K_RETRIEVAL` | `20` | Hybrid search candidates before reranking |
| `TOP_K_RERANK` | `5` | Final chunks passed to LLM |
| `BM25_INDEX_PATH` | `data/bm25_index.pkl` | BM25 persistence path |

### Observability

| Variable | Description |
|---|---|
| `LANGCHAIN_API_KEY` | LangSmith API key for LLM call tracing |
| `LANGCHAIN_TRACING_V2` | `true` to enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | LangSmith project name |

---

## 15. How to Extend This Project

### Add a new company / filing

Trigger ingestion through any of:
- Streamlit sidebar → Ingest form → enter ticker → submit
- `make ingest` (uses tickers in Makefile)
- `python scripts/ingest.py full --tickers TSLA --num-filings 3`
- `POST /ingest` with `{"tickers": ["TSLA"], "num_filings": 3}`

No code changes required. The pipeline handles download, chunking, embedding, and BM25 rebuild automatically.

### Add a new LLM provider

1. Open `src/generation/llm_router.py`.
2. Add a new `elif provider == "new_provider":` branch in `get_llm()` and `get_streaming_llm()` that returns a LangChain `BaseLanguageModel`-compatible object.
3. Add the new key to `.env.example` and `src/core/config.py`.
4. Update the `provider` field pattern in `src/core/schemas.py`: `pattern="^(groq|gemini|ollama|new_provider)$"`.

### Add support for 10-Q quarterly filings

1. In `src/ingestion/downloader.py`, pass `form_type="10-Q"` to the downloader function.
2. Add a `quarter` field to the metadata extracted by `_extract_metadata_from_path()`.
3. Add `quarter: Optional[int]` to `QueryRequest` in `src/core/schemas.py` and handle it in `_build_filters()` in `app.py`.
4. No chunking, embedding, or retrieval changes needed.

### Improve chunking for financial tables

Financial tables (revenue breakdowns, balance sheets) are chunked as plain text, which can split a table mid-row. To fix this:

1. In `src/ingestion/downloader.py`, detect `<table>` HTML elements during extraction.
2. Extract tables as atomic units (don't split them).
3. Pass them to `SECChunker` with a `is_table=True` metadata flag.
4. In `SECChunker.chunk_document()`, skip splitting for documents with this flag.

### Add a new validation rule for responses

Add business-rule validation at the API layer. In `src/api/app.py`, after `stage 4 — Generate`:

```python
# Example: flag if answer is suspiciously short
if len(answer) < 50:
    logger.warning("LLM returned unexpectedly short answer — may indicate failed retrieval")
    result["requires_review"] = True
```

### Add cost tracking per query

1. Import `tiktoken` in `app.py`.
2. Count `context_text` tokens (input) and `answer` tokens (output) using `tiktoken.encoding_for_model()`.
3. Apply the Groq pricing (or whichever provider was used) to compute USD cost.
4. Add a `cost_info` field to the response dict and to `QueryResponse` in `schemas.py`.

### Switch from ChromaDB to Pinecone

1. Set `USE_LOCAL_CHROMA=false` in `.env`.
2. Set `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`.
3. Restart the API server. `VectorStoreManager` auto-selects Pinecone on init.
4. Re-run ingestion — Pinecone and ChromaDB have separate indexes; you must re-embed.

### Change the embedding model

1. Update the embedder in `src/ingestion/embedder.py`.
2. Change the embedding dimension in `VectorStoreManager` (Pinecone index must match the new dimension exactly — create a new index and re-ingest).
3. ChromaDB will auto-recreate its collection with the new dimension on first write.

> **Warning:** Existing vectors are incompatible with a new embedding model. You must delete the old index and re-embed everything.
