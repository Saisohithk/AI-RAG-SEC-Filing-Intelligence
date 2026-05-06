"""
src/api/app.py — FastAPI application for the SEC RAG system.

Endpoints:
  POST /query             — Full pipeline: retrieve + rerank + generate (cached)
  POST /query/stream      — Same but streams tokens as Server-Sent Events
  POST /ingest            — Kick off ingestion job (RQ when Redis available, else background task)
  GET  /ingest/{job_id}   — Poll ingestion job status (requires Redis)
  GET  /health            — Public service health check
  GET  /stats             — Vector store statistics (authenticated)

Security:
  - X-API-Key header required on all endpoints except /health (when API_KEY is set in config)
  - 60 req/min rate limit on /query; 30 req/min on /query/stream
  - CORS restricted to settings.ALLOWED_ORIGINS

Run locally:
  uvicorn src.api.app:app --reload --port 8000
"""

import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.embeddings import Embeddings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.core.config import settings
from src.core.schemas import IngestRequest, QueryRequest
from src.generation.llm_router import get_llm, get_streaming_llm
from src.generation.prompt_templates import (
    QUERY_REWRITE_PROMPT,
    SEC_RAG_PROMPT,
    format_context,
)
from src.ingestion.chunker import SECChunker
from src.ingestion.downloader import download_10k_filings, extract_text_from_filing
from src.ingestion.embedder import VectorStoreManager, get_embedder
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import BGEReranker

logger = logging.getLogger(__name__)

# ── Global singletons ─────────────────────────────────────────────────────────
embedder: Embeddings | None = None
vector_store: VectorStoreManager | None = None
searcher: HybridSearcher | None = None
reranker: BGEReranker | None = None
redis_client = None  # redis.Redis | None
ingest_queue = None  # rq.Queue | None

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Auth dependency ───────────────────────────────────────────────────────────


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Validate X-API-Key header. No-op when settings.API_KEY is empty."""
    if not settings.API_KEY:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401, detail="Invalid or missing X-API-Key header"
        )


# ── Startup / shutdown ────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, vector_store, searcher, reranker, redis_client, ingest_queue

    logger.info("Starting SEC RAG API — loading components...")
    logger.info(repr(settings))

    try:
        embedder = get_embedder()
        vector_store = VectorStoreManager(use_local=settings.USE_LOCAL_CHROMA)
        searcher = HybridSearcher(vector_store, embedder)
        reranker = BGEReranker()

        # Load persisted BM25 index if available (avoids ~20s rebuild on cold start)
        bm25_path = Path(settings.BM25_INDEX_PATH)
        if bm25_path.exists():
            searcher.load_index(bm25_path)
        else:
            logger.info(
                "No persisted BM25 index found — index will build on first ingest"
            )

        logger.info("ML components loaded")
    except Exception as e:
        logger.error(f"Failed to initialize ML components: {e}")

    # Redis — optional; enables caching + RQ job queue
    if settings.REDIS_URL:
        try:
            import redis as redis_lib
            from rq import Queue

            redis_client = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
            redis_client.ping()
            ingest_queue = Queue(
                "ingestion", connection=redis_lib.from_url(settings.REDIS_URL)
            )
            logger.info("Redis connected — caching and job queue enabled")
        except Exception as e:
            logger.warning(f"Redis unavailable, disabling cache/queue: {e}")
            redis_client = None
            ingest_queue = None

    yield

    logger.info("Shutting down SEC RAG API...")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SEC Filing Intelligence API",
    description="RAG system for answering questions about SEC 10-K filings",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# CORS — restrict to configured origins in production
_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────────────────────


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({ms:.1f}ms)"
    )
    return response


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cache_key(question: str, provider: str, top_k: int) -> str:
    return hashlib.sha256(f"{question}:{provider}:{top_k}".encode()).hexdigest()


async def _get_query_variants(question: str, provider: str) -> list[str]:
    """Generate 3 query rewrite variants via LLM. Falls back to original on any error."""
    try:
        llm = get_llm(provider)
        response = await run_in_threadpool(
            lambda: llm.invoke(QUERY_REWRITE_PROMPT.format_messages(question=question))
        )
        content = response.content if hasattr(response, "content") else str(response)
        variants = json.loads(content)
        if isinstance(variants, list) and variants:
            return [question] + [str(v) for v in variants[:3]]
    except Exception as e:
        logger.warning(f"Query rewriting failed, using original query: {e}")
    return [question]


def _build_filters(req: QueryRequest) -> dict[str, str] | None:
    filters: dict[str, str] = {}
    if req.ticker:
        filters["ticker"] = req.ticker.upper()
    if req.year:
        filters["filing_date"] = str(req.year)
    return filters or None


def _build_sources(docs) -> list[dict]:
    return [
        {
            "ticker": doc.metadata.get("ticker", "?"),
            "filing_date": doc.metadata.get("filing_date", "?"),
            "chunk_text": doc.page_content[:300] + "...",
            "rerank_score": doc.metadata.get("rerank_score", 0.0),
        }
        for doc in docs
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    """Public health check — no authentication required."""
    ollama_running = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            ollama_running = resp.status_code == 200
    except Exception:
        pass

    available_models = ["ollama"] if ollama_running else []
    if settings.GROQ_API_KEY:
        available_models.append("groq")
    if settings.GOOGLE_API_KEY:
        available_models.append("gemini")

    return {
        "status": "ok",
        "vector_store": "chromadb" if settings.USE_LOCAL_CHROMA else "pinecone",
        "models_available": available_models,
        "ollama_running": ollama_running,
        "redis_connected": redis_client is not None,
    }


@app.get("/stats", dependencies=[Depends(verify_api_key)])
async def stats() -> dict:
    """Vector store statistics."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return vector_store.get_collection_stats()


@app.post("/query", dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def query(request: Request, req: QueryRequest) -> dict:
    """
    Answer a question using the full RAG pipeline.

    Pipeline stages (all timed):
      1. Query rewriting — 3 variants for broader recall
      2. Hybrid search   — BM25 + vector + RRF, with optional metadata filter
      3. BGE rerank      — cross-encoder top-20 → top-K
      4. LLM generate    — grounded answer with citations
    """
    if not all([embedder, vector_store, searcher, reranker]):
        raise HTTPException(
            status_code=503, detail="Components not initialized. Check /health"
        )
    assert searcher is not None
    assert reranker is not None

    # Cache check (Redis)
    cache_key = _cache_key(req.question, req.provider, req.top_k)
    if redis_client:
        try:
            cached = redis_client.get(f"query:{cache_key}")
            if cached:
                logger.info(f"Cache hit for query: {req.question[:50]!r}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}")

    total_start = time.perf_counter()
    latency: dict[str, float] = {}
    filters = _build_filters(req)

    # Stage 1 — Query rewriting → multi-variant hybrid search
    t0 = time.perf_counter()
    variants = await _get_query_variants(req.question, req.provider)
    all_candidates: list = []
    seen_ids: set[str] = set()
    for variant in variants:
        results = await run_in_threadpool(
            searcher.search, variant, settings.TOP_K_RETRIEVAL, filters
        )
        for doc in results:
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:64])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_candidates.append(doc)
    latency["retrieve_ms"] = (time.perf_counter() - t0) * 1000

    # Stage 2 — Rerank
    t0 = time.perf_counter()
    top_docs = await run_in_threadpool(
        reranker.rerank, req.question, all_candidates, req.top_k
    )
    latency["rerank_ms"] = (time.perf_counter() - t0) * 1000

    # Stage 3 — Generate
    t0 = time.perf_counter()
    llm = get_llm(req.provider)
    context_text = format_context(top_docs)
    response = (SEC_RAG_PROMPT | llm).invoke(
        {"context": context_text, "question": req.question}
    )
    answer = response.content if hasattr(response, "content") else str(response)
    latency["generate_ms"] = (time.perf_counter() - t0) * 1000
    latency["total_ms"] = (time.perf_counter() - total_start) * 1000

    result = {
        "answer": answer,
        "sources": _build_sources(top_docs),
        "latency_ms": latency,
        "provider": req.provider,
    }

    # Cache write (1-hour TTL)
    if redis_client:
        try:
            redis_client.setex(f"query:{cache_key}", 3600, json.dumps(result))
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")

    logger.info(f"Query answered in {latency['total_ms']:.0f}ms via {req.provider}")
    return result


@app.post("/query/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def query_stream(request: Request, req: QueryRequest) -> StreamingResponse:
    """
    Stream the LLM answer token-by-token via Server-Sent Events.

    SSE format:
      data: {"token": "Apple"}\\n\\n
      data: {"done": true, "sources": [...], "latency_ms": {...}}\\n\\n
    """
    if not all([embedder, vector_store, searcher, reranker]):
        raise HTTPException(status_code=503, detail="Components not initialized")
    assert searcher is not None
    assert reranker is not None

    total_start = time.perf_counter()
    latency: dict[str, float] = {}
    filters = _build_filters(req)

    # Retrieve and rerank before streaming
    t0 = time.perf_counter()
    variants = await _get_query_variants(req.question, req.provider)
    all_candidates: list = []
    seen_ids: set[str] = set()
    for variant in variants:
        results = await run_in_threadpool(
            searcher.search, variant, settings.TOP_K_RETRIEVAL, filters
        )
        for doc in results:
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:64])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_candidates.append(doc)
    latency["retrieve_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    top_docs = await run_in_threadpool(
        reranker.rerank, req.question, all_candidates, req.top_k
    )
    latency["rerank_ms"] = (time.perf_counter() - t0) * 1000

    context_text = format_context(top_docs)
    llm = get_streaming_llm(req.provider)
    sources = _build_sources(top_docs)

    async def generate_tokens():
        t_gen = time.perf_counter()
        async for chunk in llm.astream(
            SEC_RAG_PROMPT.format_messages(context=context_text, question=req.question)
        ):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"

        latency["generate_ms"] = (time.perf_counter() - t_gen) * 1000
        latency["total_ms"] = (time.perf_counter() - total_start) * 1000
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'latency_ms': latency})}\n\n"

    return StreamingResponse(generate_tokens(), media_type="text/event-stream")


@app.post("/ingest", dependencies=[Depends(verify_api_key)])
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Kick off an ingestion job. Uses RQ when Redis is available, else BackgroundTasks.
    Poll GET /ingest/{job_id} for status when using RQ.
    """
    job_id = f"ingest_{int(time.time())}"

    if ingest_queue is not None:
        # Enqueue to RQ — survives API restarts
        job = ingest_queue.enqueue(
            _run_ingestion_pipeline,
            kwargs={"tickers": req.tickers, "num_filings": req.num_filings},
            job_id=job_id,
            job_timeout=1800,
        )
        logger.info(f"Enqueued RQ job {job.id} for: {req.tickers}")
        return {
            "job_id": job.id,
            "status": "queued",
            "tickers": req.tickers,
            "message": f"Ingesting {len(req.tickers)} tickers via job queue.",
            "poll_url": f"/ingest/{job.id}",
        }

    # Fallback to in-process background task
    background_tasks.add_task(
        _run_ingestion_pipeline, tickers=req.tickers, num_filings=req.num_filings
    )
    logger.info(f"Started background ingestion {job_id} for: {req.tickers}")
    return {
        "job_id": job_id,
        "status": "started",
        "tickers": req.tickers,
        "message": f"Ingesting {len(req.tickers)} tickers in background. Check /stats for progress.",
        "poll_url": None,
    }


@app.get("/ingest/{job_id}", dependencies=[Depends(verify_api_key)])
async def ingest_status(job_id: str) -> dict:
    """Poll status of a queued ingestion job. Requires Redis."""
    if ingest_queue is None:
        raise HTTPException(
            status_code=404, detail="Job tracking requires Redis (set REDIS_URL)"
        )
    try:
        from rq.job import Job

        job = Job.fetch(job_id, connection=ingest_queue.connection)
        return {
            "job_id": job_id,
            "status": str(job.get_status()),
            "result": job.result,
            "error": str(job.exc_info) if job.exc_info else None,
        }
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")


# ── Background ingestion task ─────────────────────────────────────────────────


def _run_ingestion_pipeline(tickers: list[str], num_filings: int) -> dict:
    """
    Download SEC filings → chunk → embed → upsert → rebuild BM25 → persist index.
    Works both as an RQ job function and a FastAPI BackgroundTask coroutine.
    """
    logger.info(f"Ingestion starting for: {tickers}")
    chunker = SECChunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    stored = 0

    try:
        filing_dirs = download_10k_filings(tickers, num_filings=num_filings)
        all_chunks = []

        for ticker_dir in filing_dirs:
            filing_files = list(ticker_dir.rglob("*.htm")) + list(
                ticker_dir.rglob("*.pdf")
            )
            for filing_file in filing_files[:num_filings]:
                try:
                    text, metadata = extract_text_from_filing(filing_file)
                    chunks = chunker.chunk_document(text, metadata)
                    all_chunks.extend(chunks)
                    logger.info(f"Processed {filing_file.name}: {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Error processing {filing_file}: {e}")

        if all_chunks and vector_store:
            stored = vector_store.upsert_chunks(all_chunks)
            logger.info(f"Upserted {stored} vectors for {tickers}")

            # Rebuild and persist BM25 index
            if searcher:
                searcher.build_bm25_index(all_chunks)
                searcher.save_index(Path(settings.BM25_INDEX_PATH))
        else:
            logger.warning("No chunks to store or vector store not ready")

    except Exception as e:
        logger.error(f"Ingestion pipeline failed for {tickers}: {e}")
        raise

    return {"tickers": tickers, "vectors_stored": stored}
