"""
src/core/schemas.py — Canonical Pydantic data contracts for the SEC RAG system.

All data flowing between the UI, API client, and rendering layer is typed
here. This eliminates the silent dict[str, Any] anti-pattern: wrong keys or
types raise ValidationError immediately instead of crashing at render time.

Design principle: one schema module, imported everywhere.
  - UI reads QueryResponse / ChatMessage
  - API client produces QueryResponse from raw JSON
  - Charts consume LatencyBreakdown / Source
  - Session state is typed as list[ChatMessage]
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# API-level contracts (mirror api/main.py Pydantic models)
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Payload sent to POST /query or POST /query/stream."""

    question: str = Field(..., min_length=1, max_length=2000)
    provider: str = Field(default="groq", pattern="^(groq|gemini|ollama)$")
    top_k: int = Field(default=5, ge=1, le=10)
    ticker: Optional[str] = Field(
        default=None, pattern="^[A-Z]{1,5}$", description="Filter by ticker symbol"
    )
    year: Optional[int] = Field(
        default=None, ge=2000, le=2030, description="Filter by filing year"
    )


class IngestRequest(BaseModel):
    """Payload sent to POST /ingest."""

    tickers: list[str] = Field(..., min_length=1)
    num_filings: int = Field(default=2, ge=1, le=5)

    @field_validator("tickers")
    @classmethod
    def normalise_tickers(cls, v: list[str]) -> list[str]:
        return [t.strip().upper() for t in v if t.strip()]


# ---------------------------------------------------------------------------
# Response contracts (built from API JSON responses)
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """A single cited SEC filing chunk returned by the RAG pipeline."""

    ticker: str = "?"
    filing_date: str = "?"
    chunk_text: str = ""
    rerank_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("rerank_score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        # BGE scores can exceed 1.0 on highly relevant pairs — normalise for display
        try:
            return min(max(float(v), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0


class LatencyBreakdown(BaseModel):
    """Per-stage pipeline timing (milliseconds)."""

    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def stage_dict(self) -> dict[str, float]:
        """Return stage name → ms mapping for chart rendering."""
        return {
            "Retrieve": self.retrieve_ms,
            "Rerank": self.rerank_ms,
            "Generate": self.generate_ms,
        }


class QueryResponse(BaseModel):
    """Full response from POST /query."""

    answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    latency_ms: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    provider: str = "groq"

    @classmethod
    def from_api_dict(cls, data: dict) -> "QueryResponse":
        """Construct from raw API JSON, tolerating unknown fields."""
        sources = [Source(**s) for s in data.get("sources", [])]
        latency_raw = data.get("latency_ms", {})
        latency = LatencyBreakdown(**latency_raw) if latency_raw else LatencyBreakdown()
        return cls(
            answer=data.get("answer", ""),
            sources=sources,
            latency_ms=latency,
            provider=data.get("provider", "groq"),
        )


# ---------------------------------------------------------------------------
# UI-level contracts
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single turn in the conversation history stored in session_state."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    sources: list[Source] = Field(default_factory=list)
    latency: Optional[LatencyBreakdown] = None

    @property
    def has_sources(self) -> bool:
        return bool(self.sources)

    @property
    def has_latency(self) -> bool:
        return self.latency is not None and self.latency.total_ms > 0


class HealthStatus(BaseModel):
    """Response from GET /health."""

    status: str = "unknown"
    vector_store: str = "unknown"
    models_available: list[str] = Field(default_factory=list)
    ollama_running: bool = False

    @property
    def is_healthy(self) -> bool:
        return self.status == "ok"


class IngestJobResponse(BaseModel):
    """Response from POST /ingest and GET /ingest/{job_id}."""

    job_id: str
    status: str  # "queued" | "started" | "finished" | "failed"
    tickers: list[str]
    message: str
    poll_url: Optional[str] = None
