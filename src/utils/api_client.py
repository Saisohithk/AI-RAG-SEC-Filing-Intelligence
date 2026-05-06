"""
src/utils/api_client.py — Typed HTTP client for the SEC RAG FastAPI backend.

Centralises all network I/O in one class so that:
  - Every caller gets typed return values (QueryResponse, HealthStatus, etc.)
  - Connection errors are handled in one place, not scattered across UI code
  - Swapping the backend URL or auth mechanism is a single-file change
  - Tests can mock this class instead of patching requests everywhere

Architecture note:
  The Streamlit frontend is a thin consumer of the FastAPI backend.
  This client is the ONLY place in the UI layer that knows the API's URL
  schema or wire format — everything else speaks in terms of domain types.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Generator

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import Timeout

from src.core.logging_config import handle_errors, log_call
from src.core.schemas import (
    ChatMessage,
    HealthStatus,
    IngestRequest,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
    Source,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT = 120  # seconds — LLM generation can be slow


class APIClient:
    """
    HTTP client for the SEC Filing Intelligence FastAPI backend.

    Instantiate once (cached with @st.cache_resource in main.py) and pass
    to UI components that need to call the API.

    Example:
        client = APIClient()
        response = client.query(QueryRequest(question="What was AAPL revenue?"))
        print(response.answer)
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        logger.debug("APIClient initialised → %s", self.base_url)

    # ------------------------------------------------------------------
    # Core endpoints
    # ------------------------------------------------------------------

    @log_call(logger)
    def query(self, request: QueryRequest) -> QueryResponse:
        """
        POST /query — full (non-streaming) RAG pipeline.

        Returns:
            QueryResponse with answer, sources, and latency breakdown.

        Raises:
            RuntimeError: on connection failure or non-200 response.
        """
        try:
            resp = self._session.post(
                f"{self.base_url}/query",
                json=request.model_dump(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return QueryResponse.from_api_dict(resp.json())
        except ReqConnectionError:
            raise RuntimeError(
                "Cannot connect to API. Start it with: uvicorn api.main:app --reload"
            )
        except Timeout:
            raise RuntimeError(f"API request timed out after {self.timeout}s")

    def query_stream(self, request: QueryRequest) -> Generator[dict, None, None]:
        """
        POST /query/stream — streaming token-by-token SSE response.

        Yields raw event dicts as they arrive from the server:
          {"token": "Apple"} for each generated token
          {"done": True, "sources": [...], "latency_ms": {...}} as the final event

        Raises:
            RuntimeError: on connection failure or non-200 response.

        Example:
            for event in client.query_stream(req):
                if event.get("done"):
                    sources = event["sources"]
                elif "token" in event:
                    print(event["token"], end="", flush=True)
        """
        try:
            with self._session.post(
                f"{self.base_url}/query/stream",
                json=request.model_dump(),
                stream=True,
                timeout=self.timeout,
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"API returned {resp.status_code}: {resp.text[:200]}"
                    )
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    if isinstance(raw_line, bytes):
                        raw_line = raw_line.decode("utf-8")
                    if raw_line.startswith("data: "):
                        try:
                            yield json.loads(raw_line[6:])
                        except json.JSONDecodeError:
                            logger.warning("Unparseable SSE line: %s", raw_line[:80])

        except ReqConnectionError:
            raise RuntimeError(
                "Cannot connect to API. Start it with: uvicorn api.main:app --reload"
            )
        except Timeout:
            raise RuntimeError(f"Streaming request timed out after {self.timeout}s")

    @handle_errors(fallback=HealthStatus(), log_level="warning")
    def health_check(self) -> HealthStatus:
        """
        GET /health — check API and model availability.

        Returns HealthStatus(status="unknown") on connection failure
        instead of raising, so the UI can display a degraded banner gracefully.
        """
        resp = self._session.get(f"{self.base_url}/health", timeout=3)
        resp.raise_for_status()
        return HealthStatus(**resp.json())

    @handle_errors(fallback={"error": "Stats unavailable"}, log_level="warning")
    def get_stats(self) -> dict:
        """GET /stats — vector store statistics."""
        resp = self._session.get(f"{self.base_url}/stats", timeout=5)
        resp.raise_for_status()
        return resp.json()

    @log_call(logger)
    def ingest(self, request: IngestRequest) -> dict:
        """
        POST /ingest — start a background ingestion job.

        Returns:
            Dict with job_id, status, and tickers.

        Raises:
            RuntimeError: on connection failure.
        """
        try:
            resp = self._session.post(
                f"{self.base_url}/ingest",
                json=request.model_dump(),
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except ReqConnectionError:
            raise RuntimeError("Cannot connect to API.")

    # ------------------------------------------------------------------
    # Convenience helpers used by UI components
    # ------------------------------------------------------------------

    def parse_stream_response(
        self,
        request: QueryRequest,
        on_token: Callable[[str], None],
        on_done: Callable[[list[Source], LatencyBreakdown], None],
        on_error: Callable[[str], None],
    ) -> str:
        """
        High-level streaming helper that calls callbacks instead of yielding.

        Args:
            request:    The query to run.
            on_token:   Called with each text token as it arrives.
            on_done:    Called once with (sources, latency) when stream finishes.
            on_error:   Called with an error message string if something fails.

        Returns:
            The full assembled answer text.
        """
        full_text = ""
        try:
            for event in self.query_stream(request):
                if event.get("done"):
                    raw_sources = event.get("sources", [])
                    sources = [Source(**s) for s in raw_sources]
                    latency_raw = event.get("latency_ms", {})
                    latency = (
                        LatencyBreakdown(**latency_raw)
                        if latency_raw
                        else LatencyBreakdown()
                    )
                    on_done(sources, latency)
                elif "token" in event:
                    token = event["token"]
                    full_text += token
                    on_token(token)
        except RuntimeError as exc:
            on_error(str(exc))
        return full_text
