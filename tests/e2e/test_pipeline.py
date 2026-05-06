"""
tests/e2e/test_pipeline.py — End-to-end smoke test for the full RAG pipeline.

Requires a running docker-compose stack:
  docker-compose up -d

Run explicitly with:
  pytest tests/e2e/ -m e2e -v

Excluded from normal CI (mark: e2e). Runs in the nightly CI job against
docker-compose with a real ChromaDB container.
"""

import time

import httpx
import pytest

BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0


pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def api():
    """Verify the API is reachable before running any e2e tests."""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=10.0)
        resp.raise_for_status()
    except Exception as exc:
        pytest.skip(f"API not reachable at {BASE_URL}: {exc}")
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


class TestHealthE2E:
    def test_health_returns_ok(self, api):
        resp = api.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_reports_vector_store(self, api):
        data = api.get("/health").json()
        assert data["vector_store"] in ("chromadb", "pinecone")


class TestIngestAndQueryE2E:
    def test_ingest_starts_job(self, api):
        resp = api.post("/ingest", json={"tickers": ["AAPL"], "num_filings": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("started", "queued")
        assert "AAPL" in data["tickers"]

    def test_query_returns_answer_after_ingest(self, api):
        # Allow background ingestion to complete
        time.sleep(15)

        resp = api.post("/query", json={
            "question": "What was Apple's total revenue?",
            "provider": "groq",
            "top_k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["answer"], str) and len(data["answer"]) > 10
        assert "sources" in data
        assert "latency_ms" in data

    def test_query_sources_contain_aapl(self, api):
        resp = api.post("/query", json={
            "question": "Apple revenue 2024",
            "provider": "groq",
            "top_k": 5,
        })
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        tickers = [s["ticker"] for s in sources]
        assert "AAPL" in tickers, f"Expected AAPL in sources, got: {tickers}"

    def test_metadata_filter_ticker(self, api):
        resp = api.post("/query", json={
            "question": "annual revenue",
            "provider": "groq",
            "top_k": 5,
            "ticker": "AAPL",
        })
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        for source in sources:
            assert source["ticker"] == "AAPL"


class TestStatsE2E:
    def test_stats_shows_indexed_vectors(self, api):
        resp = api.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "vector_count" in data
        assert data["vector_count"] > 0


class TestRateLimitingE2E:
    def test_rate_limit_returns_429_after_threshold(self, api):
        """Rapid burst of 70 requests should hit the 60/min limit."""
        statuses = []
        for _ in range(70):
            resp = api.post("/query", json={
                "question": "test",
                "provider": "groq",
                "top_k": 1,
            })
            statuses.append(resp.status_code)
            if resp.status_code == 429:
                break
        assert 429 in statuses, "Rate limit was never triggered"
