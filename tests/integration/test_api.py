"""
tests/integration/test_api.py — Integration tests for the FastAPI backend.

Starts the real FastAPI app with mocked ML components (no GPU/API keys needed).
Verifies HTTP contracts: status codes, response shape, error handling, auth.

Run with: pytest tests/integration/test_api.py -v
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document



@pytest.fixture
def mock_components():
    """Mock all heavy ML components so tests run without GPU/API keys."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 768

    vector_store = MagicMock()
    vector_store.get_collection_stats.return_value = {
        "index_name": "test", "vector_count": 100,
        "dimension": 768, "store_type": "chromadb",
    }

    searcher = MagicMock()
    searcher.search.return_value = [
        Document(
            page_content="Apple revenue was $383 billion",
            metadata={"ticker": "AAPL", "filing_date": "2024", "chunk_id": "c1"},
        )
    ]

    reranker = MagicMock()
    reranker.rerank.return_value = [
        Document(
            page_content="Apple revenue was $383 billion",
            metadata={"ticker": "AAPL", "filing_date": "2024",
                      "chunk_id": "c1", "rerank_score": 0.92},
        )
    ]
    return embedder, vector_store, searcher, reranker


@pytest.fixture
def client(mock_components):
    """FastAPI TestClient with mocked ML components and auth disabled."""
    embedder, vector_store, searcher, reranker = mock_components

    with patch("src.api.app.embedder", embedder), \
         patch("src.api.app.vector_store", vector_store), \
         patch("src.api.app.searcher", searcher), \
         patch("src.api.app.reranker", reranker), \
         patch("src.api.app.settings.API_KEY", ""):   # disable auth for most tests
        from src.api.app import app
        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_available" in data
        assert "redis_connected" in data

    def test_health_is_public_no_auth_needed(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_stats_returns_vector_count(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert "vector_count" in resp.json()


class TestQueryEndpoint:
    def test_query_returns_expected_shape(self, client, mock_components):
        _, _, _, _ = mock_components
        mock_llm = MagicMock()
        mock_llm_resp = MagicMock()
        mock_llm_resp.content = "Apple revenue was $383B [AAPL - 10-K - 2024]"
        mock_llm.invoke.return_value = mock_llm_resp

        with patch("src.api.app.get_llm", return_value=mock_llm), \
             patch("src.api.app._get_query_variants", return_value=["What was Apple revenue?"]):
            resp = client.post("/query", json={
                "question": "What was Apple revenue?",
                "provider": "groq",
                "top_k": 5,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data
        assert "provider" in data

    def test_query_with_ticker_filter(self, client, mock_components):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Filtered answer")

        with patch("src.api.app.get_llm", return_value=mock_llm), \
             patch("src.api.app._get_query_variants", return_value=["revenue"]):
            resp = client.post("/query", json={
                "question": "revenue",
                "provider": "groq",
                "top_k": 3,
                "ticker": "AAPL",
                "year": 2024,
            })
        assert resp.status_code == 200

    def test_query_503_when_components_not_initialized(self):
        with patch("src.api.app.embedder", None), \
             patch("src.api.app.vector_store", None), \
             patch("src.api.app.searcher", None), \
             patch("src.api.app.reranker", None), \
             patch("src.api.app.settings.API_KEY", ""):
            from src.api.app import app
            tc = TestClient(app, raise_server_exceptions=False)
            resp = tc.post("/query", json={"question": "test", "provider": "groq", "top_k": 5})
        assert resp.status_code == 503

    def test_query_invalid_provider_rejected_by_schema(self, client):
        resp = client.post("/query", json={
            "question": "test", "provider": "openai", "top_k": 5
        })
        assert resp.status_code == 422


class TestAuthEndpoint:
    def test_401_with_wrong_api_key(self, mock_components):
        embedder, vector_store, searcher, reranker = mock_components
        with patch("src.api.app.embedder", embedder), \
             patch("src.api.app.vector_store", vector_store), \
             patch("src.api.app.searcher", searcher), \
             patch("src.api.app.reranker", reranker), \
             patch("src.api.app.settings.API_KEY", "secret-key"):
            from src.api.app import app
            tc = TestClient(app, raise_server_exceptions=False)
            resp = tc.post(
                "/query",
                json={"question": "test", "provider": "groq", "top_k": 5},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401

    def test_200_with_correct_api_key(self, mock_components):
        embedder, vector_store, searcher, reranker = mock_components
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")

        with patch("src.api.app.embedder", embedder), \
             patch("src.api.app.vector_store", vector_store), \
             patch("src.api.app.searcher", searcher), \
             patch("src.api.app.reranker", reranker), \
             patch("src.api.app.settings.API_KEY", "secret-key"), \
             patch("src.api.app.get_llm", return_value=mock_llm), \
             patch("src.api.app._get_query_variants", return_value=["test"]):
            from src.api.app import app
            tc = TestClient(app)
            resp = tc.post(
                "/query",
                json={"question": "test", "provider": "groq", "top_k": 5},
                headers={"X-API-Key": "secret-key"},
            )
        assert resp.status_code == 200


class TestIngestEndpoint:
    def test_ingest_starts_background_job(self, client):
        resp = client.post("/ingest", json={"tickers": ["AAPL"], "num_filings": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] in ("started", "queued")
        assert "AAPL" in data["tickers"]

    def test_ingest_normalises_ticker_case(self, client):
        resp = client.post("/ingest", json={"tickers": ["aapl", "msft"], "num_filings": 1})
        assert resp.status_code == 200
        assert "AAPL" in resp.json()["tickers"]
        assert "MSFT" in resp.json()["tickers"]
