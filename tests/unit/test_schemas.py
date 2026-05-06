"""
tests/unit/test_schemas.py — Tests for src/core/schemas.py Pydantic models.

What we're testing:
  1. Field validators clamp/normalise values correctly
  2. QueryRequest rejects invalid provider strings and out-of-range top_k
  3. IngestRequest normalises ticker case
  4. ChatMessage computed properties work correctly
  5. QueryResponse.from_api_dict() builds correctly from raw JSON

Run with: pytest tests/unit/test_schemas.py -v
"""
import pytest
from src.core.schemas import (
    ChatMessage,
    IngestRequest,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
    Source,
)


class TestSource:
    def test_clamps_rerank_score_above_one(self):
        src = Source(ticker="AAPL", filing_date="2024", chunk_text="x", rerank_score=1.8)
        assert src.rerank_score == 1.0

    def test_clamps_rerank_score_below_zero(self):
        src = Source(ticker="AAPL", filing_date="2024", chunk_text="x", rerank_score=-0.5)
        assert src.rerank_score == 0.0

    def test_handles_non_numeric_score(self):
        src = Source(ticker="AAPL", filing_date="2024", chunk_text="x", rerank_score="bad")
        assert src.rerank_score == 0.0

    def test_valid_score_unchanged(self):
        src = Source(ticker="AAPL", filing_date="2024", chunk_text="x", rerank_score=0.75)
        assert src.rerank_score == 0.75


class TestQueryRequest:
    def test_valid_request(self):
        req = QueryRequest(question="What is Apple's revenue?", provider="groq", top_k=5)
        assert req.question == "What is Apple's revenue?"
        assert req.provider == "groq"
        assert req.top_k == 5

    def test_invalid_provider_rejected(self):
        with pytest.raises(Exception):
            QueryRequest(question="test", provider="gpt-4", top_k=5)

    def test_empty_question_rejected(self):
        with pytest.raises(Exception):
            QueryRequest(question="", provider="groq", top_k=5)

    def test_top_k_range_enforced(self):
        with pytest.raises(Exception):
            QueryRequest(question="test", provider="groq", top_k=0)
        with pytest.raises(Exception):
            QueryRequest(question="test", provider="groq", top_k=11)


class TestIngestRequest:
    def test_tickers_normalised_to_uppercase(self):
        req = IngestRequest(tickers=["aapl", "msft"], num_filings=2)
        assert req.tickers == ["AAPL", "MSFT"]

    def test_whitespace_stripped_from_tickers(self):
        req = IngestRequest(tickers=["  AAPL  ", " MSFT"], num_filings=2)
        assert req.tickers == ["AAPL", "MSFT"]


class TestLatencyBreakdown:
    def test_stage_dict_returns_three_stages(self):
        lb = LatencyBreakdown(retrieve_ms=100.0, rerank_ms=200.0, generate_ms=300.0, total_ms=600.0)
        d = lb.stage_dict
        assert set(d.keys()) == {"Retrieve", "Rerank", "Generate"}
        assert d["Retrieve"] == 100.0

    def test_defaults_are_zero(self):
        lb = LatencyBreakdown()
        assert lb.total_ms == 0.0


class TestChatMessage:
    def test_has_sources_true_when_sources_present(self):
        src = Source(ticker="AAPL", filing_date="2024", chunk_text="x")
        msg = ChatMessage(role="assistant", content="answer", sources=[src])
        assert msg.has_sources is True

    def test_has_sources_false_when_empty(self):
        msg = ChatMessage(role="assistant", content="answer")
        assert msg.has_sources is False

    def test_has_latency_true_when_nonzero(self):
        lb = LatencyBreakdown(total_ms=500.0)
        msg = ChatMessage(role="assistant", content="answer", latency=lb)
        assert msg.has_latency is True

    def test_invalid_role_rejected(self):
        with pytest.raises(Exception):
            ChatMessage(role="system", content="test")


class TestQueryResponse:
    def test_from_api_dict_builds_correctly(self):
        raw = {
            "answer": "Apple revenue was $383B",
            "sources": [{"ticker": "AAPL", "filing_date": "2024", "chunk_text": "test", "rerank_score": 0.9}],
            "latency_ms": {"retrieve_ms": 100, "rerank_ms": 50, "generate_ms": 300, "total_ms": 450},
            "provider": "groq",
        }
        resp = QueryResponse.from_api_dict(raw)
        assert resp.answer == "Apple revenue was $383B"
        assert len(resp.sources) == 1
        assert resp.sources[0].ticker == "AAPL"
        assert resp.latency_ms.total_ms == 450.0
        assert resp.provider == "groq"

    def test_from_api_dict_tolerates_missing_fields(self):
        resp = QueryResponse.from_api_dict({})
        assert resp.answer == ""
        assert resp.sources == []
