"""
tests/unit/test_reranker.py — Unit tests for BGEReranker.

The CrossEncoder model is monkeypatched so no GPU/download is needed.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.reranker import BGEReranker


@pytest.fixture
def mock_reranker():
    """BGEReranker with CrossEncoder replaced by a deterministic mock."""
    with patch("src.retrieval.reranker.CrossEncoder") as MockCE:
        instance = MagicMock()
        MockCE.return_value = instance
        reranker = BGEReranker()
        yield reranker, instance


def _make_doc(text: str, ticker: str = "AAPL") -> Document:
    return Document(
        page_content=text, metadata={"ticker": ticker, "chunk_id": text[:16]}
    )


class TestBGERerankerEmpty:
    def test_empty_input_returns_empty_list(self, mock_reranker):
        reranker, _ = mock_reranker
        assert reranker.rerank("What is revenue?", [], top_k=5) == []


class TestBGERerankerOrdering:
    def test_documents_sorted_by_score_descending(self, mock_reranker):
        reranker, mock_ce = mock_reranker
        docs = [
            _make_doc("low relevance"),
            _make_doc("high relevance"),
            _make_doc("medium"),
        ]
        mock_ce.predict.return_value = [0.2, 0.9, 0.5]

        result = reranker.rerank("query", docs, top_k=3)

        assert result[0].page_content == "high relevance"
        assert result[1].page_content == "medium"
        assert result[2].page_content == "low relevance"

    def test_top_k_limits_results(self, mock_reranker):
        reranker, mock_ce = mock_reranker
        docs = [_make_doc(f"doc {i}") for i in range(10)]
        mock_ce.predict.return_value = list(range(10, 0, -1))

        result = reranker.rerank("query", docs, top_k=3)

        assert len(result) == 3

    def test_top_k_larger_than_input_returns_all(self, mock_reranker):
        reranker, mock_ce = mock_reranker
        docs = [_make_doc("a"), _make_doc("b")]
        mock_ce.predict.return_value = [0.8, 0.6]

        result = reranker.rerank("query", docs, top_k=10)

        assert len(result) == 2


class TestBGERerankerScoreAttachment:
    def test_rerank_score_attached_to_metadata(self, mock_reranker):
        reranker, mock_ce = mock_reranker
        docs = [_make_doc("Apple revenue was $383B")]
        mock_ce.predict.return_value = [0.92]

        result = reranker.rerank("Apple revenue?", docs, top_k=1)

        assert "rerank_score" in result[0].metadata
        assert abs(result[0].metadata["rerank_score"] - 0.92) < 1e-6

    def test_original_metadata_preserved(self, mock_reranker):
        reranker, mock_ce = mock_reranker
        docs = [
            Document(
                page_content="MSFT cloud revenue",
                metadata={"ticker": "MSFT", "filing_date": "2024", "chunk_id": "c1"},
            )
        ]
        mock_ce.predict.return_value = [0.75]

        result = reranker.rerank("cloud revenue", docs, top_k=1)

        assert result[0].metadata["ticker"] == "MSFT"
        assert result[0].metadata["filing_date"] == "2024"
        assert result[0].metadata["chunk_id"] == "c1"

    def test_raw_scores_above_one_stored_as_is(self, mock_reranker):
        """Reranker stores raw scores; clamping to [0,1] is Source's responsibility."""
        reranker, mock_ce = mock_reranker
        docs = [_make_doc("highly relevant text")]
        mock_ce.predict.return_value = [1.85]

        result = reranker.rerank("query", docs, top_k=1)

        assert result[0].metadata["rerank_score"] == pytest.approx(1.85)


class TestBGERerankerInputPairs:
    def test_predict_called_with_correct_pairs(self, mock_reranker):
        reranker, mock_ce = mock_reranker
        docs = [_make_doc("Apple sales"), _make_doc("Microsoft revenue")]
        mock_ce.predict.return_value = [0.6, 0.4]

        reranker.rerank("company revenue", docs, top_k=2)

        call_args = mock_ce.predict.call_args[0][0]
        assert call_args[0] == ("company revenue", "Apple sales")
        assert call_args[1] == ("company revenue", "Microsoft revenue")
