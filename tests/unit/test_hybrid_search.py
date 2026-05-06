"""
tests/test_hybrid_search.py — Tests for the HybridSearcher class.

What we're testing:
  1. BM25 returns docs with keywords from the query
  2. RRF correctly deduplicates docs that appear in both result lists
  3. RRF ranks docs appearing in both lists higher than docs in just one

We use a mock VectorStoreManager so these tests run without API keys.

Run with: pytest tests/test_hybrid_search.py -v
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.retrieval.hybrid_search import HybridSearcher


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def searcher(mock_vector_store, mock_embedder, sample_documents) -> HybridSearcher:
    """HybridSearcher with BM25 index pre-built on sample documents."""
    s = HybridSearcher(mock_vector_store, mock_embedder)
    s.build_bm25_index(sample_documents)
    return s


# ============================================================
# Tests
# ============================================================

def test_bm25_returns_relevant_docs(searcher):
    """
    BM25 should return documents containing the query keywords.
    "Apple revenue" should rank Apple revenue chunk higher than Azure chunk.
    """
    results = searcher.bm25_search("Apple revenue", top_k=3)

    assert len(results) == 3, "Should return exactly top_k results"

    # First result should contain "Apple" and/or "revenue"
    top_doc, top_score = results[0]
    assert top_score > 0, "Top result should have positive BM25 score"

    # The Apple revenue chunk should be in top 2
    top_2_ids = {doc.metadata["chunk_id"] for doc, _ in results[:2]}
    assert "chunk_1" in top_2_ids or "chunk_4" in top_2_ids, (
        "Apple revenue chunk should rank in top 2 for 'Apple revenue' query"
    )


def test_bm25_raises_without_index():
    """
    BM25 search should fail clearly if build_bm25_index() was never called.
    Better to fail fast than return wrong results silently.
    """
    mock_vs = MagicMock()
    mock_emb = MagicMock()
    s = HybridSearcher(mock_vs, mock_emb)
    # Don't call build_bm25_index

    with pytest.raises(RuntimeError, match="build_bm25_index"):
        s.bm25_search("Apple revenue", top_k=3)


def test_rrf_deduplicates_results(searcher, sample_documents):
    """
    When the same document appears in both BM25 and vector results,
    RRF should keep only ONE copy (not duplicates).
    """
    # Same doc in both lists
    chunk1 = sample_documents[0]
    chunk2 = sample_documents[1]

    bm25_results = [(chunk1, 2.5), (chunk2, 1.2)]
    vector_results = [(chunk1, 0.95), (chunk2, 0.72)]  # Same chunks

    fused = searcher.reciprocal_rank_fusion(bm25_results, vector_results)

    # Should have 2 unique docs, not 4
    chunk_ids = [doc.metadata["chunk_id"] for doc in fused]
    assert len(chunk_ids) == len(set(chunk_ids)), (
        "RRF should deduplicate — no document should appear twice"
    )
    assert len(fused) == 2, "Two unique docs in, two unique docs out"


def test_rrf_boosts_docs_in_both_lists(searcher, sample_documents):
    """
    A document appearing in BOTH lists should rank higher than
    a document appearing in only one list.
    This is the core value proposition of RRF.
    """
    chunk_in_both = sample_documents[0]    # chunk_id: "chunk_1"
    chunk_bm25_only = sample_documents[1]  # chunk_id: "chunk_2"
    chunk_vector_only = sample_documents[2]  # chunk_id: "chunk_3"

    # chunk_in_both appears in both lists (rank 1 in each)
    # chunk_bm25_only appears only in BM25 at rank 2
    # chunk_vector_only appears only in vector at rank 2
    bm25_results = [(chunk_in_both, 3.0), (chunk_bm25_only, 1.5)]
    vector_results = [(chunk_in_both, 0.98), (chunk_vector_only, 0.85)]

    fused = searcher.reciprocal_rank_fusion(bm25_results, vector_results)

    assert len(fused) > 0, "Should have results"

    # chunk_in_both should rank #1 because it appears in both lists
    top_chunk_id = fused[0].metadata["chunk_id"]
    assert top_chunk_id == "chunk_1", (
        f"Expected chunk_1 (in both lists) to rank #1, got {top_chunk_id}"
    )


def test_vector_search_calls_embedder(searcher, mock_embedder, mock_vector_store):
    """
    vector_search() should call the embedder to encode the query,
    then pass the result to the vector store.
    """
    results = searcher.vector_search("Apple revenue", top_k=3)

    # Embedder should have been called exactly once with the query
    mock_embedder.embed_query.assert_called_once_with("Apple revenue")

    # Vector store should have been called with the embedding
    mock_vector_store.similarity_search.assert_called_once()

    assert len(results) == 3, "Should return top_k results"
