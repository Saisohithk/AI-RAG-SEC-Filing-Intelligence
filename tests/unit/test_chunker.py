"""
tests/test_chunker.py — Tests for the SECChunker class.

What we're testing:
  1. Chunks have all required metadata fields
  2. Chunk sizes stay within configured bounds
  3. Short/noise chunks are filtered out
  4. Batch processing works correctly

Run with: pytest tests/test_chunker.py -v
"""

import pytest

from src.ingestion.chunker import SECChunker


# ============================================================
# Test Fixtures — shared test data
# ============================================================

@pytest.fixture
def sample_metadata() -> dict:
    """Standard metadata for a fictional Apple 10-K filing."""
    return {
        "ticker": "AAPL",
        "filing_date": "2024",
        "form_type": "10-K",
        "source_path": "/data/AAPL/10-K/2024/filing.htm",
    }


@pytest.fixture
def sample_text() -> str:
    """
    Realistic SEC filing text excerpt (~1000 chars).
    Long enough to produce multiple chunks.
    """
    return """
    Apple Inc. designs, manufactures, and markets smartphones, personal computers,
    tablets, wearables, and accessories worldwide. The Company also sells various
    related services. Apple was incorporated in California in 1977.

    Net sales for fiscal 2024 were $383.3 billion, compared to $394.3 billion in
    fiscal 2023. Net income for fiscal 2024 was $93.7 billion, compared to $97.0
    billion in fiscal 2023. Diluted earnings per share were $6.08 for fiscal 2024.

    The Company's products and services include iPhone, Mac, iPad, Apple Watch,
    AirPods, Apple TV, HomePod, and various accessories. Services include the App
    Store, Apple Music, Apple TV+, iCloud, Apple Pay, and Apple Arcade.

    Risk Factors: Our future performance depends on our ability to develop and
    introduce new products and services that achieve widespread market acceptance.
    We face intense competition in all markets in which we participate. Competition
    includes vertically integrated competitors such as Samsung and Google.

    The Company's operations are subject to various laws and regulations in the
    jurisdictions in which it operates. We are subject to potential liability
    related to privacy and handling of personal data, intellectual property
    disputes, and product liability claims.
    """ * 3  # Repeat to make it long enough for multiple chunks


@pytest.fixture
def chunker() -> SECChunker:
    """SECChunker with small chunks to produce multiple chunks for testing."""
    return SECChunker(chunk_size=256, chunk_overlap=32)


# ============================================================
# Tests
# ============================================================

def test_chunk_produces_correct_metadata(chunker, sample_text, sample_metadata):
    """
    Every chunk must have all required metadata fields.
    Missing metadata = broken citations in the final answer.
    """
    chunks = chunker.chunk_document(sample_text, sample_metadata)

    assert len(chunks) > 0, "Should produce at least one chunk"

    for chunk in chunks:
        meta = chunk.metadata

        # Required fields for citations and retrieval
        assert "chunk_id" in meta, "Missing chunk_id (needed for deduplication)"
        assert "source" in meta, "Missing source (needed for citations)"
        assert "ticker" in meta, "Missing ticker"
        assert "filing_date" in meta, "Missing filing_date"
        assert "chunk_index" in meta, "Missing chunk_index"
        assert "total_chunks" in meta, "Missing total_chunks"
        assert "char_count" in meta, "Missing char_count"

        # Verify values match the input metadata
        assert meta["ticker"] == "AAPL"
        assert meta["filing_date"] == "2024"
        assert "AAPL" in meta["source"]


def test_chunk_size_within_bounds(chunker, sample_text, sample_metadata):
    """
    Chunks should not wildly exceed the configured chunk_size.
    (LangChain's splitter allows small overruns, so we test for 2x limit)
    """
    chunks = chunker.chunk_document(sample_text, sample_metadata)

    for chunk in chunks:
        char_count = len(chunk.page_content)
        # Allow 2x chunk_size because the splitter may not split perfectly
        assert char_count <= chunker.chunk_size * 2, (
            f"Chunk too large: {char_count} chars (limit: {chunker.chunk_size})"
        )
        assert char_count == chunk.metadata["char_count"], (
            "char_count metadata doesn't match actual content length"
        )


def test_short_chunks_filtered(chunker, sample_metadata):
    """
    Chunks shorter than 50 characters should be filtered out.
    These are usually page numbers, headers, or noise.
    """
    # Create text with some very short "chunks" that are just noise
    text = "A" * 300 + "\n\n" + "p." + "\n\n" + "B" * 300 + "\n\n" + "42" + "\n\n" + "C" * 300

    chunks = chunker.chunk_document(text, sample_metadata)

    # None of the chunks should be just "p." or "42" (very short)
    for chunk in chunks:
        assert len(chunk.page_content.strip()) >= 50, (
            f"Short chunk not filtered: '{chunk.page_content[:50]}'"
        )


def test_batch_chunk_multiple_docs(chunker, sample_text, sample_metadata):
    """
    chunk_batch() should process multiple documents and combine all chunks.
    """
    # Create two documents with different metadata
    meta_aapl = {**sample_metadata, "ticker": "AAPL"}
    meta_msft = {**sample_metadata, "ticker": "MSFT"}

    docs = [(sample_text, meta_aapl), (sample_text, meta_msft)]
    all_chunks = chunker.chunk_batch(docs)

    assert len(all_chunks) > 0, "Should produce chunks from batch"

    # Both tickers should be represented
    tickers_in_chunks = {chunk.metadata["ticker"] for chunk in all_chunks}
    assert "AAPL" in tickers_in_chunks
    assert "MSFT" in tickers_in_chunks

    # Total chunks should be roughly double a single document
    single_doc_chunks = chunker.chunk_document(sample_text, meta_aapl)
    assert len(all_chunks) >= len(single_doc_chunks), (
        "Batch should produce at least as many chunks as a single document"
    )


def test_chunk_indices_are_sequential(chunker, sample_text, sample_metadata):
    """
    Chunk indices should be sequential: 0, 1, 2, ..., total-1.
    This ensures we can reconstruct document order if needed.
    """
    chunks = chunker.chunk_document(sample_text, sample_metadata)
    total = chunks[0].metadata["total_chunks"]

    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i, (
            f"Expected chunk_index={i}, got {chunk.metadata['chunk_index']}"
        )
        assert chunk.metadata["total_chunks"] == total, (
            "total_chunks should be the same in all chunks of the same document"
        )
