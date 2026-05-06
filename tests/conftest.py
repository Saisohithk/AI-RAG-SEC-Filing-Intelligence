"""
tests/conftest.py — Shared pytest fixtures for all test suites.

Fixtures defined here are automatically available to every test file
in tests/unit/ and tests/integration/ without explicit imports.
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path so `src.*` imports resolve
# when running pytest from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Five realistic SEC filing chunks — shared across unit tests."""
    return [
        Document(
            page_content="Apple reported total net sales of $383 billion in fiscal 2024.",
            metadata={
                "chunk_id": "chunk_1",
                "ticker": "AAPL",
                "source": "AAPL-10-K-2024",
                "filing_date": "2024",
                "rerank_score": 0.95,
            },
        ),
        Document(
            page_content="Microsoft Azure cloud revenue grew 29% to reach $135 billion annually.",
            metadata={
                "chunk_id": "chunk_2",
                "ticker": "MSFT",
                "source": "MSFT-10-K-2024",
                "filing_date": "2024",
                "rerank_score": 0.82,
            },
        ),
        Document(
            page_content="Google advertising revenue from Search was $175 billion in 2024.",
            metadata={
                "chunk_id": "chunk_3",
                "ticker": "GOOGL",
                "source": "GOOGL-10-K-2024",
                "filing_date": "2024",
                "rerank_score": 0.71,
            },
        ),
        Document(
            page_content="Apple's gross margin was 45.9% driven by services growth.",
            metadata={
                "chunk_id": "chunk_4",
                "ticker": "AAPL",
                "source": "AAPL-10-K-2024",
                "filing_date": "2024",
                "rerank_score": 0.65,
            },
        ),
        Document(
            page_content="Risk factors include supply chain disruptions and competition from Samsung.",
            metadata={
                "chunk_id": "chunk_5",
                "ticker": "AAPL",
                "source": "AAPL-10-K-2024",
                "filing_date": "2024",
                "rerank_score": 0.45,
            },
        ),
    ]


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock embedder returning a fake 768-dim vector."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.0] * 768
    mock.embed_documents.return_value = [[0.0] * 768]
    return mock


@pytest.fixture
def mock_vector_store(sample_documents) -> MagicMock:
    """Mock VectorStoreManager — no API keys needed."""
    mock = MagicMock()
    mock.similarity_search.return_value = sample_documents[:3]
    return mock
