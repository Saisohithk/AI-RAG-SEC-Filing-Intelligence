# src/ingestion — SEC EDGAR download, text chunking, and vector embedding pipeline
from src.ingestion.chunker import SECChunker
from src.ingestion.downloader import download_10k_filings, extract_text_from_filing
from src.ingestion.embedder import VectorStoreManager, get_embedder

__all__ = [
    "SECChunker",
    "download_10k_filings",
    "extract_text_from_filing",
    "VectorStoreManager",
    "get_embedder",
]
