"""
chunker.py — Splits large SEC filing text into smaller, searchable chunks.

What this file does:
  - Takes a full 10-K filing (often 100,000+ characters) and splits it into
    512-character chunks with 64-character overlap
  - Each chunk keeps metadata so we know WHERE it came from
  - Filters out tiny chunks (< 50 chars) that are just noise (page numbers, headers)

Why chunking matters:
  - LLMs have limited context windows — we can't feed them a whole 10-K
  - Smaller chunks = more precise retrieval (we find the exact relevant paragraph)
  - Overlap ensures we don't cut sentences at boundaries and lose meaning

Why RecursiveCharacterTextSplitter?
  - Tries to split on paragraph breaks first (\n\n), then sentences (. ),
    then words ( ), then characters — preserving as much semantic meaning as possible
"""

import logging
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SECChunker:
    """
    Splits SEC filing text into chunks suitable for embedding and retrieval.

    Each chunk is a LangChain Document with page_content (the text) and
    metadata (source info for citations).

    Example:
        >>> chunker = SECChunker(chunk_size=512, chunk_overlap=64)
        >>> chunks = chunker.chunk_document(text, {"ticker": "AAPL", ...})
        >>> print(len(chunks))
        342
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        """
        Args:
            chunk_size: Maximum characters per chunk (default: 512)
            chunk_overlap: Characters shared between adjacent chunks (default: 64)
                           Overlap preserves context at chunk boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter tries each separator in order
        # It only uses the next separator if the text is still too long
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],  # Prefer paragraph > sentence > word
            length_function=len,  # Count characters (not tokens)
        )

        logger.info(
            f"SECChunker ready: chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk_document(self, text: str, metadata: dict) -> list[Document]:
        """
        Split a single document into chunks, each with rich metadata.

        Args:
            text: Raw text from a SEC filing
            metadata: Dict with at minimum: ticker, filing_date, form_type

        Returns:
            List of LangChain Documents. Each has:
            - page_content: The actual text chunk
            - metadata: Source info (ticker, date, chunk position, etc.)

        Example:
            >>> chunks = chunker.chunk_document(text, {"ticker": "AAPL", "filing_date": "2024"})
            >>> chunks[0].metadata["ticker"]
            'AAPL'
        """
        # Split the text into raw string chunks
        raw_chunks = self.splitter.split_text(text)

        # Filter out chunks that are too short to be meaningful
        # (e.g., standalone page numbers, table headers like "Table of Contents")
        MIN_CHUNK_LENGTH = 50
        raw_chunks = [c for c in raw_chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]

        total = len(raw_chunks)

        # Convert each string chunk into a LangChain Document with metadata
        documents: list[Document] = []
        for i, chunk_text in enumerate(raw_chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    # Unique ID for deduplication in hybrid search
                    "chunk_id": str(uuid.uuid4()),
                    # Human-readable source for citations in answers
                    "source": f"{metadata.get('ticker', 'UNKNOWN')}-{metadata.get('form_type', '10-K')}-{metadata.get('filing_date', 'unknown')}",
                    # Individual fields for filtering
                    "ticker": metadata.get("ticker", "UNKNOWN"),
                    "filing_date": metadata.get("filing_date", "unknown"),
                    "form_type": metadata.get("form_type", "10-K"),
                    # Position info (useful for debugging and ordering)
                    "chunk_index": i,
                    "total_chunks": total,
                    "char_count": len(chunk_text),
                },
            )
            documents.append(doc)

        logger.info(
            f"Chunked {metadata.get('ticker', '?')} filing: "
            f"{len(text):,} chars -> {total} chunks"
        )
        return documents

    def chunk_batch(self, documents: list[tuple[str, dict]]) -> list[Document]:
        """
        Process multiple documents at once.

        Args:
            documents: List of (text, metadata) tuples

        Returns:
            All chunks from all documents combined into one flat list

        Example:
            >>> all_chunks = chunker.chunk_batch([(text1, meta1), (text2, meta2)])
        """
        all_chunks: list[Document] = []

        for i, (text, metadata) in enumerate(documents):
            logger.info(
                f"Processing document {i + 1}/{len(documents)}: {metadata.get('ticker', '?')}"
            )
            chunks = self.chunk_document(text, metadata)
            all_chunks.extend(chunks)

        logger.info(
            f"Batch complete: {len(documents)} docs → {len(all_chunks)} total chunks"
        )
        return all_chunks
