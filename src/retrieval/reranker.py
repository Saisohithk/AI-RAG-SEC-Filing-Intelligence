"""
reranker.py — Reranks retrieved documents using a BGE cross-encoder model.

What this file does:
  Takes the top 20 candidates from hybrid search and uses a more powerful
  (but slower) cross-encoder model to score them more accurately.
  Returns only the top 5.

Why two-stage retrieval?
  Stage 1 (hybrid search): Fast bi-encoder — embeds query and documents SEPARATELY
    → Scales to millions of documents
    → Less accurate (no direct query-document interaction)

  Stage 2 (reranker): Slow cross-encoder — processes query + document TOGETHER
    → Much more accurate (full attention over both texts)
    → Too slow to run on all documents — only applied to top 20 candidates

This two-stage approach is the industry standard pattern. The reranker is the
single biggest quality improvement in this pipeline (adds ~15% faithfulness).

Model: BAAI/bge-reranker-base
  - 278M parameters
  - Specifically trained for relevance ranking
  - Free to use, runs on CPU (no GPU required)
"""

import logging
import time

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# BGE reranker model from Hugging Face
RERANKER_MODEL = "BAAI/bge-reranker-base"


class BGEReranker:
    """
    Reranks document candidates using the BGE cross-encoder model.

    The cross-encoder processes each (query, document) pair together,
    giving much better relevance scores than vector similarity alone.

    Example:
        >>> reranker = BGEReranker()
        >>> top5 = reranker.rerank("What is Apple's revenue?", candidates)
    """

    def __init__(self) -> None:
        """Load the BGE reranker model. Downloads from HuggingFace on first use (~540MB)."""
        logger.info(f"Loading BGE reranker model: {RERANKER_MODEL}")
        start = time.time()

        # CrossEncoder processes (query, document) pairs and outputs relevance scores
        self.model = CrossEncoder(RERANKER_MODEL)

        elapsed = time.time() - start
        logger.info(f"BGE reranker loaded in {elapsed:.1f}s")

    def rerank(
        self, query: str, documents: list[Document], top_k: int = 5
    ) -> list[Document]:
        """
        Score and rerank documents by relevance to the query.

        Args:
            query: The user's question
            documents: Candidate documents from hybrid search (typically 20)
            top_k: How many top documents to keep (default: 5)

        Returns:
            Top-K documents sorted by reranker score, with rerank_score in metadata

        Example:
            >>> top5 = reranker.rerank("Apple revenue 2024", candidates, top_k=5)
            >>> print(top5[0].metadata["rerank_score"])
            0.94
        """
        if not documents:
            return []

        # Build input pairs: [(query, doc1_text), (query, doc2_text), ...]
        # The cross-encoder needs to see query + document together to score relevance
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs at once (more efficient than one at a time)
        scores = self.model.predict(pairs)

        # Attach scores to documents as metadata (useful for debugging and display)
        scored_docs: list[tuple[Document, float]] = []
        for doc, score in zip(documents, scores):
            # Create a copy with the rerank score added to metadata
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(score)},
            )
            scored_docs.append((enriched_doc, float(score)))

        # Sort by score descending — highest relevance first
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Log score distribution for debugging
        if scored_docs:
            top_score = scored_docs[0][1]
            bottom_score = scored_docs[-1][1]
            logger.info(
                f"Reranked {len(documents)} docs → top {top_k} "
                f"(scores: {top_score:.3f} to {bottom_score:.3f})"
            )

        return [doc for doc, _score in scored_docs[:top_k]]
