"""
hybrid_search.py — Combines keyword search (BM25) with semantic vector search.

What this file does:
  1. BM25 search: Finds chunks that share EXACT WORDS with the query
     Example: "revenue 2024" → finds chunks containing "revenue" and "2024"

  2. Vector search: Finds chunks that are SEMANTICALLY SIMILAR to the query
     Example: "total sales" → also finds chunks about "net revenue" (same meaning)

  3. RRF (Reciprocal Rank Fusion): Merges both result lists intelligently
     Documents that appear high in BOTH lists get the best score

Why hybrid search?
  - BM25 alone misses synonyms and paraphrases ("earnings" vs "profit")
  - Vector search alone misses exact matches for specific numbers/dates
  - Hybrid catches BOTH — consistently beats either method alone by 15-20% on RAG benchmarks

RRF Formula: score(doc) = Σ 1/(k + rank)
  where k=60 is a constant that smooths the impact of high-ranked documents
"""

import logging
import shutil
from pathlib import Path

import joblib
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from src.ingestion.embedder import VectorStoreManager

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Retrieves relevant SEC filing chunks using hybrid BM25 + vector search.

    Pipeline:
      1. Query → BM25 (keyword search) → top 20 results
      2. Query → Jina embed → vector search → top 20 results
      3. RRF fusion → deduplicated, re-ranked list → top K results

    Example:
        >>> searcher = HybridSearcher(vector_store, embedder)
        >>> searcher.build_bm25_index(all_chunks)
        >>> results = searcher.search("What was Apple's revenue?", top_k=10)
    """

    def __init__(self, vector_store: VectorStoreManager, embedder: Embeddings) -> None:
        """
        Args:
            vector_store: Initialized VectorStoreManager (Pinecone or ChromaDB)
            embedder: Any Embeddings-compatible embedder (Jina or Local)
        """
        self.vector_store = vector_store
        self.embedder = embedder

        # These are set when build_bm25_index() is called
        self.bm25: BM25Okapi | None = None
        self.bm25_corpus: list[Document] = []  # Original documents for BM25

        logger.info("HybridSearcher initialized")

    def save_index(self, path: Path) -> None:
        """Atomically serialize BM25 index to disk using joblib (compress=3)."""
        if self.bm25 is None:
            logger.warning("save_index() called with no index built — skipping")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        joblib.dump((self.bm25, self.bm25_corpus), tmp, compress=3)
        shutil.move(str(tmp), str(path))
        logger.info(f"BM25 index saved → {path} ({len(self.bm25_corpus)} docs)")

    def load_index(self, path: Path) -> bool:
        """Load a persisted BM25 index. Returns True on success."""
        try:
            self.bm25, self.bm25_corpus = joblib.load(path)
            logger.info(f"BM25 index loaded ← {path} ({len(self.bm25_corpus)} docs)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 index from {path}: {e}")
            return False

    def build_bm25_index(self, documents: list[Document]) -> None:
        """
        Build a BM25 index from a list of documents (must be called before searching).

        BM25 works by pre-computing term frequencies across all documents.
        This is fast at search time but requires loading documents into memory.

        Args:
            documents: All Document chunks to index for keyword search
        """
        logger.info(f"Building BM25 index for {len(documents)} documents...")

        # Tokenize each document: split into lowercase words
        # BM25 needs tokenized text to count word frequencies
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = documents  # Keep original docs to return with results

        logger.info("BM25 index built successfully")

    def bm25_search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """
        Find documents that contain words matching the query (keyword search).

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of (Document, bm25_score) pairs, sorted by score descending

        Raises:
            RuntimeError: If build_bm25_index() has not been called yet
        """
        if self.bm25 is None:
            raise RuntimeError("Call build_bm25_index() before searching")

        # Tokenize the query the same way we tokenized the corpus
        tokenized_query = query.lower().split()

        # BM25 scores every document against the query
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each document with its score and sort by score descending
        doc_score_pairs = list(zip(self.bm25_corpus, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return doc_score_pairs[:top_k]

    def vector_search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """
        Find documents semantically similar to the query (meaning-based search).

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (Document, similarity_score) pairs
        """
        # Convert query text to a 768-dim vector
        query_embedding = self.embedder.embed_query(query)

        # Find nearest neighbor vectors in the store
        results = self.vector_store.similarity_search(query_embedding, top_k=top_k)

        # Assign descending scores (1.0 for best, decreasing for rest)
        # Vector stores don't always return scores, so we synthesize rank-based ones
        scored_results = [
            (doc, 1.0 - (i * 0.01))  # 1.0, 0.99, 0.98, ...
            for i, doc in enumerate(results)
        ]

        return scored_results

    def reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[Document, float]],
        vector_results: list[tuple[Document, float]],
        k: int = 60,
    ) -> list[Document]:
        """
        Merge BM25 and vector search results using Reciprocal Rank Fusion.

        RRF formula: score(doc) = Σ 1/(k + rank_in_list)

        The constant k=60 prevents the very top results from dominating.
        Documents appearing in BOTH lists get summed scores — a big boost.

        Args:
            bm25_results: Results from BM25 keyword search
            vector_results: Results from vector similarity search
            k: RRF smoothing constant (default: 60, per the original paper)

        Returns:
            Deduplicated documents sorted by RRF score (best first)
        """
        # Map chunk_id → (document, accumulated_rrf_score)
        rrf_scores: dict[str, tuple[Document, float]] = {}

        # Add scores from BM25 results
        for rank, (doc, _score) in enumerate(bm25_results):
            chunk_id = doc.metadata.get("chunk_id", str(rank))
            rrf_contribution = 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed

            if chunk_id in rrf_scores:
                # Document already seen from vector results — add to its score
                existing_doc, existing_score = rrf_scores[chunk_id]
                rrf_scores[chunk_id] = (existing_doc, existing_score + rrf_contribution)
            else:
                rrf_scores[chunk_id] = (doc, rrf_contribution)

        # Add scores from vector results
        for rank, (doc, _score) in enumerate(vector_results):
            chunk_id = doc.metadata.get("chunk_id", str(rank))
            rrf_contribution = 1.0 / (k + rank + 1)

            if chunk_id in rrf_scores:
                existing_doc, existing_score = rrf_scores[chunk_id]
                rrf_scores[chunk_id] = (existing_doc, existing_score + rrf_contribution)
            else:
                rrf_scores[chunk_id] = (doc, rrf_contribution)

        # Sort by RRF score descending and return just the documents
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)

        logger.debug(
            f"RRF fusion: {len(bm25_results)} BM25 + {len(vector_results)} vector "
            f"→ {len(sorted_docs)} unique docs"
        )

        return [doc for doc, _score in sorted_docs]

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, str] | None = None,
    ) -> list[Document]:
        """
        Run the full hybrid search pipeline.

        Args:
            query: Natural language question about SEC filings
            top_k: Number of final results to return (default: 20)
            filters: Optional metadata equality filters, e.g. {"ticker": "AAPL", "filing_date": "2024"}

        Returns:
            Top-K most relevant Document chunks, optionally filtered by metadata
        """
        logger.info(f"Hybrid search for: '{query}' (top_k={top_k}, filters={filters})")

        bm25_results = self.bm25_search(query, top_k=top_k) if self.bm25 else []
        vector_results = self.vector_search(query, top_k=top_k)
        fused = self.reciprocal_rank_fusion(bm25_results, vector_results)

        if filters:
            fused = [
                doc for doc in fused
                if all(str(doc.metadata.get(k, "")) == str(v) for k, v in filters.items())
            ]

        return fused[:top_k]
