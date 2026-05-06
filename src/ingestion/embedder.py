"""
embedder.py — Converts text chunks into vector embeddings and stores them.

What this file does:
  1. LocalEmbedder: Uses sentence-transformers locally — NO API KEY required.
     Uses all-MiniLM-L6-v2 (384 dims, fast, good quality, ~90MB download).
     Best for: development, testing, demos without internet.

  2. JinaEmbedder: Calls Jina AI API — free tier, 1M tokens/month.
     Uses jina-embeddings-v2-base-en (768 dims, 8K token window, higher quality).
     Best for: production, longer documents.

  3. VectorStoreManager: Stores and retrieves vectors.
     Automatically picks the right embedder based on JINA_API_KEY availability.

Auto-selection logic:
  - JINA_API_KEY set in .env → use JinaEmbedder (production quality)
  - JINA_API_KEY not set    → use LocalEmbedder (no API key, works offline)
"""

import logging
import time
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.config import settings

logger = logging.getLogger(__name__)

# Jina AI API endpoint for embeddings
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v2-base-en"

# Max documents per API request (Jina's limit)
BATCH_SIZE = 20


class LocalEmbedder(Embeddings):
    """
    Free local embedder using sentence-transformers — no API key required.

    Uses all-MiniLM-L6-v2:
    - 384 dimensions (smaller than Jina's 768, but very fast)
    - Downloads ~90MB from HuggingFace on first use
    - Good quality for semantic search, great for development

    This is the fallback when JINA_API_KEY is not set.
    Swap to JinaEmbedder for production use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Args:
            model_name: SentenceTransformer model to use.
                        "all-MiniLM-L6-v2" is small (90MB) and fast.
                        "all-mpnet-base-v2" is larger (420MB) but higher quality.
        """
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading local embedder: {model_name} (downloads on first use)")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Local embedder ready: {model_name} ({self.dimension} dims)")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents. Runs on CPU — no GPU required."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.model.encode([text])[0].tolist()


def get_embedder() -> Embeddings:
    """
    Return the best available embedder based on configuration.

    Auto-selects:
    - JinaEmbedder if JINA_API_KEY is set in .env (production quality, 768 dims)
    - LocalEmbedder if not (free, no API key, 384 dims, works offline)
    """
    if settings.JINA_API_KEY:
        logger.info("Using JinaEmbedder (JINA_API_KEY found)")
        return JinaEmbedder()
    else:
        logger.info(
            "JINA_API_KEY not set — using LocalEmbedder (sentence-transformers, free)"
        )
        return LocalEmbedder()


class JinaEmbedder(Embeddings):
    """
    Calls Jina AI API to generate text embeddings.

    Implements LangChain's Embeddings interface so it works with
    any LangChain-compatible vector store.

    The key methods LangChain expects:
    - embed_documents(): Embed a list of texts (for indexing)
    - embed_query(): Embed a single query (for searching)
    """

    def __init__(self) -> None:
        if not settings.JINA_API_KEY:
            raise ValueError("JINA_API_KEY is not set. Get a free key at jina.ai")
        self.api_key = settings.JINA_API_KEY
        logger.info("JinaEmbedder initialized with model: " + JINA_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents (used during ingestion/indexing).

        Batches requests to stay within API limits.
        Retries on rate limiting (HTTP 429) with exponential backoff.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of 768 floats)
        """
        # Process in batches of 100 to respect API limits
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)
            logger.debug(
                f"Embedded batch {batch_start // BATCH_SIZE + 1}: {len(batch)} texts"
            )

            time.sleep(1.5)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single search query (used during retrieval).

        Args:
            text: The search query string

        Returns:
            Single embedding vector (768 floats)
        """
        # Single text still needs to be sent as a list to the API
        return self._embed_with_retry([text])[0]

    def _embed_with_retry(
        self, texts: list[str], max_retries: int = 3
    ) -> list[list[float]]:
        """
        Call Jina API with exponential backoff retry on rate limit errors.

        Rate limit (HTTP 429) means we're sending too many requests — wait and retry.
        Other errors are logged and re-raised immediately.
        """
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    JINA_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": JINA_MODEL, "input": texts},
                    timeout=30.0,  # 30 second timeout
                )

                if response.status_code == 429:
                    # Rate limited — wait longer each retry (1s, 2s, 4s)
                    wait_time = 2**attempt
                    logger.warning(
                        f"Rate limited by Jina API. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()  # Raise for other HTTP errors (4xx, 5xx)

                # Extract embeddings from response
                data = response.json()
                # Sort by index to ensure correct ordering
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]

            except httpx.TimeoutException:
                logger.error(f"Jina API timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    raise

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Jina API HTTP error: {e.response.status_code} - {e.response.text}"
                )
                raise

        raise RuntimeError(f"Jina API failed after {max_retries} retries")


class VectorStoreManager:
    """
    Manages storing and retrieving vector embeddings.

    Supports two backends:
    - ChromaDB (local): No API key, stores on disk, great for development
    - Pinecone (cloud): Production-grade, scalable, requires API key

    Usage:
        >>> vsm = VectorStoreManager(use_local=True)   # ChromaDB
        >>> vsm = VectorStoreManager(use_local=False)  # Pinecone
    """

    def __init__(self, use_local: bool = False) -> None:
        """
        Args:
            use_local: If True, use ChromaDB locally. If False, use Pinecone.
        """
        self.use_local = use_local
        # Auto-select embedder: Jina if API key set, else free local sentence-transformers
        self.embedder = get_embedder()

        # Detect embedding dimension from the chosen embedder
        # (Jina=768, LocalEmbedder/MiniLM=384)
        if isinstance(self.embedder, LocalEmbedder):
            self._embed_dim = self.embedder.dimension
        else:
            self._embed_dim = settings.PINECONE_DIMENSION

        if use_local:
            self._init_chroma()
        else:
            self._init_pinecone()

    def _init_chroma(self) -> None:
        """Set up ChromaDB — stores vectors locally in a folder."""
        import chromadb
        from langchain_chroma import Chroma

        collection_name = f"sec_filings_{self._embed_dim}"

        logger.info(f"Initializing ChromaDB at {settings.CHROMA_PERSIST_DIR}")
        logger.info(f"Using collection: {collection_name}")

        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

        self.store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self.embedder,
        )

        self.store_type = "chromadb"
        logger.info("ChromaDB initialized successfully")

    def _init_pinecone(self) -> None:
        """Set up Pinecone — cloud vector database."""
        from pinecone import Pinecone, ServerlessSpec

        if not settings.PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is not set. Get a free key at app.pinecone.io"
            )

        logger.info(f"Initializing Pinecone index: {settings.PINECONE_INDEX_NAME}")

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        # Create the index if it doesn't exist yet
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {settings.PINECONE_INDEX_NAME}")
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.PINECONE_DIMENSION,  # 768 for Jina embeddings
                metric="cosine",  # Cosine similarity for semantic search
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for index to be ready
            time.sleep(5)

        self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
        self.store_type = "pinecone"
        logger.info("Pinecone initialized successfully")

    def upsert_chunks(self, chunks: list[Document]) -> int:
        """
        Embed all chunks and store them in the vector database.

        Args:
            chunks: List of LangChain Documents to store

        Returns:
            Number of vectors successfully stored
        """
        logger.info(f"Upserting {len(chunks)} chunks to {self.store_type}...")

        if self.use_local:
            # ChromaDB: LangChain handles embedding + storage
            self.store.add_documents(chunks)
            stored = len(chunks)
        else:
            # Pinecone: Embed manually, then upsert
            stored = self._upsert_to_pinecone(chunks)

        logger.info(f"Successfully stored {stored} vectors")
        return stored

    def _upsert_to_pinecone(self, chunks: list[Document]) -> int:
        """
        Embed chunks and upsert to Pinecone in batches.
        Pinecone's upsert accepts (id, vector, metadata) tuples.
        """
        # Extract texts for embedding
        texts = [chunk.page_content for chunk in chunks]

        # Get all embeddings at once (JinaEmbedder handles batching internally)
        embeddings = self.embedder.embed_documents(texts)

        # Build Pinecone upsert format: list of (id, vector, metadata) dicts
        vectors: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(
                {
                    "id": chunk.metadata["chunk_id"],
                    "values": embedding,
                    "metadata": {
                        **chunk.metadata,
                        "text": chunk.page_content,  # Store text so we can retrieve it
                    },
                }
            )

        # Upsert in batches of 100 (Pinecone recommended batch size)
        PINECONE_BATCH_SIZE = 200
        for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
            batch = vectors[i : i + PINECONE_BATCH_SIZE]
            self.pinecone_index.upsert(vectors=batch)
            logger.debug(
                f"Upserted batch {i // PINECONE_BATCH_SIZE + 1}: {len(batch)} vectors"
            )
            time.sleep(1.5)

        return len(vectors)

    def similarity_search(
        self, query_embedding: list[float], top_k: int
    ) -> list[Document]:
        """
        Find the most semantically similar documents to a query vector.

        Args:
            query_embedding: The query embedded as a 768-dim vector
            top_k: Number of results to return

        Returns:
            List of Documents sorted by similarity (most similar first)
        """
        if self.use_local:
            # ChromaDB: pass embedding directly
            return self.store.similarity_search_by_vector(query_embedding, k=top_k)
        else:
            # Pinecone: query the index
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
            )
            # Convert Pinecone results to LangChain Documents
            documents = []
            for match in results["matches"]:
                meta = match["metadata"].copy()
                text = meta.pop(
                    "text", ""
                )  # Remove text from metadata, use as page_content
                documents.append(Document(page_content=text, metadata=meta))
            return documents

    def get_collection_stats(self) -> dict:
        """
        Get basic statistics about the vector store.

        Returns:
            Dict with index_name, vector_count, dimension, store_type
        """
        if self.use_local:
            try:
                count = self.store._collection.count()
            except Exception:
                count = 0
            return {
                "index_name": f"sec_filings_{self._embed_dim}",
                "vector_count": count,
                "dimension": self._embed_dim,
                "store_type": "chromadb",
            }
        else:
            stats = self.pinecone_index.describe_index_stats()
            return {
                "index_name": settings.PINECONE_INDEX_NAME,
                "vector_count": stats.get("total_vector_count", 0),
                "dimension": settings.PINECONE_DIMENSION,
                "store_type": "pinecone",
            }
