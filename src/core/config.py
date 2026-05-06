"""
config.py — Central configuration for the SEC RAG system.

All settings are loaded from the .env file automatically.
Import `settings` anywhere in the project to access config values.

Why pydantic-settings?
  - Type-safe: wrong types raise errors at startup, not at runtime
  - Auto-loads .env file: no manual os.getenv() scattered everywhere
  - Self-documenting: all config lives in one place
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration for the SEC RAG system.
    Values are loaded from .env file (see .env.example for setup).
    """

    # --- API Keys ---
    JINA_API_KEY: str = ""               # Jina AI embeddings (jina.ai)
    PINECONE_API_KEY: str = ""           # Pinecone vector DB (app.pinecone.io)
    GROQ_API_KEY: str = ""               # Groq fast inference (console.groq.com)
    GOOGLE_API_KEY: str = ""             # Google Gemini (aistudio.google.com)
    LANGSMITH_API_KEY: str = ""          # LangSmith tracing (smith.langchain.com)

    # --- Security ---
    API_KEY: str = ""                    # If set, X-API-Key header is required on all endpoints
    ALLOWED_ORIGINS: str = "*"           # Comma-separated CORS origins; "*" allows all

    # --- Redis (optional — enables caching and job queue) ---
    REDIS_URL: str = ""                  # e.g. redis://localhost:6379/0

    # --- Pinecone Vector Store ---
    PINECONE_INDEX_NAME: str = "sec-rag"  # Name of the Pinecone index
    PINECONE_DIMENSION: int = 768         # Jina embeddings output 768 dimensions

    # --- LangSmith Observability ---
    LANGCHAIN_TRACING_V2: str = "true"    # Enable LangSmith tracing
    LANGCHAIN_PROJECT: str = "sec-rag"    # Project name in LangSmith dashboard

    # --- Local Ollama (runs on your machine, no API key) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # --- Text Chunking ---
    CHUNK_SIZE: int = 512      # Max characters per chunk
    CHUNK_OVERLAP: int = 64    # Overlap between chunks (preserves context at boundaries)

    # --- Retrieval ---
    TOP_K_RETRIEVAL: int = 20  # Fetch top 20 candidates from hybrid search
    TOP_K_RERANK: int = 5      # Keep top 5 after BGE reranker (quality filter)

    # --- Vector Store Selection ---
    USE_LOCAL_CHROMA: bool = False          # True = ChromaDB (local), False = Pinecone (cloud)
    CHROMA_PERSIST_DIR: str = "./chroma_db"  # Where ChromaDB saves data locally

    # --- Default LLM ---
    DEFAULT_LLM_PROVIDER: str = "ollama"  # Options: "ollama", "groq", "gemini"

    # --- BM25 Persistence ---
    BM25_INDEX_PATH: str = "./data/bm25_index.pkl"  # Serialized BM25 index location

    def __repr__(self) -> str:
        """Mask secret values to prevent accidental log leakage."""
        def _mask(v: str) -> str:
            return f"{v[:4]}***" if len(v) > 4 else "***"

        return (
            f"Settings("
            f"DEFAULT_LLM_PROVIDER={self.DEFAULT_LLM_PROVIDER!r}, "
            f"USE_LOCAL_CHROMA={self.USE_LOCAL_CHROMA}, "
            f"API_KEY={_mask(self.API_KEY) if self.API_KEY else 'not set'}, "
            f"GROQ_API_KEY={_mask(self.GROQ_API_KEY) if self.GROQ_API_KEY else 'not set'}, "
            f"GOOGLE_API_KEY={_mask(self.GOOGLE_API_KEY) if self.GOOGLE_API_KEY else 'not set'}"
            f")"
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton instance — import this everywhere
settings = Settings()
