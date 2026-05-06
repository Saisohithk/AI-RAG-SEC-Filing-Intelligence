# src/generation — LLM provider routing and SEC-domain prompt templates
from src.generation.llm_router import get_llm, get_streaming_llm
from src.generation.prompt_templates import (
    CITATION_EXTRACTION_PROMPT,
    QUERY_REWRITE_PROMPT,
    SEC_RAG_PROMPT,
    format_context,
)

__all__ = [
    "get_llm",
    "get_streaming_llm",
    "SEC_RAG_PROMPT",
    "QUERY_REWRITE_PROMPT",
    "CITATION_EXTRACTION_PROMPT",
    "format_context",
]
