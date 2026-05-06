"""
llm_router.py — Routes LLM calls to Ollama, Groq, or Gemini based on configuration.

What this file does:
  Returns the right LangChain LLM object based on the provider string.
  All providers implement the same BaseLanguageModel interface, so the
  rest of the code doesn't care which one is being used.

Why three LLM providers?
  - Ollama (llama3.2): Runs LOCALLY — no API key, no cost, no internet required.
    Use this for development and testing.

  - Groq (llama-3.3-70b): CLOUD INFERENCE at 500+ tokens/second. Free tier available.
    Use this when you need fast responses in demos.

  - Gemini (gemini-1.5-flash): Google's model. Best quality on complex financial questions.
    Use this for evaluations and production.

LangSmith tracing:
  All providers automatically trace their calls to LangSmith when the
  LANGCHAIN_TRACING_V2 and LANGSMITH_API_KEY env vars are set.
  This gives you a web dashboard showing every prompt, response, and latency.
"""

import logging

from langchain_core.language_models import BaseLanguageModel

from src.core.config import settings

logger = logging.getLogger(__name__)


def get_llm(provider: str = "ollama") -> BaseLanguageModel:
    """
    Get a configured LLM for the specified provider.

    Args:
        provider: One of "ollama", "groq", "gemini" (default: from settings)

    Returns:
        A LangChain BaseLanguageModel — works the same regardless of provider

    Raises:
        ValueError: If provider is not one of the supported options
        ValueError: If required API key is not configured for cloud providers

    Example:
        >>> llm = get_llm("groq")
        >>> response = llm.invoke("What is Apple's revenue?")
    """
    provider = provider.lower().strip()

    if provider == "ollama":
        # Ollama runs locally — no API key needed
        # Install at: https://ollama.ai, then run: ollama pull llama3.2
        try:
            from langchain_ollama import OllamaLLM as Ollama
        except ImportError:
            from langchain_community.llms import Ollama  # type: ignore[no-redef]

        logger.info(f"Using Ollama with model: {settings.OLLAMA_MODEL}")
        return Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    elif provider == "groq":
        # Groq: fastest cloud inference, free tier at console.groq.com
        from langchain_groq import ChatGroq
        from pydantic import SecretStr

        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com"
            )

        logger.info("Using Groq with model: llama-3.3-70b-versatile")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=SecretStr(settings.GROQ_API_KEY),
            temperature=0.1,  # Low temperature = more factual, less creative
        )

    elif provider == "gemini":
        # Gemini: Google's model, best quality, free tier at aistudio.google.com
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Get a free key at aistudio.google.com"
            )

        logger.info("Using Google Gemini with model: gemini-1.5-flash")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Choose from: 'ollama', 'groq', 'gemini'"
        )


def get_streaming_llm(provider: str = "ollama") -> BaseLanguageModel:
    """
    Get a configured LLM with streaming enabled.

    Streaming returns tokens one at a time as they're generated,
    instead of waiting for the full response. This makes the UI feel faster.

    Args:
        provider: Same options as get_llm()

    Returns:
        LLM configured with streaming=True
    """
    provider = provider.lower().strip()

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaLLM as Ollama
        except ImportError:
            from langchain_community.llms import Ollama  # type: ignore[no-redef]
        logger.info(f"Using streaming Ollama: {settings.OLLAMA_MODEL}")
        return Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        from pydantic import SecretStr

        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")
        logger.info("Using streaming Groq: llama-3.3-70b-versatile")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=SecretStr(settings.GROQ_API_KEY),
            temperature=0.1,
            streaming=True,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        logger.info("Using streaming Gemini: gemini-1.5-flash")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
            streaming=True,
        )

    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. Choose from: 'ollama', 'groq', 'gemini'"
        )
