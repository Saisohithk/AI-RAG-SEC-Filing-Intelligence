"""
tests/test_llm_router.py — Tests for the LLM router.

What we're testing:
  1. Each provider returns the correct LangChain class
  2. Invalid provider raises a clear ValueError
  3. Cloud providers require API keys (don't silently fail)
  4. Streaming versions also return correct classes

We mock the LLM classes to avoid real API calls in tests.

Run with: pytest tests/test_llm_router.py -v
"""

from unittest.mock import patch

import pytest

from src.generation.llm_router import get_llm, get_streaming_llm


# ============================================================
# Tests for get_llm()
# ============================================================


def test_ollama_provider_returns_correct_class():
    """
    get_llm("ollama") should return an Ollama instance.
    This uses local inference — no API key needed.
    """
    with patch("langchain_ollama.OllamaLLM") as mock_ollama:
        # Mock Ollama so we don't need a real Ollama server running
        mock_ollama.return_value = mock_ollama

        get_llm("ollama")

        # Verify Ollama was instantiated
        mock_ollama.assert_called_once()

        # Verify the model setting was passed
        call_kwargs = mock_ollama.call_args.kwargs
        assert "model" in call_kwargs or "base_url" in call_kwargs, (
            "Ollama should be initialized with model and base_url"
        )


def test_invalid_provider_raises_value_error():
    """
    Passing an unsupported provider name should raise ValueError immediately.
    This prevents silent failures where the wrong LLM is used.
    """
    with pytest.raises(ValueError) as exc_info:
        get_llm("gpt-4")  # OpenAI is not in our supported list

    # Error message should tell the user what's valid
    error_msg = str(exc_info.value)
    assert "gpt-4" in error_msg, "Error should mention the invalid provider name"
    assert any(valid in error_msg for valid in ["ollama", "groq", "gemini"]), (
        "Error should list the valid providers"
    )


def test_gemini_provider_requires_api_key(monkeypatch):
    """
    Requesting Gemini without GOOGLE_API_KEY set should raise ValueError.
    This fails fast instead of making an API call that will 401.
    """
    # Clear the API key in settings
    monkeypatch.setattr("src.generation.llm_router.settings.GOOGLE_API_KEY", "")

    with pytest.raises(ValueError) as exc_info:
        get_llm("gemini")

    error_msg = str(exc_info.value)
    assert "GOOGLE_API_KEY" in error_msg, (
        "Error should mention the missing env var so user knows how to fix it"
    )


def test_groq_provider_requires_api_key(monkeypatch):
    """
    Requesting Groq without GROQ_API_KEY set should raise ValueError.
    """
    monkeypatch.setattr("src.generation.llm_router.settings.GROQ_API_KEY", "")

    with pytest.raises(ValueError) as exc_info:
        get_llm("groq")

    error_msg = str(exc_info.value)
    assert "GROQ_API_KEY" in error_msg


def test_provider_names_are_case_insensitive():
    """
    "Ollama", "OLLAMA", "ollama" should all work.
    User input is often mixed case.
    """
    with patch("langchain_ollama.OllamaLLM") as mock_ollama:
        mock_ollama.return_value = mock_ollama

        get_llm("Ollama")  # Title case
        get_llm("OLLAMA")  # Upper case
        get_llm("ollama")  # Lower case

        # All three calls should succeed (3 calls total)
        assert mock_ollama.call_count == 3


# ============================================================
# Tests for get_streaming_llm()
# ============================================================


def test_streaming_ollama_is_initialized():
    """
    get_streaming_llm("ollama") should return an OllamaLLM instance.
    OllamaLLM supports streaming natively via stream()/astream() — no constructor kwarg needed.
    """
    with patch("langchain_ollama.OllamaLLM") as mock_ollama:
        mock_ollama.return_value = mock_ollama

        get_streaming_llm("ollama")

        call_kwargs = mock_ollama.call_args.kwargs
        assert "model" in call_kwargs or "base_url" in call_kwargs, (
            "Streaming Ollama should be initialized with model and base_url"
        )


def test_streaming_invalid_provider_raises_error():
    """Streaming version should also raise ValueError for unknown providers."""
    with pytest.raises(ValueError):
        get_streaming_llm("unknown_provider")
