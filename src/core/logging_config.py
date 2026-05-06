"""
src/core/logging_config.py — Structured logging setup and cross-cutting decorators.

Two concerns live here:
  1. setup_logging() — call once at app startup to configure the root logger
     with a consistent format, level, and optional file handler.

  2. Decorators — applied to functions throughout the codebase to add
     observability without polluting business logic with try/except/timing blocks.

     @log_call     — logs entry, exit, and elapsed time
     @handle_errors — catches exceptions, logs them, and returns a fallback value
                      instead of propagating (useful at the UI boundary)

Why not a third-party library (structlog, loguru)?
  Standard logging is available everywhere with zero extra deps. The format
  below is JSON-compatible enough to be parsed by log aggregators (Datadog,
  CloudWatch) while remaining human-readable in a terminal.
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger once at application startup.

    Args:
        level: Log level string — "DEBUG", "INFO", "WARNING", "ERROR"

    Side effects:
        - Removes any handlers Streamlit may have attached before us
        - Writes to stderr (captured by uvicorn / Docker)
    """
    numeric = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric)

    # Prevent duplicate handlers on Streamlit hot-reload
    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)

    # Silence noisy third-party libraries
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("sec_rag").setLevel(numeric)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def log_call(logger: logging.Logger | None = None) -> Callable[[F], F]:
    """
    Decorator factory: log function entry, exit, and wall-clock time.

    Usage:
        logger = logging.getLogger(__name__)

        @log_call(logger)
        def my_function(x: int) -> int:
            ...

    Output:
        2024-01-01T12:00:00 | INFO     | mymodule | → my_function(args)
        2024-01-01T12:00:00 | INFO     | mymodule | ← my_function (42.3ms)
    """
    _logger = logger or logging.getLogger("sec_rag")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build a compact args preview (truncated at 80 chars)
            arg_preview = repr(args[1:] if args else ())[:80]
            _logger.debug("→ %s(%s)", func.__name__, arg_preview)
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                _logger.info("← %s (%.1fms)", func.__name__, elapsed)
                return result
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                _logger.error("✗ %s failed after %.1fms: %s", func.__name__, elapsed, exc)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def handle_errors(
    fallback: Any = None,
    log_level: str = "error",
    reraise: bool = False,
) -> Callable[[F], F]:
    """
    Decorator factory: catch exceptions at UI boundaries and return a fallback.

    Prevents a single API error from crashing the entire Streamlit session.
    Always logs the exception before swallowing it.

    Args:
        fallback: Value to return when an exception is caught
        log_level: "error" or "warning"
        reraise: If True, log and then re-raise (useful in non-UI contexts)

    Usage:
        @handle_errors(fallback={"status": "unknown"}, log_level="warning")
        def check_health(self) -> dict:
            ...
    """
    _log_method = log_level.lower()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logging.getLogger(
                getattr(args[0], "__class__", type(args[0])).__name__
                if args
                else "sec_rag"
            )
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                getattr(_logger, _log_method)(
                    "%s raised %s: %s", func.__name__, type(exc).__name__, exc
                )
                if reraise:
                    raise
                return fallback

        return wrapper  # type: ignore[return-value]

    return decorator
