# src/api/main.py — compatibility shim
# The FastAPI application now lives in src/api/app.py
# This file exists only so that any tooling that references the old path still works.
from src.api.app import app  # noqa: F401
