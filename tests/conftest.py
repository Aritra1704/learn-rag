"""
Shared fixtures and helpers for all test files.

Two test categories:
  unit        — pure Python, no network, no Ollama. Always fast.
  integration — requires Ollama running at localhost:11434.

Integration tests are skipped automatically when Ollama is unreachable.
Run only unit tests:        pytest -m unit
Run only integration tests: pytest -m integration
Run everything:             pytest
"""

import json
import math
import sys
import urllib.request
from pathlib import Path

import pytest

# Make the learn-rag root importable so tests can do `from step2_similarity import ...`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OLLAMA_URL = "http://localhost:11434"


def ollama_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3):
            return True
    except Exception:
        return False


requires_ollama = pytest.mark.usefixtures()  # placeholder replaced below


def _make_requires_ollama():
    skip = pytest.mark.skipif(
        not ollama_is_up(),
        reason="Ollama not reachable at localhost:11434 — skipping integration test",
    )
    integration = pytest.mark.integration
    # Stack both marks so `-m integration` selects these tests
    # and they are also skipped when Ollama is down
    def decorator(fn):
        return integration(skip(fn))
    return decorator


requires_ollama = _make_requires_ollama()
