"""
Tests for step1_embeddings.py

Unit tests  : none (the whole step is one Ollama call)
Integration : shape, dtype, and basic sanity of the returned vector
"""

import math
import pytest
from conftest import requires_ollama
from step1_embeddings import get_embedding


@requires_ollama
@pytest.mark.timeout(30)
def test_embedding_returns_list_of_floats():
    vec = get_embedding("hello world")
    assert isinstance(vec, list), "embedding should be a list"
    assert all(isinstance(v, float) for v in vec), "all elements should be float"


@requires_ollama
@pytest.mark.timeout(30)
def test_embedding_length_is_768():
    """nomic-embed-text produces 768-dimensional vectors."""
    vec = get_embedding("test sentence")
    assert len(vec) == 768, f"expected 768 dimensions, got {len(vec)}"


@requires_ollama
@pytest.mark.timeout(30)
def test_embedding_values_are_finite():
    vec = get_embedding("finite values only")
    bad = [v for v in vec if not math.isfinite(v)]
    assert not bad, f"found non-finite values: {bad[:5]}"


@requires_ollama
@pytest.mark.timeout(60)
def test_same_text_gives_same_embedding():
    """Embeddings are deterministic — same input → same output."""
    text = "the quick brown fox"
    vec1 = get_embedding(text)
    vec2 = get_embedding(text)
    assert vec1 == vec2, "same text should produce identical vectors"


@requires_ollama
@pytest.mark.timeout(60)
def test_different_texts_give_different_embeddings():
    vec1 = get_embedding("I love cats")
    vec2 = get_embedding("quantum physics equations")
    assert vec1 != vec2, "different texts should not produce identical vectors"
