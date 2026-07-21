"""
Tests for step2_similarity.py

Unit tests  : cosine_similarity math (no Ollama needed — pure Python)
Integration : similar sentences score higher than unrelated ones
"""

import math
import pytest
from conftest import requires_ollama
from step2_similarity import cosine_similarity, get_embedding


# ── unit tests (no Ollama) ────────────────────────────────────────────────────

@pytest.mark.unit
def test_identical_vectors_score_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_opposite_vectors_score_minus_one():
    v = [1.0, 0.0]
    w = [-1.0, 0.0]
    assert cosine_similarity(v, w) == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.unit
def test_orthogonal_vectors_score_zero():
    """Perpendicular vectors have zero similarity — completely unrelated."""
    v = [1.0, 0.0]
    w = [0.0, 1.0]
    assert cosine_similarity(v, w) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_zero_vector_returns_zero():
    """A zero vector has no direction — similarity is undefined, return 0."""
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0


@pytest.mark.unit
def test_similarity_is_symmetric():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


@pytest.mark.unit
def test_score_range():
    """Cosine similarity is always in [-1, 1]."""
    import random
    random.seed(42)
    for _ in range(20):
        a = [random.uniform(-1, 1) for _ in range(8)]
        b = [random.uniform(-1, 1) for _ in range(8)]
        score = cosine_similarity(a, b)
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6, f"score out of range: {score}"


# ── integration tests (need Ollama) ──────────────────────────────────────────

@requires_ollama
@pytest.mark.timeout(60)
def test_similar_sentences_score_higher_than_unrelated():
    """
    'The cat sat on the mat' and 'A feline rested on the rug' should be
    more similar to each other than to 'The stock market crashed today'.
    """
    cat1 = get_embedding("The cat sat on the mat.")
    cat2 = get_embedding("A feline rested on the rug.")
    stock = get_embedding("The stock market crashed today.")

    score_related   = cosine_similarity(cat1, cat2)
    score_unrelated = cosine_similarity(cat1, stock)

    assert score_related > score_unrelated, (
        f"Expected similar sentences ({score_related:.3f}) to outscore "
        f"unrelated ones ({score_unrelated:.3f})"
    )


@requires_ollama
@pytest.mark.timeout(60)
def test_query_finds_best_match():
    """The highest-scoring sentence for a query should be the most relevant one."""
    query = get_embedding("What did the cat do?")

    candidates = {
        "cat":   get_embedding("The cat sat on the mat."),
        "stock": get_embedding("The stock market crashed today."),
        "pizza": get_embedding("I love eating pizza on Fridays."),
    }

    scores = {name: cosine_similarity(query, vec) for name, vec in candidates.items()}
    best = max(scores, key=scores.get)

    assert best == "cat", (
        f"Expected 'cat' to be the best match, got '{best}'. Scores: {scores}"
    )
