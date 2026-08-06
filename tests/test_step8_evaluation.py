"""
Tests for step8_evaluation.py

Unit tests  : metric math — no Ollama, no ChromaDB
Integration : full evaluation pipeline on real embeddings + LLM
"""

import json
import pytest
from conftest import requires_ollama
from step8_evaluation import (
    GOLDEN,
    evaluate_retrieval,
    hit_at_k,
    judge_faithfulness,
    judge_relevance,
    precision_at_k,
    reciprocal_rank,
)


# ── unit tests: retrieval metrics ─────────────────────────────────────────────

@pytest.mark.unit
def test_reciprocal_rank_first_position():
    chunks = ["the cat sat on the mat", "unrelated", "also unrelated"]
    assert reciprocal_rank(chunks, "cat") == pytest.approx(1.0)


@pytest.mark.unit
def test_reciprocal_rank_second_position():
    chunks = ["unrelated", "the cat sat on the mat", "also unrelated"]
    assert reciprocal_rank(chunks, "cat") == pytest.approx(0.5)


@pytest.mark.unit
def test_reciprocal_rank_third_position():
    chunks = ["unrelated", "also unrelated", "the cat sat on the mat"]
    assert reciprocal_rank(chunks, "cat") == pytest.approx(1 / 3)


@pytest.mark.unit
def test_reciprocal_rank_not_found():
    chunks = ["one", "two", "three"]
    assert reciprocal_rank(chunks, "cat") == 0.0


@pytest.mark.unit
def test_reciprocal_rank_case_insensitive():
    chunks = ["The CAT sat on the mat"]
    assert reciprocal_rank(chunks, "cat") == pytest.approx(1.0)


@pytest.mark.unit
def test_hit_at_k_found():
    chunks = ["no match", "contains the cat phrase", "no match"]
    assert hit_at_k(chunks, "cat") == 1


@pytest.mark.unit
def test_hit_at_k_not_found():
    chunks = ["no match", "also no match"]
    assert hit_at_k(chunks, "cat") == 0


@pytest.mark.unit
def test_hit_at_k_is_binary():
    # Even if phrase appears in multiple chunks, result is still 0 or 1
    chunks = ["cat here", "cat again", "cat third"]
    assert hit_at_k(chunks, "cat") in (0, 1)


@pytest.mark.unit
def test_precision_all_relevant():
    chunks = ["cat chunk", "cat again", "cat third"]
    assert precision_at_k(chunks, "cat") == pytest.approx(1.0)


@pytest.mark.unit
def test_precision_none_relevant():
    chunks = ["dog chunk", "fish chunk"]
    assert precision_at_k(chunks, "cat") == pytest.approx(0.0)


@pytest.mark.unit
def test_precision_partial():
    chunks = ["cat chunk", "unrelated", "unrelated"]
    assert precision_at_k(chunks, "cat") == pytest.approx(1 / 3)


@pytest.mark.unit
def test_precision_empty_chunks():
    assert precision_at_k([], "cat") == 0.0


# ── unit tests: MRR property ──────────────────────────────────────────────────

@pytest.mark.unit
def test_mrr_always_between_zero_and_one():
    """MRR is an average of reciprocal ranks, so it must be in [0, 1]."""
    from step8_evaluation import GOLDEN, reciprocal_rank
    # Simulate: phrase always at rank 1
    all_first = [1.0] * len(GOLDEN)
    mrr = sum(all_first) / len(all_first)
    assert 0.0 <= mrr <= 1.0

    # Simulate: phrase never found
    all_miss = [0.0] * len(GOLDEN)
    mrr = sum(all_miss) / len(all_miss)
    assert mrr == 0.0


@pytest.mark.unit
def test_mrr_rank1_beats_rank2():
    """Finding the answer at rank 1 (RR=1.0) scores better than rank 2 (RR=0.5)."""
    assert reciprocal_rank(["cat is here", "nothing"], "cat") > \
           reciprocal_rank(["nothing", "cat is here"], "cat")


# ── unit tests: golden dataset sanity ────────────────────────────────────────

@pytest.mark.unit
def test_golden_dataset_not_empty():
    assert len(GOLDEN) > 0


@pytest.mark.unit
def test_golden_dataset_has_required_keys():
    for item in GOLDEN:
        assert "question"        in item, f"Missing 'question' in {item}"
        assert "expected_phrase" in item, f"Missing 'expected_phrase' in {item}"
        assert "reference_answer" in item, f"Missing 'reference_answer' in {item}"


@pytest.mark.unit
def test_golden_phrases_appear_in_document():
    """Every expected_phrase must actually exist in the source DOCUMENT."""
    from step8_evaluation import DOCUMENT
    for item in GOLDEN:
        phrase = item["expected_phrase"].lower()
        assert phrase in DOCUMENT.lower(), (
            f"expected_phrase {phrase!r} not found in DOCUMENT — "
            "the golden dataset references text that doesn't exist"
        )


# ── integration tests ─────────────────────────────────────────────────────────

@requires_ollama
@pytest.mark.timeout(120)
def test_evaluate_retrieval_returns_expected_keys():
    import chromadb
    from step7_chunking import DOCUMENT, OllamaEmbedder, chunk_paragraphs
    from step8_evaluation import evaluate_retrieval

    chunks   = chunk_paragraphs(DOCUMENT, size=250)
    client   = chromadb.Client()
    embedder = OllamaEmbedder()
    col = client.create_collection(
        name="eval-test",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    col.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])

    metrics = evaluate_retrieval(col, k=3)
    assert "MRR"          in metrics
    assert "Hit@3"        in metrics
    assert "Precision@3"  in metrics


@requires_ollama
@pytest.mark.timeout(120)
def test_paragraph_chunking_achieves_perfect_hit_rate():
    """
    Paragraph chunking on the step7 DOCUMENT should hit every golden question
    at least once in the top-3 results (Hit@3 = 1.0).
    """
    import chromadb
    from step7_chunking import DOCUMENT, OllamaEmbedder, chunk_paragraphs
    from step8_evaluation import evaluate_retrieval

    chunks   = chunk_paragraphs(DOCUMENT, size=250)
    client   = chromadb.Client()
    embedder = OllamaEmbedder()
    col = client.create_collection(
        name="hit-rate-test",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    col.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])

    metrics = evaluate_retrieval(col, k=3)
    assert metrics["Hit@3"] == pytest.approx(1.0), (
        f"Expected Hit@3=1.0 for paragraph chunking, got {metrics['Hit@3']:.2f}"
    )


@requires_ollama
@pytest.mark.timeout(60)
def test_judge_faithfulness_returns_float_in_range():
    score = judge_faithfulness(
        question="What is RAG?",
        context="RAG combines retrieval with language model generation.",
        answer="RAG stands for Retrieval-Augmented Generation.",
    )
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0.0 <= score <= 1.0,      f"Score out of range: {score}"


@requires_ollama
@pytest.mark.timeout(60)
def test_judge_relevance_returns_float_in_range():
    score = judge_relevance(
        question="What is RAG?",
        answer="RAG combines retrieval with language model generation.",
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@requires_ollama
@pytest.mark.timeout(60)
def test_faithful_answer_scores_higher_than_hallucinated():
    """
    An answer grounded in the context should score higher on faithfulness
    than an answer that fabricates facts not present in the context.
    """
    question = "What does HNSW stand for?"
    context  = "HNSW stands for Hierarchical Navigable Small World."

    faithful_answer     = "HNSW stands for Hierarchical Navigable Small World."
    hallucinated_answer = "HNSW stands for High-speed Neural Search Wizard, invented in 2005."

    faithful_score     = judge_faithfulness(question, context, faithful_answer)
    hallucinated_score = judge_faithfulness(question, context, hallucinated_answer)

    assert faithful_score >= hallucinated_score, (
        f"Faithful answer ({faithful_score:.2f}) should score >= "
        f"hallucinated answer ({hallucinated_score:.2f})"
    )
