"""
Tests for step3_chromadb.py

Unit tests  : ChromaDB add/query with a fake embedder (no Ollama)
Integration : full flow with real Ollama embeddings
"""

import uuid
import pytest
import chromadb
from conftest import requires_ollama
from step3_chromadb import OllamaEmbedder


# ── fake embedder for unit tests ──────────────────────────────────────────────

class FakeEmbedder:
    """
    Returns hand-crafted 4-dimensional vectors so we can predict
    exactly which results ChromaDB should return — no Ollama needed.
    """
    is_legacy = False   # required by newer ChromaDB versions

    _vectors = {
        "paris travel":  [1.0, 0.0, 0.0, 0.0],
        "eiffel tower":  [0.9, 0.1, 0.0, 0.0],
        "python code":   [0.0, 0.0, 1.0, 0.0],
        "football sport":[0.0, 0.0, 0.0, 1.0],
        "query paris":   [1.0, 0.0, 0.0, 0.0],  # identical to paris travel
        "query python":  [0.0, 0.0, 1.0, 0.0],
    }

    def _embed(self, text: str) -> list[float]:
        for key, vec in self._vectors.items():
            if key in text.lower():
                return vec
        return [0.25, 0.25, 0.25, 0.25]  # neutral fallback

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def name(self) -> str:
        return "fake-embedder"


def make_test_collection(embedder=None):
    client = chromadb.Client()
    ef = embedder or FakeEmbedder()
    # unique name per call so tests in the same process never collide
    name = f"test-{uuid.uuid4().hex[:8]}"
    return client.create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ── unit tests ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_collection_stores_correct_count():
    col = make_test_collection()
    col.add(
        documents=["doc one", "doc two", "doc three"],
        ids=["1", "2", "3"],
    )
    assert col.count() == 3


@pytest.mark.unit
def test_query_returns_n_results():
    col = make_test_collection()
    col.add(
        documents=["paris travel", "python code", "football sport"],
        ids=["1", "2", "3"],
    )
    results = col.query(query_texts=["query paris"], n_results=2)
    assert len(results["documents"][0]) == 2


@pytest.mark.unit
def test_query_returns_metadata():
    col = make_test_collection()
    col.add(
        documents=["paris travel"],
        ids=["1"],
        metadatas=[{"source": "travel.txt", "page": 1}],
    )
    results = col.query(query_texts=["query paris"], n_results=1)
    meta = results["metadatas"][0][0]
    assert meta["source"] == "travel.txt"
    assert meta["page"] == 1


@pytest.mark.unit
def test_most_similar_doc_is_first():
    """With our fake embedder, 'paris travel' should be nearest to 'query paris'."""
    col = make_test_collection()
    col.add(
        documents=["paris travel", "python code", "football sport"],
        ids=["1", "2", "3"],
    )
    results = col.query(query_texts=["query paris"], n_results=3)
    docs = results["documents"][0]
    assert docs[0] == "paris travel", f"Expected 'paris travel' first, got: {docs}"


@pytest.mark.unit
def test_distances_are_ordered_ascending():
    """ChromaDB returns distances (not similarities) in ascending order (closest first)."""
    col = make_test_collection()
    col.add(
        documents=["paris travel", "python code", "football sport"],
        ids=["1", "2", "3"],
    )
    results = col.query(query_texts=["query paris"], n_results=3)
    distances = results["distances"][0]
    assert distances == sorted(distances), f"Distances not sorted: {distances}"


@pytest.mark.unit
def test_add_duplicate_id_keeps_original():
    """
    ChromaDB silently ignores a duplicate add — no exception, no overwrite.
    The original document is preserved. Use upsert() to intentionally overwrite.
    """
    col = make_test_collection()
    col.add(documents=["first"], ids=["1"])
    col.add(documents=["second"], ids=["1"])   # silently ignored
    result = col.get(ids=["1"])
    assert result["documents"][0] == "first", (
        "Expected original document to be preserved on duplicate add"
    )


@pytest.mark.unit
def test_upsert_overwrites_existing():
    """upsert() is the correct way to update an existing document."""
    col = make_test_collection()
    col.add(documents=["first"], ids=["1"])
    col.upsert(documents=["second"], ids=["1"])
    result = col.get(ids=["1"])
    assert result["documents"][0] == "second"


# ── integration tests ─────────────────────────────────────────────────────────

@requires_ollama
@pytest.mark.timeout(60)
def test_ollama_embedder_shape():
    embedder = OllamaEmbedder()
    vecs = embedder(["hello world"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 768


@requires_ollama
@pytest.mark.timeout(120)
def test_real_query_finds_correct_document():
    """Paris question should retrieve the Paris document, not the Python one."""
    embedder = OllamaEmbedder()
    col = chromadb.Client().create_collection(
        name="integration_test",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    col.add(
        documents=[
            "The Eiffel Tower is located in Paris, France.",
            "Python was created by Guido van Rossum in 1991.",
        ],
        ids=["paris", "python"],
    )
    results = col.query(query_texts=["What is in Paris?"], n_results=1)
    assert results["ids"][0][0] == "paris", (
        f"Expected 'paris' doc, got: {results['ids'][0][0]}"
    )
