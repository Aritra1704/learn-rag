"""
Tests for step6_persistent.py

Unit tests  : manifest logic, fingerprinting, skip detection (no Ollama)
Integration : full persistent multi-doc index + cross-doc retrieval
"""

import shutil
import tempfile
from pathlib import Path

import chromadb
import pytest
from conftest import requires_ollama
from step6_persistent import (
    OllamaEmbedder,
    _file_fingerprint,
    chunk_text,
    is_indexed,
    mark_indexed,
    pdf_to_chunks,
)

DOCS_DIR   = Path(__file__).parent.parent / "docs"
SAMPLE_PDF = DOCS_DIR / "ai_concepts.pdf"


# ── helpers ───────────────────────────────────────────────────────────────────

def make_manifest():
    """In-memory manifest collection for unit tests."""
    client = chromadb.Client()
    return client.get_or_create_collection(name="index-manifest")


# ── unit tests: fingerprint ───────────────────────────────────────────────────

@pytest.mark.unit
def test_fingerprint_includes_filename():
    if not SAMPLE_PDF.exists():
        pytest.skip("docs/ not found — run make_sample_pdf.py first")
    fp = _file_fingerprint(str(SAMPLE_PDF))
    assert "ai_concepts.pdf" in fp


@pytest.mark.unit
def test_fingerprint_includes_filesize():
    if not SAMPLE_PDF.exists():
        pytest.skip("docs/ not found")
    fp = _file_fingerprint(str(SAMPLE_PDF))
    size = SAMPLE_PDF.stat().st_size
    assert str(size) in fp, f"Expected file size {size} in fingerprint '{fp}'"


@pytest.mark.unit
def test_fingerprint_is_consistent():
    if not SAMPLE_PDF.exists():
        pytest.skip("docs/ not found")
    fp1 = _file_fingerprint(str(SAMPLE_PDF))
    fp2 = _file_fingerprint(str(SAMPLE_PDF))
    assert fp1 == fp2


# ── unit tests: manifest (is_indexed / mark_indexed) ─────────────────────────

@pytest.mark.unit
def test_new_file_is_not_indexed():
    if not SAMPLE_PDF.exists():
        pytest.skip("docs/ not found")
    manifest = make_manifest()
    assert not is_indexed(manifest, str(SAMPLE_PDF))


@pytest.mark.unit
def test_mark_then_is_indexed_returns_true():
    if not SAMPLE_PDF.exists():
        pytest.skip("docs/ not found")
    manifest = make_manifest()
    mark_indexed(manifest, str(SAMPLE_PDF))
    assert is_indexed(manifest, str(SAMPLE_PDF))


@pytest.mark.unit
def test_marking_twice_does_not_raise():
    """mark_indexed uses upsert so calling it twice should be idempotent."""
    if not SAMPLE_PDF.exists():
        pytest.skip("docs/ not found")
    manifest = make_manifest()
    mark_indexed(manifest, str(SAMPLE_PDF))
    mark_indexed(manifest, str(SAMPLE_PDF))  # should not raise
    assert is_indexed(manifest, str(SAMPLE_PDF))


@pytest.mark.unit
def test_different_files_tracked_independently():
    if len(list(DOCS_DIR.glob("*.pdf"))) < 2:
        pytest.skip("need at least 2 PDFs in docs/")
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    manifest = make_manifest()

    mark_indexed(manifest, str(pdfs[0]))
    assert     is_indexed(manifest, str(pdfs[0]))
    assert not is_indexed(manifest, str(pdfs[1]))


# ── unit tests: chunking (inherited from step6) ───────────────────────────────

@pytest.mark.unit
def test_chunk_overlap_shared_content():
    text = "0123456789AB"   # 12 chars, step=6 -> 2 non-empty chunks
    chunks = chunk_text(text, chunk_size=10, overlap=4)
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}: {chunks}"
    assert chunks[0][-4:] == chunks[1][:4]


@pytest.mark.unit
def test_pdf_to_chunks_source_matches_filename():
    if not SAMPLE_PDF.exists():
        pytest.skip("docs/ not found")
    chunks = pdf_to_chunks(str(SAMPLE_PDF))
    for _, meta in chunks:
        assert meta["source"] == "ai_concepts.pdf"


# ── integration tests ─────────────────────────────────────────────────────────

@requires_ollama
@pytest.mark.timeout(60)
def test_persistent_index_survives_new_client():
    """
    Write vectors with one PersistentClient, read them back with a new one.
    This verifies the index actually persisted to disk.
    """
    if not SAMPLE_PDF.exists():
        pytest.skip("docs/ not found")

    with tempfile.TemporaryDirectory() as tmp:
        embedder = OllamaEmbedder()

        # --- write ---
        client1 = chromadb.PersistentClient(path=tmp)
        col1 = client1.get_or_create_collection(
            name="documents",
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"},
        )
        chunks = pdf_to_chunks(str(SAMPLE_PDF))
        col1.add(
            documents=[c for c, _ in chunks],
            ids=[str(i) for i in range(len(chunks))],
            metadatas=[m for _, m in chunks],
        )
        written_count = col1.count()

        # --- read with a brand-new client pointing at the same path ---
        client2 = chromadb.PersistentClient(path=tmp)
        col2 = client2.get_or_create_collection(
            name="documents",
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"},
        )
        assert col2.count() == written_count, (
            f"Expected {written_count} vectors after reload, got {col2.count()}"
        )


@requires_ollama
@pytest.mark.timeout(180)
def test_cross_document_retrieval():
    """
    Index two documents on different topics.
    Each query should retrieve chunks from the correct document.
    """
    if not DOCS_DIR.exists() or len(list(DOCS_DIR.glob("*.pdf"))) < 2:
        pytest.skip("need docs/ with at least 2 PDFs")

    embedder = OllamaEmbedder()
    client   = chromadb.Client()
    col = client.create_collection(
        name="cross_doc_test",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    # Index exactly two known PDFs
    ai_pdf      = DOCS_DIR / "ai_concepts.pdf"
    python_pdf  = DOCS_DIR / "python_guide.pdf"

    for pdf in [ai_pdf, python_pdf]:
        chunks = pdf_to_chunks(str(pdf))
        col.add(
            documents=[c for c, _ in chunks],
            ids=[f"{pdf.stem}::{i}" for i in range(len(chunks))],
            metadatas=[m for _, m in chunks],
        )

    # AI question → should hit ai_concepts.pdf
    ai_results = col.query(query_texts=["What is a neural network?"], n_results=1)
    top_source  = ai_results["metadatas"][0][0]["source"]
    assert top_source == "ai_concepts.pdf", (
        f"Expected ai_concepts.pdf for AI question, got '{top_source}'"
    )

    # Python question → should hit python_guide.pdf
    py_results  = col.query(query_texts=["What are Python lists and tuples?"], n_results=1)
    top_source  = py_results["metadatas"][0][0]["source"]
    assert top_source == "python_guide.pdf", (
        f"Expected python_guide.pdf for Python question, got '{top_source}'"
    )
