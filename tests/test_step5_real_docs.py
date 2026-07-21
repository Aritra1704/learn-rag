"""
Tests for step5_real_docs.py

Unit tests  : chunk_text logic (no Ollama, no files)
Integration : PDF extraction + full RAG pipeline on sample.pdf
"""

import pytest
from pathlib import Path
from conftest import requires_ollama
from step5_real_docs import chunk_text, extract_pages, pdf_to_chunks


SAMPLE_PDF = Path(__file__).parent.parent / "sample.pdf"


# ── unit tests: chunking ──────────────────────────────────────────────────────

@pytest.mark.unit
def test_short_text_is_single_chunk():
    text = "Hello world"
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world"


@pytest.mark.unit
def test_exact_size_text_is_single_chunk():
    # chunk_size=500, overlap=0 → exactly one chunk when text == chunk_size
    text = "A" * 500
    chunks = chunk_text(text, chunk_size=500, overlap=0)
    assert len(chunks) == 1


@pytest.mark.unit
def test_text_slightly_over_size_gives_two_chunks():
    text = "A" * 501
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 2


@pytest.mark.unit
def test_overlap_means_shared_content():
    """
    With chunk_size=10 and overlap=4, step = 10-4 = 6:
      text = "0123456789AB"  (12 chars)
      chunk 1: text[0:10]  = "0123456789"
      chunk 2: text[6:16]  = "6789AB"       <- last 4 of chunk1 == first 4 of chunk2
      chunk 3: text[12:22] = ""             <- empty, filtered out
    """
    text = "0123456789AB"   # 12 chars -> 2 non-empty chunks with step=6
    chunks = chunk_text(text, chunk_size=10, overlap=4)
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}: {chunks}"
    # The last 4 chars of chunk 1 should be the first 4 chars of chunk 2
    assert chunks[0][-4:] == chunks[1][:4], (
        f"Expected overlap: '{chunks[0][-4:]}' == '{chunks[1][:4]}'"
    )


@pytest.mark.unit
def test_empty_chunks_are_excluded():
    text = "A" * 10
    # chunk_size=5, overlap=4 → steps of 1 → lots of chunks, none empty
    chunks = chunk_text(text, chunk_size=5, overlap=4)
    assert all(c.strip() for c in chunks), "No empty or whitespace-only chunks expected"


@pytest.mark.unit
def test_chunk_content_covers_full_text():
    """
    Every character in the original text must appear in at least one chunk.
    Use a small overlap so this is checkable.
    """
    text = "ABCDEFGHIJ"
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    recovered = set()
    for chunk in chunks:
        for ch in chunk:
            recovered.add(ch)
    for ch in text:
        assert ch in recovered, f"Character '{ch}' missing from all chunks"


# ── unit tests: PDF extraction ────────────────────────────────────────────────

@pytest.mark.unit
def test_extract_pages_returns_list():
    if not SAMPLE_PDF.exists():
        pytest.skip("sample.pdf not found — run make_sample_pdf.py first")
    pages = extract_pages(str(SAMPLE_PDF))
    assert isinstance(pages, list)
    assert len(pages) > 0


@pytest.mark.unit
def test_extract_pages_returns_tuples_of_int_and_str():
    if not SAMPLE_PDF.exists():
        pytest.skip("sample.pdf not found")
    pages = extract_pages(str(SAMPLE_PDF))
    for page_num, text in pages:
        assert isinstance(page_num, int), f"page_num should be int, got {type(page_num)}"
        assert isinstance(text, str),     f"text should be str, got {type(text)}"
        assert len(text) > 0,             "page text should not be empty"


@pytest.mark.unit
def test_extract_pages_page_numbers_are_sequential():
    if not SAMPLE_PDF.exists():
        pytest.skip("sample.pdf not found")
    pages = extract_pages(str(SAMPLE_PDF))
    page_nums = [p for p, _ in pages]
    assert page_nums == sorted(page_nums), "pages should be in order"
    assert page_nums[0] == 1,             "first page should be 1"


@pytest.mark.unit
def test_pdf_to_chunks_metadata_has_source_and_page():
    if not SAMPLE_PDF.exists():
        pytest.skip("sample.pdf not found")
    chunks = pdf_to_chunks(str(SAMPLE_PDF))
    for text, meta in chunks:
        assert "source" in meta, "metadata missing 'source'"
        assert "page"   in meta, "metadata missing 'page'"
        assert meta["source"] == SAMPLE_PDF.name
        assert isinstance(meta["page"], int)


@pytest.mark.unit
def test_pdf_to_chunks_no_empty_chunks():
    if not SAMPLE_PDF.exists():
        pytest.skip("sample.pdf not found")
    chunks = pdf_to_chunks(str(SAMPLE_PDF))
    empty = [(t, m) for t, m in chunks if not t.strip()]
    assert not empty, f"Found {len(empty)} empty chunks"


# ── integration tests ─────────────────────────────────────────────────────────

@requires_ollama
@pytest.mark.timeout(120)
def test_end_to_end_rag_on_sample_pdf():
    """Index sample.pdf and verify a factual question returns a grounded answer."""
    if not SAMPLE_PDF.exists():
        pytest.skip("sample.pdf not found — run make_sample_pdf.py first")

    import chromadb
    from step5_real_docs import OllamaEmbedder, generate_answer, pdf_to_chunks

    client = chromadb.Client()
    embedder = OllamaEmbedder()
    col = client.create_collection(
        name="e2e_test",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = pdf_to_chunks(str(SAMPLE_PDF))
    texts  = [c for c, _ in chunks]
    metas  = [m for _, m in chunks]
    ids    = [str(i) for i in range(len(chunks))]
    col.add(documents=texts, ids=ids, metadatas=metas)

    results = col.query(query_texts=["What is RAG?"], n_results=2)
    retrieved_chunks = list(zip(results["documents"][0], results["metadatas"][0]))

    answer = generate_answer("What is RAG?", retrieved_chunks)

    assert isinstance(answer, str), "answer should be a string"
    assert len(answer) > 20,        "answer is suspiciously short"
    # The answer should mention retrieval or generation (it's about RAG after all)
    lower = answer.lower()
    assert any(kw in lower for kw in ["retrieval", "generation", "rag", "language model"]), (
        f"Answer doesn't seem to be about RAG: {answer[:200]}"
    )
