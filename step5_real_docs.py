"""
STEP 5: RAG on a real PDF.

New problems we didn't face with hardcoded strings:
  - A PDF page can have thousands of words — too big to embed as one chunk.
    Embedding models have a token limit, and one giant chunk gives poor retrieval
    because it mixes many topics.
  - We need to split text into overlapping chunks so context isn't lost at
    the boundary between two chunks.

Two new concepts:
  CHUNK SIZE    — max characters per chunk (e.g. 500)
  OVERLAP       — how many characters the next chunk re-uses from the previous one
                  so a sentence cut at a boundary still appears in full somewhere

Flow:
  PDF → extract text per page → chunk with overlap → embed → store → ask
"""

import json
import sys
import urllib.request
from pathlib import Path

import chromadb
from pypdf import PdfReader


OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL  = "llama3.2:3b"

CHUNK_SIZE  = 500   # characters per chunk
OVERLAP     = 100   # characters shared between consecutive chunks


# ── text extraction ───────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for every page in the PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i, text))
    return pages


# ── chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    Split text into overlapping windows.

    Example with chunk_size=20, overlap=5 and text="ABCDEFGHIJKLMNOPQRSTUVWXYZ":
      chunk 1: ABCDEFGHIJKLMNOPQRST          (pos 0–19)
      chunk 2:         NOPQRSTUVWXYZ...       (pos 15–34)  ← overlap of 5 chars
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step forward by (size - overlap)
    return chunks


def pdf_to_chunks(pdf_path: str) -> list[tuple[str, dict]]:
    """
    Returns list of (chunk_text, metadata) pairs ready for ChromaDB.
    Metadata carries source file name and page number for citations.
    """
    file_name = Path(pdf_path).name
    all_chunks = []
    pages = extract_pages(pdf_path)
    for page_num, page_text in pages:
        for chunk in chunk_text(page_text):
            if chunk.strip():
                meta = {"source": file_name, "page": page_num}
                all_chunks.append((chunk, meta))
    return all_chunks


# ── embedding ─────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["embedding"]


class OllamaEmbedder:
    def __call__(self, input: list[str]) -> list[list[float]]:
        return [get_embedding(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)


# ── generation ────────────────────────────────────────────────────────────────

def generate_answer(question: str, context_chunks: list[tuple[str, dict]]) -> str:
    context_text = ""
    for i, (chunk, meta) in enumerate(context_chunks, 1):
        context_text += f"[{i}] ({meta['source']} page {meta['page']})\n{chunk}\n\n"

    prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."
Cite which source and page you used.

Context:
{context_text}
Question: {question}
Answer:"""

    payload = json.dumps({
        "model": CHAT_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


# ── index a PDF ───────────────────────────────────────────────────────────────

def build_index(pdf_path: str):
    print(f"\nLoading: {pdf_path}")
    chunks = pdf_to_chunks(pdf_path)
    print(f"  Pages extracted, split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={OVERLAP})")

    client   = chromadb.Client()
    embedder = OllamaEmbedder()
    collection = client.create_collection(
        name="pdf_docs",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    print("  Embedding chunks (this may take ~30s)...")
    texts  = [c for c, _ in chunks]
    metas  = [m for _, m in chunks]
    ids    = [str(i) for i in range(len(chunks))]
    collection.add(documents=texts, ids=ids, metadatas=metas)
    print(f"  Index ready — {collection.count()} vectors stored.\n")
    return collection


# ── ask ───────────────────────────────────────────────────────────────────────

def ask(collection, question: str, top_k: int = 3) -> None:
    print(f"{'─'*60}")
    print(f"Q: {question}")

    results   = collection.query(query_texts=[question], n_results=top_k)
    chunks    = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    print("\nTop retrieved chunks:")
    for chunk, meta, dist in zip(chunks, metas, distances):
        score = 1 - dist
        preview = chunk[:80].replace("\n", " ")
        print(f"  [{score:.3f}] ({meta['source']} p{meta['page']}) {preview}...")

    answer = generate_answer(question, list(zip(chunks, metas)))
    print(f"\nA: {answer}\n")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    collection = build_index(pdf_path)

    ask(collection, "What is retrieval-augmented generation?")
    ask(collection, "What are the three types of machine learning?")
    ask(collection, "What is HNSW and why is it used?")
    ask(collection, "Name some examples of large language models.")
    # out-of-scope question
    ask(collection, "What is the capital of France?")
