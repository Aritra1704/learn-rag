"""
STEP 6: Persistent index + multi-document RAG.

Two problems with step 5:
  - ChromaDB lived in memory — every run re-embedded everything from scratch.
    Embedding is slow (one LLM call per chunk). For 1000 chunks that's painful.
  - We only handled a single PDF.

Solutions introduced here:
  - PersistentClient  saves the ChromaDB index to disk (./chroma_db/).
                      Second run loads it instantly — no re-embedding.
  - File manifest     we track which files are already indexed using a
                      separate ChromaDB collection called "_manifest".
                      On startup we skip any file whose path+size hasn't changed.
  - Folder scan       index every PDF in a directory in one command.

Run it twice and watch what happens:
  First run  → embeds all chunks, saves to disk
  Second run → loads from disk, skips already-indexed files, starts instantly
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import chromadb
from pypdf import PdfReader


OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL  = "llama3.2:3b"
DB_PATH     = "./chroma_db"   # where the index lives on disk

CHUNK_SIZE  = 500
OVERLAP     = 100


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

    def name(self) -> str:
        return "ollama-embedder"


# ── chunking ──────────────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def pdf_to_chunks(pdf_path: str) -> list[tuple[str, dict]]:
    file_name = Path(pdf_path).name
    result = []
    for page_num, page_text in extract_pages(pdf_path):
        for chunk in chunk_text(page_text):
            result.append((chunk, {"source": file_name, "page": page_num}))
    return result


# ── manifest: track what's already indexed ────────────────────────────────────

def _file_fingerprint(pdf_path: str) -> str:
    """A string that changes if the file content changes."""
    p = Path(pdf_path)
    return f"{p.name}::{p.stat().st_size}"


def is_indexed(manifest, pdf_path: str) -> bool:
    fp = _file_fingerprint(pdf_path)
    results = manifest.get(ids=[fp])
    return len(results["ids"]) > 0


def mark_indexed(manifest, pdf_path: str) -> None:
    fp = _file_fingerprint(pdf_path)
    manifest.upsert(documents=[fp], ids=[fp])


# ── indexing ──────────────────────────────────────────────────────────────────

def index_pdf(collection, manifest, pdf_path: str) -> int:
    """Index a single PDF. Returns number of chunks added (0 if skipped)."""
    if is_indexed(manifest, pdf_path):
        print(f"  [skip]  {Path(pdf_path).name}  (already in index)")
        return 0

    chunks = pdf_to_chunks(pdf_path)
    if not chunks:
        print(f"  [empty] {Path(pdf_path).name}")
        return 0

    # Use file path + chunk position as a stable unique ID
    base_id = Path(pdf_path).stem
    ids    = [f"{base_id}::{i}" for i in range(len(chunks))]
    texts  = [c for c, _ in chunks]
    metas  = [m for _, m in chunks]

    print(f"  [index] {Path(pdf_path).name}  → {len(chunks)} chunks ...", end="", flush=True)
    t0 = time.time()
    collection.add(documents=texts, ids=ids, metadatas=metas)
    mark_indexed(manifest, pdf_path)
    print(f" done in {time.time() - t0:.1f}s")
    return len(chunks)


def index_folder(collection, manifest, folder: str) -> None:
    pdfs = sorted(Path(folder).glob("**/*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        return

    print(f"\nFound {len(pdfs)} PDF(s) in '{folder}':")
    total = 0
    for pdf in pdfs:
        total += index_pdf(collection, manifest, str(pdf))

    stored = collection.count()
    print(f"\nIndex ready — {stored} total vectors on disk at '{DB_PATH}'\n")


# ── generation ────────────────────────────────────────────────────────────────

def generate_answer(question: str, context_chunks: list[tuple[str, dict]]) -> str:
    context_text = ""
    for i, (chunk, meta) in enumerate(context_chunks, 1):
        context_text += f"[{i}] ({meta['source']} page {meta['page']})\n{chunk}\n\n"

    prompt = (
        "You are a helpful assistant. Answer using ONLY the context below.\n"
        "If the answer is not in the context, say 'I don't know based on the provided documents.'\n"
        "Cite which source and page you used.\n\n"
        f"Context:\n{context_text}\n"
        f"Question: {question}\nAnswer:"
    )

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
        return json.loads(resp.read())["message"]["content"]


# ── query ─────────────────────────────────────────────────────────────────────

def ask(collection, question: str, top_k: int = 3) -> None:
    print(f"{'─'*60}")
    print(f"Q: {question}")

    results   = collection.query(query_texts=[question], n_results=top_k)
    chunks    = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    print("  Retrieved:")
    for chunk, meta, dist in zip(chunks, metas, distances):
        score   = 1 - dist
        preview = chunk[:70].replace("\n", " ")
        print(f"    [{score:.3f}] ({meta['source']} p{meta['page']}) {preview}...")

    answer = generate_answer(question, list(zip(chunks, metas)))
    print(f"\n  A: {answer}\n")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"

    # PersistentClient saves everything to DB_PATH on disk.
    # On second run it loads the existing index — no re-embedding needed.
    client   = chromadb.PersistentClient(path=DB_PATH)
    embedder = OllamaEmbedder()

    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    # Separate collection just to track which files we've indexed
    manifest = client.get_or_create_collection(name="index-manifest")

    index_folder(collection, manifest, folder)

    # These questions span all three documents
    ask(collection, "What is HNSW and why is it important?")
    ask(collection, "What are Python's four built-in collection types?")
    ask(collection, "Who was first to walk on the Moon?")
    ask(collection, "What triggered World War I?")
    ask(collection, "How does RAG reduce hallucination?")
