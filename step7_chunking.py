"""
STEP 7: Chunking strategies — why HOW you split matters as much as WHAT you split.

The problem with fixed-character chunking (what we built in steps 5-6):
  text = "...neural networks excel at image recognition. RAG combines a retrieval..."
  chunk 1: "...neural networks excel at image recogni"   ← cuts mid-word
  chunk 2: "tion. RAG combines a retrieval..."           ← starts with "tion"

When "tion" gets embedded, the model has no idea what word it belongs to.
Retrieval for "image recognition" might miss it entirely.

Four strategies, in order of quality:

  1. FIXED CHARACTER  — split every N chars (what we had)
  2. SENTENCE-AWARE   — accumulate whole sentences until size limit
  3. PARAGRAPH-AWARE  — split on paragraph breaks (\n\n), fallback to sentences
  4. RECURSIVE        — try paragraph → sentence → word → character (best general purpose)

We run ALL FOUR on the same document and compare:
  - where do chunk boundaries fall?
  - how many chunks are produced?
  - which strategy retrieves the most relevant chunk for a given query?
"""

import json
import math
import re
import urllib.request

import chromadb


OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"

# A realistic multi-paragraph document to chunk
DOCUMENT = """
Machine learning is a branch of artificial intelligence that enables systems to learn from data and improve their performance without being explicitly programmed. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

In supervised learning, models are trained on labelled data where each input has a corresponding correct output. Common examples include email spam detection, image classification, and house price prediction. The model learns to map inputs to outputs by minimising a loss function during training.

A neural network is a computational model loosely inspired by the human brain. It consists of layers of interconnected nodes called neurons. The first layer is the input layer, the last is the output layer, and everything in between are hidden layers. Deep learning refers to neural networks with many hidden layers.

Retrieval-Augmented Generation, or RAG, combines a retrieval system with a language model. Instead of relying solely on the model's memorised knowledge, RAG fetches relevant documents at query time and provides them as context. This reduces hallucination and allows the model to cite sources. The pipeline has three stages: embed documents into a vector space, retrieve the nearest chunks for a query, then generate an answer grounded in the retrieved context.

Vector databases store high-dimensional vectors and support fast similarity search. Popular choices include ChromaDB, Pinecone, Weaviate, and Qdrant. The most common indexing algorithm is HNSW, which builds a graph where nearby vectors are connected, enabling sub-linear search time even across millions of vectors.
""".strip()


# ── strategy 1: fixed character (baseline) ────────────────────────────────────

def chunk_fixed(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ── strategy 2: sentence-aware ────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? boundaries."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def chunk_sentences(text: str, size: int = 200, overlap_sentences: int = 1) -> list[str]:
    """
    Accumulate whole sentences into a chunk until adding the next sentence
    would exceed `size` characters. Carry the last `overlap_sentences`
    sentences into the next chunk so context isn't lost at boundaries.
    """
    sentences = split_sentences(text)
    chunks, current = [], []

    for sentence in sentences:
        current.append(sentence)
        if sum(len(s) for s in current) >= size:
            chunks.append(" ".join(current))
            # keep tail sentences as overlap for the next chunk
            current = current[-overlap_sentences:] if overlap_sentences else []

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ── strategy 3: paragraph-aware ───────────────────────────────────────────────

def chunk_paragraphs(text: str, size: int = 200, overlap_sentences: int = 1) -> list[str]:
    """
    Split on paragraph breaks first (\n\n).
    If a paragraph is still too long, fall back to sentence chunking.
    Short paragraphs are merged with the next one until `size` is reached.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks, buffer = [], ""

    for para in paragraphs:
        if len(para) > size:
            # paragraph too long — split it by sentence
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(chunk_sentences(para, size=size,
                                          overlap_sentences=overlap_sentences))
        elif len(buffer) + len(para) + 1 <= size:
            buffer = (buffer + " " + para).strip()
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para

    if buffer:
        chunks.append(buffer)

    return [c for c in chunks if c.strip()]


# ── strategy 4: recursive ─────────────────────────────────────────────────────

def chunk_recursive(text: str, size: int = 200, overlap: int = 40,
                    separators: list[str] | None = None) -> list[str]:
    """
    Try splitting by separators from coarsest to finest:
      paragraph break → sentence end → space (word) → character

    For each split result:
      - if the piece is small enough, keep it as a chunk
      - if it's still too big, recurse with the next separator
      - merge small adjacent pieces if they fit together
    """
    if separators is None:
        separators = ["\n\n", r"(?<=[.!?])\s+", r"\s+", ""]

    def _split(t: str, sep_index: int) -> list[str]:
        if len(t) <= size or sep_index >= len(separators):
            return [t]

        sep = separators[sep_index]
        parts = re.split(sep, t) if sep else list(t)
        parts = [p for p in parts if p]

        result: list[str] = []
        buffer = ""

        for part in parts:
            if len(buffer) + len(part) + 1 <= size:
                buffer = (buffer + (" " if buffer else "") + part)
            else:
                if buffer:
                    result.append(buffer)
                # piece itself might still be too big — recurse
                if len(part) > size:
                    result.extend(_split(part, sep_index + 1))
                    buffer = ""
                else:
                    buffer = part

        if buffer:
            result.append(buffer)

        return result

    raw = _split(text, 0)

    # add overlap: for each chunk (except the first), prepend the tail of the previous
    if overlap <= 0:
        return [c.strip() for c in raw if c.strip()]

    overlapped = [raw[0]]
    for i in range(1, len(raw)):
        tail = raw[i - 1][-overlap:]
        overlapped.append(tail + " " + raw[i])

    return [c.strip() for c in overlapped if c.strip()]


# ── comparison helpers ────────────────────────────────────────────────────────

def show_strategy(name: str, chunks: list[str]) -> None:
    sizes = [len(c) for c in chunks]
    avg   = sum(sizes) / len(sizes) if sizes else 0
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Chunks : {len(chunks)}")
    print(f"  Sizes  : min={min(sizes)}  avg={avg:.0f}  max={max(sizes)}")
    print(f"\n  First boundary (end of chunk 1 / start of chunk 2):")
    print(f"  ...{chunks[0][-60:]!r}")
    print(f"   {chunks[1][:60]!r}...")


# ── retrieval comparison ──────────────────────────────────────────────────────

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
    is_legacy = False
    def __call__(self, input: list[str]) -> list[list[float]]:
        return [get_embedding(t) for t in input]
    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)
    def name(self) -> str:
        return "ollama-embedder"


def build_collection(name: str, chunks: list[str]) -> chromadb.Collection:
    client   = chromadb.Client()
    embedder = OllamaEmbedder()
    col = client.create_collection(
        name=name,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    col.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))],
    )
    return col


def compare_retrieval(collections: dict[str, chromadb.Collection], query: str) -> None:
    print(f"\n{'='*60}")
    print(f"Query: {query!r}")
    print(f"{'='*60}")

    for name, col in collections.items():
        res   = col.query(query_texts=[query], n_results=1)
        score = 1 - res["distances"][0][0]
        best  = res["documents"][0][0]
        print(f"\n  [{name}]")
        print(f"    Score : {score:.4f}")
        print(f"    Chunk : {best[:120]!r}...")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CHUNK_SIZE = 250   # target chars per chunk for all strategies

    strategies = {
        "1-fixed":     chunk_fixed(DOCUMENT,      size=CHUNK_SIZE, overlap=40),
        "2-sentence":  chunk_sentences(DOCUMENT,  size=CHUNK_SIZE, overlap_sentences=1),
        "3-paragraph": chunk_paragraphs(DOCUMENT, size=CHUNK_SIZE, overlap_sentences=1),
        "4-recursive": chunk_recursive(DOCUMENT,  size=CHUNK_SIZE, overlap=40),
    }

    # ── Part A: show what each strategy does to the text ──
    print("PART A: What does each strategy produce?\n")
    for name, chunks in strategies.items():
        show_strategy(name, chunks)

    # ── Part B: which retrieves best for specific queries ──
    print("\n\nPART B: Retrieval quality comparison\n")
    print("Building one index per strategy (takes ~30s)...")

    collections = {name: build_collection(name, chunks)
                   for name, chunks in strategies.items()}

    queries = [
        "What is retrieval augmented generation?",
        "How do neural networks work?",
        "Why is HNSW used in vector databases?",
    ]

    for query in queries:
        compare_retrieval(collections, query)
