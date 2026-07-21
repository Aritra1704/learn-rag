"""
STEP 4: Full RAG pipeline.

RAG = Retrieval-Augmented Generation

The three moves:
  1. RETRIEVE  — embed the question, find top-k similar chunks in ChromaDB
  2. AUGMENT   — build a prompt that contains those chunks as context
  3. GENERATE  — send prompt to the LLM, get a grounded answer

Without RAG the LLM can only use what it memorised during training.
With RAG it reads your documents at query time and answers from them.
"""

import json
import urllib.request

import chromadb


OLLAMA_URL    = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text:latest"
CHAT_MODEL    = "llama3.2:3b"


# ── embedding (same as before) ────────────────────────────────────────────────

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


# ── LLM call ─────────────────────────────────────────────────────────────────

def generate_answer(question: str, context_chunks: list[tuple[str, dict]]) -> str:
    """
    Build a prompt from retrieved chunks and ask the LLM to answer.
    The LLM is told: use ONLY the provided context.
    This prevents hallucination — the model can't invent facts.
    """
    context_text = ""
    for i, (chunk, meta) in enumerate(context_chunks, 1):
        source = f"{meta.get('source', '?')} p{meta.get('page', '?')}"
        context_text += f"[{i}] ({source})\n{chunk}\n\n"

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say "I don't know based on the provided documents."
Always cite which source(s) you used.

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


# ── build the knowledge base ──────────────────────────────────────────────────

CHUNKS = [
    ("1", "The Eiffel Tower is located in Paris, France.",               {"source": "travel.txt", "page": 1}),
    ("2", "Paris is the capital and most populous city of France.",       {"source": "travel.txt", "page": 1}),
    ("3", "The Louvre is the world's largest art museum, also in Paris.", {"source": "travel.txt", "page": 2}),
    ("4", "Python was created by Guido van Rossum in 1991.",             {"source": "python.txt", "page": 1}),
    ("5", "Python is a high-level, general-purpose programming language.",{"source": "python.txt", "page": 1}),
    ("6", "The FIFA World Cup is held every four years.",                 {"source": "sports.txt", "page": 1}),
    ("7", "Argentina won the 2022 FIFA World Cup in Qatar.",              {"source": "sports.txt", "page": 2}),
]


def build_collection():
    client = chromadb.Client()
    embedder = OllamaEmbedder()
    collection = client.create_collection(
        name="my_docs",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    ids, texts, metas = zip(*CHUNKS)
    collection.add(documents=list(texts), ids=list(ids), metadatas=list(metas))
    return collection


# ── RAG function ──────────────────────────────────────────────────────────────

def ask(collection, question: str, top_k: int = 2) -> None:
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")

    # 1. RETRIEVE
    results = collection.query(query_texts=[question], n_results=top_k)
    chunks   = results["documents"][0]
    metas    = results["metadatas"][0]
    scores   = [1 - d for d in results["distances"][0]]

    print("\nRetrieved chunks:")
    for chunk, meta, score in zip(chunks, metas, scores):
        print(f"  [{score:.3f}] ({meta['source']} p{meta['page']}) {chunk}")

    # 2+3. AUGMENT + GENERATE
    print("\nGenerating answer...")
    answer = generate_answer(question, list(zip(chunks, metas)))
    print(f"\nA: {answer}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building knowledge base...")
    collection = build_collection()
    print(f"Stored {collection.count()} chunks.\n")

    ask(collection, "Where is the Eiffel Tower?")
    ask(collection, "Who created Python and when?")
    ask(collection, "Which country won the last World Cup?")

    # This question has NO answer in our documents — watch what happens
    ask(collection, "What is the capital of Japan?")
