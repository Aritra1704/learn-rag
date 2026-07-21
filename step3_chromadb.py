"""
STEP 3: ChromaDB — a real vector store.

In step 2 we scored 4 sentences by hand. That works for 4 items.
For 10,000 document chunks it's too slow and we need smarter indexing.

ChromaDB does that for us:
  - stores vectors on disk
  - builds an index (HNSW) for fast approximate nearest-neighbour search
  - lets us attach metadata (source file, page number, etc.) to each chunk

We still write our own embedding function backed by Ollama,
so there's no magic — we just hand the vectors to Chroma to manage.
"""

import json
import urllib.request

import chromadb


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"


# ---------- embedding (same as before) ----------

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


# ChromaDB wants an object with a __call__ method, not a plain function.
class OllamaEmbedder:
    def __call__(self, input: list[str]) -> list[list[float]]:
        return [get_embedding(text) for text in input]

    # newer ChromaDB versions call embed_query for query-time embedding
    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)


# ---------- our "documents" (pretend these are chunks from real files) ----------

CHUNKS = [
    # id,  text,                                               metadata
    ("1", "The Eiffel Tower is located in Paris, France.",    {"source": "travel.txt", "page": 1}),
    ("2", "Paris is the capital and most populous city of France.", {"source": "travel.txt", "page": 1}),
    ("3", "The Louvre is the world's largest art museum.",    {"source": "travel.txt", "page": 2}),
    ("4", "Python was created by Guido van Rossum in 1991.",  {"source": "python.txt", "page": 1}),
    ("5", "Python is a high-level, general-purpose programming language.", {"source": "python.txt", "page": 1}),
    ("6", "The FIFA World Cup is held every four years.",     {"source": "sports.txt", "page": 1}),
    ("7", "Argentina won the 2022 FIFA World Cup in Qatar.",  {"source": "sports.txt", "page": 2}),
]


if __name__ == "__main__":
    # --- 1. Create an in-memory ChromaDB collection ---
    client = chromadb.Client()  # in-memory; use chromadb.PersistentClient(path="./db") to save to disk
    embedder = OllamaEmbedder()

    collection = client.create_collection(
        name="my_docs",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},  # use cosine distance
    )

    print("Adding chunks to ChromaDB...\n")
    ids, texts, metadatas = zip(*CHUNKS)
    collection.add(documents=list(texts), ids=list(ids), metadatas=list(metadatas))
    print(f"  Stored {collection.count()} chunks.\n")

    # --- 2. Query ---
    queries = [
        "What is in Paris?",
        "Who made Python?",
        "football tournament winners",
    ]

    for query in queries:
        print(f"Query: {query!r}")
        results = collection.query(
            query_texts=[query],
            n_results=2,  # return top 2 most similar chunks
        )

        docs      = results["documents"][0]
        scores    = results["distances"][0]   # lower = more similar (cosine distance)
        metadatas = results["metadatas"][0]

        for doc, dist, meta in zip(docs, scores, metadatas):
            similarity = 1 - dist   # convert distance → similarity
            print(f"  [{similarity:.3f}] ({meta['source']} p{meta['page']}) {doc}")
        print()

    # --- 3. What ChromaDB is doing under the hood ---
    print("---")
    print("ChromaDB stored your vectors in an HNSW index.")
    print("HNSW = Hierarchical Navigable Small World graph.")
    print("It lets us find the top-k nearest vectors WITHOUT comparing every single one.")
    print("That's what makes it fast at scale.")
