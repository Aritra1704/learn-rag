"""
STEP 1: What is an embedding?

An embedding is just a list of numbers (a vector) that represents the
*meaning* of a piece of text. Similar sentences get similar vectors.

We'll call Ollama's API directly — no libraries, just urllib.
"""

import json
import urllib.request


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"


def get_embedding(text: str) -> list[float]:
    """Ask Ollama to convert text into a vector of numbers."""
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["embedding"]


if __name__ == "__main__":
    sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",   # similar meaning to sentence 1
        "The stock market crashed today.", # unrelated
    ]

    print(f"Embedding model: {EMBED_MODEL}\n")

    embeddings = []
    for sentence in sentences:
        vec = get_embedding(sentence)
        embeddings.append(vec)
        # Don't print all 768 numbers — just show the shape
        print(f"Text : {sentence!r}")
        print(f"Vector length : {len(vec)}")
        print(f"First 5 values: {vec[:5]}\n")
