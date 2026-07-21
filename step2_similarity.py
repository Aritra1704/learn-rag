"""
STEP 2: Cosine similarity — measuring how "close" two vectors are.

Two vectors that point in the same direction = similar meaning = score near 1.0
Two vectors pointing in different directions = unrelated = score near 0.0

We write this by hand. No libraries.
"""

import json
import math
import urllib.request


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"


def get_embedding(text: str) -> list[float]:
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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity = dot(a, b) / (|a| * |b|)

    dot product: multiply each pair of numbers and sum them up
    magnitude:   square root of sum of squares
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


if __name__ == "__main__":
    query = "What did the cat do?"

    candidates = [
        "The cat sat on the mat.",          # should score HIGH
        "A feline rested on the rug.",      # should score HIGH (synonym)
        "The stock market crashed today.",  # should score LOW
        "I love eating pizza on Fridays.",  # should score LOW
    ]

    print(f"Query: {query!r}\n")
    print("Embedding query...")
    query_vec = get_embedding(query)

    print("Embedding candidates and scoring...\n")
    results = []
    for text in candidates:
        vec = get_embedding(text)
        score = cosine_similarity(query_vec, vec)
        results.append((score, text))

    # Sort highest score first
    results.sort(reverse=True)

    for score, text in results:
        bar = "#" * int(score * 40)
        print(f"  {score:.4f}  {bar}")
        print(f"          {text!r}\n")

    print("---")
    print("The model doesn't know English grammar rules.")
    print("It just learned that similar-meaning sentences live close together in vector space.")
