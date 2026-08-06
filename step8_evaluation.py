"""
STEP 8: Evaluation — measuring whether your RAG actually works.

Without evaluation you're guessing. A pipeline that "looks fine" on 3 manual
tests might silently fail on 30% of real queries.

Two things to measure:

  RETRIEVAL QUALITY — did the right chunk come back?
    • Hit Rate @k  : was the correct chunk anywhere in the top-k results?   (0 or 1)
    • MRR          : Mean Reciprocal Rank — rewards finding it at rank 1
                     over rank 3. MRR=1.0 means always first. MRR=0.33 means
                     usually third.
    • Precision @k : of the k chunks returned, how many were actually relevant?

  ANSWER QUALITY — did the LLM answer well?
    • Faithfulness : does the answer use ONLY the retrieved context, or does
                     it hallucinate facts not present in the chunks?
                     Scored by a second LLM call that acts as a judge (0–1).
    • Relevance    : does the answer actually address the question?
                     Also scored by the judge LLM (0–1).

GOLDEN DATASET
  A small hand-crafted set of (question, expected_phrase) pairs.
  "expected_phrase" is a substring that must appear in the retrieved chunk
  for the retrieval to be considered correct.
  In production you'd build this from real user queries + human labels.

We run all metrics across the four chunking strategies from step 7 so you
can see the number difference, not just feel it.
"""

import json
import urllib.request

import chromadb

from step7_chunking import (
    DOCUMENT,
    OllamaEmbedder,
    chunk_fixed,
    chunk_paragraphs,
    chunk_recursive,
    chunk_sentences,
)


OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2:3b"

# ── golden dataset ────────────────────────────────────────────────────────────
# Each entry: question the user might ask, and a phrase that MUST appear in
# the correctly retrieved chunk. Keep phrases short and unique to one section.

GOLDEN = [
    {
        "question": "What is retrieval-augmented generation?",
        "expected_phrase": "Retrieval-Augmented Generation",
        "reference_answer": (
            "RAG combines a retrieval system with a language model to fetch "
            "relevant documents at query time instead of relying on memorised knowledge."
        ),
    },
    {
        "question": "What are the three types of machine learning?",
        "expected_phrase": "supervised learning, unsupervised learning, and reinforcement",
        "reference_answer": (
            "The three types are supervised learning, unsupervised learning, "
            "and reinforcement learning."
        ),
    },
    {
        "question": "How do neural networks relate to the human brain?",
        "expected_phrase": "loosely inspired by the human brain",
        "reference_answer": (
            "A neural network is a computational model loosely inspired by the "
            "human brain, consisting of layers of interconnected nodes called neurons."
        ),
    },
    {
        "question": "What indexing algorithm do vector databases use?",
        "expected_phrase": "HNSW",
        "reference_answer": (
            "The most common indexing algorithm is HNSW (Hierarchical Navigable "
            "Small World), which enables sub-linear search time."
        ),
    },
    {
        "question": "How does a model learn in supervised learning?",
        "expected_phrase": "minimising a loss function",
        "reference_answer": (
            "In supervised learning the model learns to map inputs to outputs "
            "by minimising a loss function during training."
        ),
    },
    {
        "question": "What does RAG do to reduce hallucination?",
        "expected_phrase": "reduces hallucination",
        "reference_answer": (
            "RAG reduces hallucination by grounding answers in retrieved context "
            "and allowing the model to cite sources."
        ),
    },
]


# ── retrieval metrics ─────────────────────────────────────────────────────────

def reciprocal_rank(chunks: list[str], expected_phrase: str) -> float:
    """1/rank if expected_phrase found in retrieved chunks, else 0."""
    for rank, chunk in enumerate(chunks, start=1):
        if expected_phrase.lower() in chunk.lower():
            return 1.0 / rank
    return 0.0


def hit_at_k(chunks: list[str], expected_phrase: str) -> int:
    """1 if expected_phrase appears in any of the top-k chunks, else 0."""
    return int(any(expected_phrase.lower() in c.lower() for c in chunks))


def precision_at_k(chunks: list[str], expected_phrase: str) -> float:
    """Fraction of retrieved chunks that contain the expected phrase."""
    hits = sum(1 for c in chunks if expected_phrase.lower() in c.lower())
    return hits / len(chunks) if chunks else 0.0


def evaluate_retrieval(collection, k: int = 3) -> dict:
    rr_scores, hit_scores, prec_scores = [], [], []

    for item in GOLDEN:
        results = collection.query(query_texts=[item["question"]], n_results=k)
        chunks  = results["documents"][0]
        phrase  = item["expected_phrase"]

        rr_scores.append(reciprocal_rank(chunks, phrase))
        hit_scores.append(hit_at_k(chunks, phrase))
        prec_scores.append(precision_at_k(chunks, phrase))

    return {
        "MRR":           sum(rr_scores)   / len(rr_scores),
        f"Hit@{k}":      sum(hit_scores)  / len(hit_scores),
        f"Precision@{k}": sum(prec_scores) / len(prec_scores),
    }


# ── answer quality: LLM-as-judge ──────────────────────────────────────────────

def llm_call(prompt: str) -> str:
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


def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    prompt  = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return llm_call(prompt)


def judge_faithfulness(question: str, context: str, answer: str) -> float:
    """
    Ask a second LLM call to score whether the answer is grounded in context.
    Returns a float 0.0–1.0 extracted from the judge's response.
    """
    prompt = f"""You are an evaluation judge. Score the answer below on FAITHFULNESS.

Faithfulness means: every claim in the answer is supported by the context.
An answer that adds facts not in the context scores 0.
An answer using only context scores 1.

Question : {question}
Context  : {context}
Answer   : {answer}

Reply with ONLY a JSON object like: {{"score": 0.8, "reason": "one sentence"}}
Do not add any other text."""

    raw = llm_call(prompt)
    try:
        # extract the first {...} from the response
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        data  = json.loads(raw[start:end])
        return float(data.get("score", 0.0))
    except Exception:
        return 0.0


def judge_relevance(question: str, answer: str) -> float:
    """Score whether the answer actually addresses the question (0–1)."""
    prompt = f"""You are an evaluation judge. Score the answer on RELEVANCE.

Relevance means: the answer directly addresses what the question asked.
A tangential or off-topic answer scores 0. A direct, complete answer scores 1.

Question : {question}
Answer   : {answer}

Reply with ONLY a JSON object like: {{"score": 0.9, "reason": "one sentence"}}
Do not add any other text."""

    raw = llm_call(prompt)
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        data  = json.loads(raw[start:end])
        return float(data.get("score", 0.0))
    except Exception:
        return 0.0


def evaluate_answers(collection, k: int = 3) -> dict:
    faith_scores, rel_scores = [], []

    for item in GOLDEN:
        results  = collection.query(query_texts=[item["question"]], n_results=k)
        chunks   = results["documents"][0]
        context  = "\n\n".join(chunks)
        answer   = generate_answer(item["question"], chunks)

        faith_scores.append(judge_faithfulness(item["question"], context, answer))
        rel_scores.append(judge_relevance(item["question"], answer))

    return {
        "Faithfulness": sum(faith_scores) / len(faith_scores),
        "Relevance":    sum(rel_scores)   / len(rel_scores),
    }


# ── index builder ─────────────────────────────────────────────────────────────

def build_collection(name: str, chunks: list[str]) -> chromadb.Collection:
    client   = chromadb.Client()
    embedder = OllamaEmbedder()
    col = client.create_collection(
        name=name,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    col.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
    return col


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CHUNK_SIZE = 250
    K = 3

    strategies = {
        "1-fixed":     chunk_fixed(DOCUMENT,      size=CHUNK_SIZE, overlap=40),
        "2-sentence":  chunk_sentences(DOCUMENT,  size=CHUNK_SIZE, overlap_sentences=1),
        "3-paragraph": chunk_paragraphs(DOCUMENT, size=CHUNK_SIZE, overlap_sentences=1),
        "4-recursive": chunk_recursive(DOCUMENT,  size=CHUNK_SIZE, overlap=40),
    }

    # ── PART A: retrieval metrics (fast — no LLM generation) ──────────────────
    print("=" * 65)
    print("PART A: Retrieval Quality")
    print(f"  Golden dataset : {len(GOLDEN)} questions")
    print(f"  k              : {K}  (top-{K} chunks retrieved per query)")
    print("=" * 65)
    print(f"\n  Building indexes...")

    retrieval_results = {}
    for name, chunks in strategies.items():
        print(f"    [{name}]  {len(chunks)} chunks — embedding...", end="", flush=True)
        col = build_collection(name, chunks)
        metrics = evaluate_retrieval(col, k=K)
        retrieval_results[name] = (col, metrics)
        print(f"  MRR={metrics['MRR']:.2f}  Hit@{K}={metrics[f'Hit@{K}']:.2f}")

    print(f"\n{'Strategy':<16} {'MRR':>6} {'Hit@'+str(K):>8} {'Prec@'+str(K):>10}")
    print("─" * 44)
    for name, (_, m) in retrieval_results.items():
        print(f"  {name:<14} {m['MRR']:>6.2f} {m[f'Hit@{K}']:>8.2f} {m[f'Precision@{K}']:>10.2f}")

    # ── PART B: answer quality on best-retrieval strategy ─────────────────────
    best_name = max(retrieval_results, key=lambda n: retrieval_results[n][1]["MRR"])
    best_col  = retrieval_results[best_name][0]

    print(f"\n{'=' * 65}")
    print(f"PART B: Answer Quality  (using best strategy: {best_name!r})")
    print(f"  Judge model : {CHAT_MODEL}")
    print(f"  Scoring {len(GOLDEN)} questions — this takes ~60s...")
    print("=" * 65)

    answer_metrics = evaluate_answers(best_col, k=K)

    print(f"\n  Faithfulness : {answer_metrics['Faithfulness']:.2f}  "
          f"(1.0 = answer uses only retrieved context, 0.0 = pure hallucination)")
    print(f"  Relevance    : {answer_metrics['Relevance']:.2f}  "
          f"(1.0 = directly answers the question)")

    # ── PART C: per-question breakdown ────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("PART C: Per-question breakdown")
    print("=" * 65)

    for item in GOLDEN:
        results = best_col.query(query_texts=[item["question"]], n_results=K)
        chunks  = results["documents"][0]
        phrase  = item["expected_phrase"]
        rr      = reciprocal_rank(chunks, phrase)
        hit     = hit_at_k(chunks, phrase)
        answer  = generate_answer(item["question"], chunks)
        faith   = judge_faithfulness(item["question"], "\n\n".join(chunks), answer)

        print(f"\n  Q: {item['question']}")
        print(f"     Hit={hit}  RR={rr:.2f}  Faithfulness={faith:.2f}")
        print(f"     A: {answer[:150].strip()}...")
