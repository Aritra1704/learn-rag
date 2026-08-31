"""
STEP 9: Reranking — cross-encoder vs bi-encoder.

Every step so far has used a BI-ENCODER:
  embed(query)  ──┐
                  ├──► cosine_similarity ──► ranked chunks
  embed(chunk)  ──┘

The query and the chunk are embedded SEPARATELY. Neither ever "sees" the
other. This is what makes vector search fast: you can pre-compute every
chunk's embedding once, store it in an ANN index (HNSW), and at query time
you only need to embed the query and do a nearest-neighbour lookup — even
across millions of chunks.

The cost of that speed is accuracy. Two chunks that are topically similar
but answer different questions can end up with near-identical embeddings,
because the embedding was never conditioned on what was actually being
asked.

A CROSS-ENCODER fixes this by scoring the pair jointly:
  score = model(query, chunk)      ← both go in together, one forward pass

Because the model can attend across query and chunk tokens simultaneously,
cross-encoders are consistently more accurate at judging relevance. The
cost: you can't precompute anything. Scoring N candidates means N forward
passes, done at query time. Cross-encoding an entire corpus of 100k chunks
per query would be far too slow — which is why cross-encoders are never
used for stage-1 retrieval.

THE STANDARD PATTERN — retrieve-then-rerank:
  Stage 1 (bi-encoder)  : ANN search over the whole corpus → top ~10-50 candidates. Fast.
  Stage 2 (cross-encoder): score just those candidates, pair by pair       → top-k. Slow, but only over a shortlist.

A dedicated cross-encoder is usually a small model fine-tuned purely to
output a relevance score for a (query, passage) pair (e.g. ms-marco-MiniLM).
We don't have one of those sitting in Ollama, so instead we approximate the
same idea with the general-purpose chat model we already have: give it the
query and the passage together, ask it to output a relevance score. This is
a real technique used in production (e.g. RankGPT, Cohere Rerank) — using an
LLM as a zero-shot cross-encoder — and it keeps us on pure Ollama with no
new dependency (no torch/sentence-transformers, which also isn't reliably
available yet for Python 3.14).
"""

import json

from step7_chunking import DOCUMENT, chunk_paragraphs
from step8_evaluation import GOLDEN, build_collection, hit_at_k, llm_call, reciprocal_rank


CHAT_MODEL = "llama3.2:3b"


# ── cross-encoder-style scoring ────────────────────────────────────────────────

def parse_score(raw: str, max_score: float = 10.0) -> float:
    """
    Extract the numeric "score" field from the judge's JSON reply and
    normalise it to 0.0-1.0. Same fragile-parse-with-fallback pattern as the
    judges in step 8: LLMs don't always follow formatting instructions
    perfectly, so a malformed reply degrades to a score of 0 rather than
    crashing the whole rerank.
    """
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        data  = json.loads(raw[start:end])
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score / max_score))
    except Exception:
        return 0.0


def score_relevance_llm(query: str, chunk: str) -> float:
    """
    Cross-encoder-style scoring: query and chunk go into the SAME prompt, so
    the model can weigh them jointly — unlike the bi-encoder, where they're
    embedded independently and only ever compared via a dot product.
    """
    prompt = f"""You are a relevance-scoring system used to rerank search results.

Score how relevant the PASSAGE is to the QUERY on a scale of 0-10.
0  = completely unrelated
10 = directly and completely answers the query

QUERY   : {query}
PASSAGE : {chunk}

Reply with ONLY a JSON object like: {{"score": 7}}
Do not add any other text."""

    raw = llm_call(prompt)
    return parse_score(raw, max_score=10.0)


# ── rerank stage ──────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[str],
    top_k: int = 3,
    scorer=score_relevance_llm,
) -> list[tuple[str, float]]:
    """
    Score every candidate against the query with `scorer`, then return the
    top_k, highest score first.

    `scorer` is injectable (defaults to the LLM-based cross-encoder above)
    so this function — and the sort logic that actually matters — can be
    unit-tested with a fake, deterministic scorer instead of hitting Ollama.
    """
    scored = [(chunk, scorer(query, chunk)) for chunk in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def retrieve_and_rerank(
    collection,
    query: str,
    k_candidates: int = 10,
    k_final: int = 3,
    scorer=score_relevance_llm,
) -> list[str]:
    """
    The full two-stage pipeline:
      stage 1 — bi-encoder ANN search narrows the corpus to k_candidates
      stage 2 — cross-encoder-style scorer reranks those down to k_final
    """
    results    = collection.query(query_texts=[query], n_results=k_candidates)
    candidates = results["documents"][0]
    reranked   = rerank(query, candidates, top_k=k_final, scorer=scorer)
    return [chunk for chunk, _score in reranked]


# ── walkthrough: watch reranking change the order for one query ───────────────

def show_rerank_walkthrough(collection, query: str, k_candidates: int = 10) -> None:
    print(f"\nQuery: {query!r}\n")

    results    = collection.query(query_texts=[query], n_results=k_candidates)
    candidates = results["documents"][0]
    distances  = results["distances"][0]

    print(f"  Stage 1 — bi-encoder ANN search (top {k_candidates} by cosine similarity):")
    for i, (chunk, dist) in enumerate(zip(candidates, distances), start=1):
        sim = 1 - dist
        print(f"    #{i:<2} cos_sim={sim:.3f}   {chunk[:70]!r}...")

    print(f"\n  Stage 2 — cross-encoder-style rerank (LLM scores each pair 0-10):")
    reranked = rerank(query, candidates, top_k=k_candidates)
    for i, (chunk, score) in enumerate(reranked, start=1):
        print(f"    #{i:<2} score={score:.2f}     {chunk[:70]!r}...")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    K_CANDIDATES = 10
    K_FINAL = 3

    print("=" * 65)
    print("STEP 9: Reranking")
    print("=" * 65)

    chunks = chunk_paragraphs(DOCUMENT, size=250)
    print(f"\nBuilding bi-encoder index ({len(chunks)} chunks, paragraph strategy)...")
    collection = build_collection("step9-rerank", chunks)

    # ── PART A: walk through one query in detail ───────────────────────────────
    print(f"\n{'=' * 65}")
    print("PART A: One query, watched closely")
    print("=" * 65)
    show_rerank_walkthrough(collection, GOLDEN[0]["question"], k_candidates=K_CANDIDATES)

    # ── PART B: does reranking improve the metrics from step 8? ────────────────
    print(f"\n{'=' * 65}")
    print(f"PART B: Bi-encoder-only vs +reranked, across all {len(GOLDEN)} golden questions")
    print("=" * 65)
    print(f"\n{'Question':<50} {'BiEnc Hit@3':>12} {'Rerank Hit@3':>13}")
    print("-" * 77)

    bi_rr, bi_hit = [], []
    rerank_rr, rerank_hit = [], []

    for item in GOLDEN:
        question = item["question"]
        phrase   = item["expected_phrase"]

        # stage 1 only: what plain bi-encoder retrieval would have returned
        bi_results = collection.query(query_texts=[question], n_results=K_FINAL)
        bi_chunks  = bi_results["documents"][0]
        bi_rr.append(reciprocal_rank(bi_chunks, phrase))
        bi_hit.append(hit_at_k(bi_chunks, phrase))

        # stage 1 + stage 2: wider candidate pool, then rerank down
        reranked_chunks = retrieve_and_rerank(
            collection, question, k_candidates=K_CANDIDATES, k_final=K_FINAL
        )
        rerank_rr.append(reciprocal_rank(reranked_chunks, phrase))
        rerank_hit.append(hit_at_k(reranked_chunks, phrase))

        print(f"  {question[:48]:<48} {bi_hit[-1]:>12} {rerank_hit[-1]:>13}")

    print(f"\n{'Metric':<10} {'Bi-encoder only':>18} {'+ Reranked':>14}")
    print("-" * 44)
    print(f"  {'MRR':<8} {sum(bi_rr) / len(bi_rr):>18.2f} {sum(rerank_rr) / len(rerank_rr):>14.2f}")
    print(f"  {'Hit@3':<8} {sum(bi_hit) / len(bi_hit):>18.2f} {sum(rerank_hit) / len(rerank_hit):>14.2f}")

    print(
        "\nNote: on a tiny 5-chunk document, bi-encoder retrieval is often already "
        "near-perfect, so the gap here may look small. The value of reranking grows "
        "with corpus size and query ambiguity — it's a precision fix for stage 1's "
        "recall-optimised shortlist, not a replacement for it."
    )
