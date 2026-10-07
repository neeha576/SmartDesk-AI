"""Calibrate CACHE_THRESHOLD for the semantic cache with your real embedding model.

    python scripts/cache_check.py

Prints the cosine similarity for pairs that SHOULD share a cached answer (paraphrases) and pairs that
should NOT (different questions on a similar topic). Pick a CACHE_THRESHOLD between the two groups:
above every "different" score, and below as many "same" scores as possible. When unsure, stay high –
a miss only costs one normal answer, a wrong hit gives a wrong answer.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from rag.embeddings import get_dense_embedder  # noqa: E402

SAME = [
    ("How many sick days do I get per year?", "What's my annual sick leave allowance?"),
    ("How do I reset my password?", "I forgot my password, how can I reset it?"),
    ("How do I connect to the VPN?", "What are the steps to connect to VPN?"),
    ("When is payday?", "What day do we get paid?"),
    ("How much PTO do I get?", "How many vacation days do I have?"),
]
DIFFERENT = [
    ("How many sick days do I get per year?", "How much PTO do I get?"),
    ("How do I reset my password?", "What are the password requirements?"),
    ("How do I connect to the VPN?", "Can I use the VPN on my personal laptop?"),
    ("When is payday?", "When will I get my first paycheck?"),
    ("How much paternity leave do I get?", "How much maternity leave do I get?"),
]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    emb = get_dense_embedder()
    print(f"Embedding model: {settings.embedding_model()}   current CACHE_THRESHOLD={settings.CACHE_THRESHOLD}\n")
    scores = {}
    for label, pairs in (("SAME (should hit)", SAME), ("DIFFERENT (must miss)", DIFFERENT)):
        print(label)
        scores[label] = []
        for a, b in pairs:
            s = cosine(emb.embed_query(a), emb.embed_query(b))
            scores[label].append(s)
            mark = "hit " if s >= settings.CACHE_THRESHOLD else "miss"
            print(f"  {s:.3f}  [{mark}]  {a}  ≈  {b}")
        print()
    hi_diff = max(scores["DIFFERENT (must miss)"])
    print(f"Highest 'different' score: {hi_diff:.3f} -> keep CACHE_THRESHOLD above it (e.g. {hi_diff + 0.02:.2f}).")


if __name__ == "__main__":
    main()
