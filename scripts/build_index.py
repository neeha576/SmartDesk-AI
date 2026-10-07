"""Build the SmartDesk knowledge-base index in Qdrant.

Reads knowledge_base/ (Markdown policies + Q&A pairs) -> chunks (500 tokens, 50 overlap)
-> dense embeddings (EMBEDDING_PROVIDER / EMBEDDING_MODEL from .env) + BM25 sparse vectors
-> upserts into the Qdrant collection (QDRANT_COLLECTION), then runs sample queries as a check.

Usage (from the project root):
    python scripts/build_index.py              # rebuild the collection from scratch
    python scripts/build_index.py --dry-run    # load + chunk only, print stats, no API calls
    python scripts/build_index.py --no-recreate  # upsert into the existing collection
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config.settings  # noqa: E402,F401  (loads .env before anything reads env vars)
from rag.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents, count_tokens, qa_to_chunks  # noqa: E402
from rag.loader import load_markdown_docs, load_qa_pairs  # noqa: E402

SAMPLE_QUERIES = [
    ("How do I reset my password?", "IT-001"),
    ("How many sick days do I get per year?", "HR-001"),
    ("When is payday?", "PAY-001"),
    ("My monitor has been flickering for two days.", None),  # deliberate gap -> low score expected
]


def parse_args():
    p = argparse.ArgumentParser(description="Ingest the knowledge base into Qdrant.")
    p.add_argument("--kb-dir", default=str(ROOT / "knowledge_base"))
    p.add_argument("--dry-run", action="store_true", help="Load and chunk only; no embeddings or Qdrant writes.")
    p.add_argument("--no-recreate", action="store_true", help="Keep the existing collection and upsert into it.")
    p.add_argument("--skip-check", action="store_true", help="Skip the sample-query check at the end.")
    return p.parse_args()


def build_chunks(kb_dir: str) -> list[dict]:
    docs = load_markdown_docs(kb_dir)
    qa = load_qa_pairs(kb_dir)
    chunks = chunk_documents(docs) + qa_to_chunks(qa)
    tokens = [count_tokens(c["text"]) for c in chunks]
    policy = [t for c, t in zip(chunks, tokens) if c["chunk_type"] == "policy"]
    print(f"Loaded {len(docs)} policy documents and {len(qa)} Q&A pairs from {kb_dir}")
    print(f"Chunking: size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP} tokens")
    print(f"  -> {len(policy)} policy chunks (tokens min/avg/max: "
          f"{min(policy)}/{sum(policy) // len(policy)}/{max(policy)}) + {len(qa)} Q&A chunks "
          f"= {len(chunks)} total")
    by_domain = Counter(c["domain"] for c in chunks)
    print(f"  -> by domain: {dict(by_domain)}")
    return chunks


def sample_check(client, dense):
    from rag.vector_store import COLLECTION, DENSE
    print("\nSample queries (dense cosine score, top hit):")
    for query, expected in SAMPLE_QUERIES:
        hits = client.query_points(COLLECTION, query=dense.embed_query(query), using=DENSE,
                                   limit=1, with_payload=True).points
        if not hits:
            print(f"  [no results] {query}")
            continue
        top = hits[0]
        mark = "" if expected is None else ("OK " if top.payload["doc_id"] == expected else "?? ")
        print(f"  {mark}{top.score:.3f}  {top.payload['doc_id']:<8} {query}")
    print("The gap question should score clearly lower than the others; use that to tune CONFIDENCE_THRESHOLD.")


def main():
    args = parse_args()
    chunks = build_chunks(args.kb_dir)
    if args.dry_run:
        print("\nDry run – nothing embedded or written.")
        return

    from rag.embeddings import get_dense_embedder, get_sparse_embedder
    from rag.vector_store import COLLECTION, ensure_collection, get_client, upsert_chunks

    texts = [c["text"] for c in chunks]
    t0 = time.time()
    dense = get_dense_embedder()
    dense_vecs = dense.embed_documents(texts)
    print(f"\nDense embeddings: {len(dense_vecs)} x {len(dense_vecs[0])} dims ({time.time() - t0:.1f}s)")

    t0 = time.time()
    sparse_vecs = list(get_sparse_embedder().embed(texts))
    print(f"Sparse BM25 vectors: {len(sparse_vecs)} ({time.time() - t0:.1f}s)")

    client = get_client()
    try:
        ensure_collection(client, dense_dim=len(dense_vecs[0]), recreate=not args.no_recreate)
        n = upsert_chunks(client, chunks, dense_vecs, sparse_vecs)
        total = client.count(COLLECTION, exact=True).count
        print(f"Upserted {n} points into '{COLLECTION}' (collection now holds {total}).")
        # cached answers may be out of date now – start the semantic cache fresh
        from memory.semantic_cache import SemanticCache
        SemanticCache(client, dense).clear()
        print("Semantic cache cleared.")
        if not args.skip_check:
            sample_check(client, dense)
    finally:
        client.close()


if __name__ == "__main__":
    main()
