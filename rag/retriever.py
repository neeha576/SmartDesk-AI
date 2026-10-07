"""HybridRetriever on Qdrant: dense + BM25 sparse candidates fused with Qdrant's built-in RRF.

RRF scores are rank-based, not similarity, so each hit also carries its dense cosine similarity
('dense_score'). The confidence check (rag/confidence.py) uses dense_score, not the RRF score.
"""
from __future__ import annotations

from qdrant_client import models

from rag.vector_store import COLLECTION, DENSE, SPARSE, get_client

CANDIDATES = 20  # per-branch candidates before fusion


class HybridRetriever:
    def __init__(self, dense_embedder, sparse_embedder, client=None):
        self.client = client or get_client()
        self.dense = dense_embedder
        self.sparse = sparse_embedder

    def search(self, query: str, domain: str | None = None, k: int = 5) -> list[dict]:
        """Return up to k hits: [{'text', 'doc_id', 'title', 'domain', ..., 'score', 'dense_score'}]."""
        dense_q = self.dense.embed_query(query)
        sparse_q = next(iter(self.sparse.query_embed(query)))
        flt = (models.Filter(must=[models.FieldCondition(key="domain", match=models.MatchValue(value=domain))])
               if domain else None)

        # 1) Dense-only search: gives true cosine similarity for every dense candidate
        dense_hits = self.client.query_points(
            COLLECTION, query=dense_q, using=DENSE, query_filter=flt,
            limit=CANDIDATES, with_payload=False).points
        dense_scores = {str(h.id): h.score for h in dense_hits}

        # 2) Hybrid search: dense + BM25 prefetch fused with Reciprocal Rank Fusion
        fused = self.client.query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_q, using=DENSE, limit=CANDIDATES, filter=flt),
                models.Prefetch(query=models.SparseVector(indices=sparse_q.indices.tolist(),
                                                          values=sparse_q.values.tolist()),
                                using=SPARSE, limit=CANDIDATES, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k, with_payload=True).points

        return [{**h.payload, "score": h.score, "dense_score": dense_scores.get(str(h.id), 0.0)}
                for h in fused]
