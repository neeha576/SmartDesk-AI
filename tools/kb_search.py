"""Knowledge-base search tool (hybrid: dense + BM25, RRF fusion)."""
from __future__ import annotations

from config import settings

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from rag.embeddings import get_dense_embedder, get_sparse_embedder
        from rag.retriever import HybridRetriever
        _retriever = HybridRetriever(get_dense_embedder(), get_sparse_embedder())
    return _retriever


def kb_search(query: str, domain: str | None = None, k: int | None = None) -> dict:
    """Return {'chunks': [...], 'top_score': float | None}.

    top_score is the highest dense cosine similarity among the returned chunks, or None if
    nothing was retrieved. domain filters to 'IT' or 'HR'.
    """
    chunks = _get_retriever().search(query, domain=domain, k=k or settings.RETRIEVAL_TOP_K)
    top = max((c["dense_score"] for c in chunks), default=None)
    return {"chunks": chunks, "top_score": top}
