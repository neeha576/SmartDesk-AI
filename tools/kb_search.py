"""Knowledge-base search tool (hybrid: dense + BM25, RRF fusion)."""
from __future__ import annotations

from functools import lru_cache

from config import settings

_retriever = None


class _MemoQueryEmbedder:
    """Wraps the dense embedder so the same question is embedded once – the semantic cache lookup and
    the retrieval that follows a cache miss share one embedding call."""

    def __init__(self, embedder, size: int = 256):
        self._inner = embedder
        self._cached = lru_cache(maxsize=size)(lambda text: tuple(embedder.embed_query(text)))

    def embed_query(self, text: str) -> list[float]:
        return list(self._cached(text))

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _get_retriever():
    global _retriever
    if _retriever is None:
        from rag.embeddings import get_dense_embedder, get_sparse_embedder
        from rag.retriever import HybridRetriever
        _retriever = HybridRetriever(_MemoQueryEmbedder(get_dense_embedder()), get_sparse_embedder())
    return _retriever


def kb_search(query: str, domain: str | None = None, k: int | None = None) -> dict:
    """Return {'chunks': [...], 'top_score': float | None}.

    top_score is the highest dense cosine similarity among the returned chunks, or None if
    nothing was retrieved. domain filters to 'IT' or 'HR'.
    """
    chunks = _get_retriever().search(query, domain=domain, k=k or settings.RETRIEVAL_TOP_K)
    top = max((c["dense_score"] for c in chunks), default=None)
    return {"chunks": chunks, "top_score": top}
