"""Semantic cache for repeated IT / HR questions.

    "How many sick days do I get?"  ─┐
    "what's my sick leave allowance" ─┼─► embed ─► nearest cached question (same domain) ─► score ≥ 0.93 ? ─► cached answer
    "sick days per year?"            ─┘                                                       └─ no ─► normal RAG pipeline ─► store

Stored in its own Qdrant collection (CACHE_COLLECTION, default "smartdesk_cache") next to the knowledge base,
using the same client and the same dense embedding model as retrieval, so a lookup costs one vector search
(the question's embedding is memoized and reused by retrieval on a miss).

What is cached – only confident, grounded IT / HR knowledge-base answers. Never: ticket status or ticket creation
(personal, changing), "couldn't find it" escalations, partial answers, small talk, or short follow-ups whose meaning
depends on the conversation ("what about on a Mac?"). agents/kb_agent.py decides this.

Safety:
  - entries are filtered by domain, so an IT answer is never served for an HR question;
  - each entry carries the knowledge-base version (hash of knowledge_base/); after the KB changes, old entries are
    ignored – scripts/build_index.py also clears the cache when it rebuilds the index;
  - entries expire after CACHE_TTL_HOURS;
  - any cache error is logged and treated as a miss – the cache can never break an answer.

Settings (.env, all optional): SEMANTIC_CACHE_ENABLED=true, CACHE_COLLECTION=smartdesk_cache,
CACHE_THRESHOLD=0.93, CACHE_TTL_HOURS=168.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path

from qdrant_client import models

from config import settings

log = logging.getLogger(__name__)

KB_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"


def normalize(question: str) -> str:
    """Lowercase, collapse spaces and drop trailing punctuation – used for the point id and logs."""
    return re.sub(r"\s+", " ", question.strip().lower()).rstrip("?!. ")


@lru_cache(maxsize=1)
def kb_version() -> str:
    """Short hash of the knowledge-base files; changes whenever a document or Q&A pair changes."""
    h = hashlib.sha256()
    for p in sorted(KB_DIR.rglob("*")):
        if p.is_file() and p.suffix in (".md", ".json", ".txt"):
            h.update(str(p.relative_to(KB_DIR)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


class SemanticCache:
    def __init__(self, client, embedder, collection: str | None = None,
                 threshold: float | None = None, ttl_hours: float | None = None):
        self.client = client
        self.embedder = embedder
        self.collection = collection or settings.CACHE_COLLECTION
        self.threshold = settings.CACHE_THRESHOLD if threshold is None else threshold
        self.ttl_seconds = (settings.CACHE_TTL_HOURS if ttl_hours is None else ttl_hours) * 3600

    # ------------------------------------------------------------------ lookup
    def lookup(self, question: str, domain: str) -> dict | None:
        """Best cached answer for a semantically similar question in this domain, or None.

        Returns {'answer', 'sources', 'question' (the cached one), 'score'}.
        """
        try:
            if not self.client.collection_exists(self.collection):
                return None
            hits = self.client.query_points(
                self.collection, query=self.embedder.embed_query(question), limit=1,
                score_threshold=self.threshold, with_payload=True,
                query_filter=models.Filter(must=[
                    models.FieldCondition(key="domain", match=models.MatchValue(value=domain)),
                    models.FieldCondition(key="kb_version", match=models.MatchValue(value=kb_version())),
                    models.FieldCondition(key="created_at",
                                          range=models.Range(gte=time.time() - self.ttl_seconds)),
                ])).points
        except Exception as e:  # noqa: BLE001 – a broken cache is just a miss
            log.warning("Semantic cache lookup failed: %s", e)
            return None
        if not hits:
            log.info("Cache miss (%s): %s", domain, normalize(question))
            return None
        p = hits[0].payload
        log.info("Cache HIT (%s, %.3f): '%s' ≈ '%s'", domain, hits[0].score, normalize(question), p["question"])
        return {"answer": p["answer"], "sources": p.get("sources", []), "question": p["question"],
                "score": hits[0].score}

    # ------------------------------------------------------------------ store
    def store(self, question: str, domain: str, answer: str, sources: list[str]) -> None:
        """Remember a grounded answer. Re-asking the same question overwrites its entry (same point id)."""
        try:
            vector = self.embedder.embed_query(question)
            self._ensure_collection(len(vector))
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{domain}|{normalize(question)}"))
            self.client.upsert(self.collection, wait=True, points=[models.PointStruct(
                id=point_id, vector=vector,
                payload={"question": question.strip(), "domain": domain, "answer": answer, "sources": sources,
                         "kb_version": kb_version(), "created_at": time.time()})])
            log.info("Cached answer (%s): %s", domain, normalize(question))
        except Exception as e:  # noqa: BLE001
            log.warning("Semantic cache store failed: %s", e)

    def clear(self) -> None:
        """Drop every cached answer (called after the knowledge base is re-indexed)."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def _ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(self.collection, vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.COSINE))
        if not settings.QDRANT_URL:                 # payload indexes only matter on a Qdrant server
            return
        for field, kind in (("domain", models.PayloadSchemaType.KEYWORD),
                            ("kb_version", models.PayloadSchemaType.KEYWORD),
                            ("created_at", models.PayloadSchemaType.FLOAT)):
            self.client.create_payload_index(self.collection, field_name=field, field_schema=kind)


_cache: SemanticCache | None = None


def get_cache() -> SemanticCache | None:
    """The app-wide cache (shares the retriever's Qdrant client and embedder), or None if disabled."""
    global _cache
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None
    if _cache is None:
        from tools.kb_search import _get_retriever
        r = _get_retriever()
        _cache = SemanticCache(r.client, r.dense)
    return _cache
