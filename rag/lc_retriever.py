"""LangChain retriever over SmartDesk's Qdrant hybrid search.

Wraps tools.kb_search so it can be piped straight into an LCEL chain:
    retriever = SmartDeskRetriever(domain="IT")
    docs = retriever.invoke("How do I reset my password?")

Each Document carries the chunk text as page_content and the Qdrant payload as metadata,
including 'dense_score' (cosine similarity) used by the confidence threshold.
"""
from __future__ import annotations

import logging

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from tools.kb_search import kb_search

log = logging.getLogger(__name__)


class SmartDeskRetriever(BaseRetriever):
    domain: str | None = None   # "IT" | "HR" | None (search everything)
    k: int | None = None        # defaults to RETRIEVAL_TOP_K

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        try:
            chunks = kb_search(query, domain=self.domain, k=self.k)["chunks"]
        except Exception as e:  # noqa: BLE001 – vector store down => no docs => escalation, never a guess
            log.exception("Knowledge-base search failed: %s", e)
            return []
        return [Document(page_content=c["text"], metadata={k: v for k, v in c.items() if k != "text"})
                for c in chunks]


def docs_to_chunks(docs: list[Document]) -> list[dict]:
    """Convert Documents back to chunk dicts (text + metadata)."""
    return [{**d.metadata, "text": d.page_content} for d in docs]
