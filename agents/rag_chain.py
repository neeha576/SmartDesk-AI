"""LCEL RAG chain: retriever -> confidence gate -> prompt -> LLM -> text.

    chain = build_rag_chain(retriever, llm, system_prompt)
    out = chain.invoke({"query": ..., "question": ..., "history": [...]})
    # out = {"docs": [...], "top_score": float|None, "gate": None|"no_results"|"low_confidence",
    #        "answer": str|None, "llm_error": bool}

Pipeline:
    {"docs": query -> retriever, "question", "history"}         (RunnableParallel)
      -> assign top_score + gate (retrieval_gate)                (RunnablePassthrough.assign)
      -> branch: gate set  -> answer=None   (LLM never called)   (RunnableBranch)
                 otherwise -> answer = prompt | llm | StrOutputParser()
                              (if the LLM still fails after retries/fallback model,
                               answer=None and llm_error is set – the docs are kept)
"""
from __future__ import annotations

import logging
from operator import itemgetter

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (Runnable, RunnableBranch, RunnableLambda,
                                      RunnableParallel, RunnablePassthrough)

from agents import prompts
from rag.confidence import retrieval_gate
from rag.lc_retriever import docs_to_chunks

log = logging.getLogger(__name__)


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        prompts.CHUNK_TEMPLATE.format(doc_id=d.metadata["doc_id"], title=d.metadata.get("title", ""),
                                      text=d.page_content)
        for d in docs)


def _top_score(x: dict) -> float | None:
    return max((d.metadata.get("dense_score", 0.0) for d in x["docs"]), default=None)


def build_prompt(system_prompt: str) -> ChatPromptTemplate:
    # SystemMessage object (not a template string) so braces in the prompt are never parsed
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder("history"),
        ("human", prompts.CONTEXT_TEMPLATE),
    ])


def _on_llm_error(x: dict) -> None:
    log.error("LLM failed after retries/fallback: %s", x.get("error"))
    return None


def build_rag_chain(retriever: Runnable, llm: Runnable, system_prompt: str) -> Runnable:
    generate = (
        {"chunks": itemgetter("docs") | RunnableLambda(format_docs),
         "question": itemgetter("question"),
         "history": itemgetter("history")}
        | build_prompt(system_prompt)
        | llm
        | StrOutputParser()
    ).with_fallbacks([RunnableLambda(_on_llm_error)], exception_key="error")
    return (
        RunnableParallel(docs=itemgetter("query") | retriever,
                         question=itemgetter("question"),
                         history=itemgetter("history"))
        | RunnablePassthrough.assign(top_score=RunnableLambda(_top_score))
        | RunnablePassthrough.assign(gate=lambda x: retrieval_gate(docs_to_chunks(x["docs"]), x["top_score"]))
        | RunnableBranch(
            (lambda x: x["gate"] is not None,
             RunnablePassthrough.assign(answer=lambda _: None, llm_error=lambda _: False)),
            RunnablePassthrough.assign(answer=generate)
            | RunnablePassthrough.assign(llm_error=lambda x: x["answer"] is None),
        )
    ).with_config(run_name="smartdesk_rag")
