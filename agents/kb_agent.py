"""Shared answer pipeline for the IT and HR agents, built on the LCEL chain in agents/rag_chain.py.

    retriever (hybrid Qdrant, domain-filtered) -> confidence gate -> prompt -> ChatOpenAI -> text
      ├─ 0 docs                        -> offer ticket   (no-retrieval fallback; LLM not called)
      ├─ top dense score < threshold   -> offer ticket   (retrieval-score threshold; LLM not called)
      └─ LLM answer (with_retry + with_fallbacks)
            ├─ LLM unavailable         -> friendly fallback listing relevant articles + offer ticket
            ├─ "[ROUTE:HR|IT]"          -> hand over to the other domain agent
            ├─ "I don't have enough…"  -> offer ticket   (LLM self-assessment)
            ├─ partial answer          -> show answer + offer ticket for the rest
            └─ grounded answer         -> respond directly

A ticket is never created here. Every escalation sets needs_escalation=True and ends with a
"(yes / no)" offer; the graph then pauses at the offer_ticket node (interrupt) for the employee's
answer (human-in-the-loop).
"""
from __future__ import annotations

import logging
import re

from agents import prompts
from agents.llm import get_llm
from agents.rag_chain import build_rag_chain
from agents.state import AgentState
from config import settings
from rag.confidence import answer_gate
from rag.lc_retriever import SmartDeskRetriever, docs_to_chunks

log = logging.getLogger(__name__)

_FOLLOW_UP_WORDS = 6  # short messages are treated as follow-ups for retrieval


# ---------------------------------------------------------------------------- building blocks
def _get_retriever(domain: str):
    return SmartDeskRetriever(domain=domain)


def _get_llm():
    return get_llm()


# ---------------------------------------------------------------------------- helpers
def _last_user_message(messages: list[dict]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def _retrieval_query(messages: list[dict]) -> str:
    """Make short follow-ups ("what about on a Mac?") searchable by adding the previous question."""
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    if len(user_msgs) >= 2 and len(user_msgs[-1].split()) < _FOLLOW_UP_WORDS:
        return f"{user_msgs[-2]} {user_msgs[-1]}"
    return user_msgs[-1] if user_msgs else ""


def _history(messages: list[dict]) -> list[tuple[str, str]]:
    """Previous turns (excluding the current question) as (role, content) for MessagesPlaceholder."""
    past = [m for m in messages[:-1] if m["role"] in ("user", "assistant")][-settings.HISTORY_TURNS:]
    return [("human" if m["role"] == "user" else "ai", m["content"]) for m in past]


def _reply(state: AgentState, text: str, **updates) -> AgentState:
    messages = state.get("messages", []) + [{"role": "assistant", "content": text}]
    return {**state, **updates, "messages": messages}


def _offer_ticket(state: AgentState, domain: str, reason: str, question: str,
                  prefix: str = "", **updates) -> AgentState:
    text = prompts.ESCALATION_OFFER.format(
        reason=prompts.ESCALATION_REASON_TEXT[reason].format(domain=domain),
        team=prompts.TEAM[domain])
    if prefix:
        text = f"{prefix}\n\n{text}"
    log.info("Escalation offered (%s, %s): %s", domain, reason, question)
    return _reply(state, text, needs_escalation=True, escalation_reason=reason,
                  pending_question=question, domain=domain, **updates)


def _llm_fallback(state: AgentState, domain: str, question: str, chunks: list[dict], **updates) -> AgentState:
    titles = {c["doc_id"]: c.get("title", c["doc_id"]) for c in chunks if c.get("chunk_type") == "policy"}
    sources = "\n".join(f"• {t} ({d})" for d, t in titles.items()) or \
        "\n".join(f"• {d}" for d in dict.fromkeys(c["doc_id"] for c in chunks))
    text = prompts.LLM_FALLBACK.format(sources=sources, team=prompts.TEAM[domain])
    return _reply(state, text, needs_escalation=True, escalation_reason="llm_unavailable",
                  pending_question=question, domain=domain, **updates)


# ---------------------------------------------------------------------------- main entry
def answer_from_kb(state: AgentState, domain: str, system_prompt: str) -> AgentState:
    """Answer the latest user message from the knowledge base, or offer a ticket."""
    messages = state.get("messages", [])
    question = _last_user_message(messages)
    state = {**state, "reroute_to": None, "needs_escalation": False, "escalation_reason": None}

    chain = build_rag_chain(_get_retriever(domain), _get_llm(), system_prompt)
    inputs = {"query": _retrieval_query(messages), "question": question, "history": _history(messages)}

    out = chain.invoke(inputs)   # retriever and LLM failures are both handled inside the chain
    chunks = docs_to_chunks(out["docs"])
    common = {"retrieved_chunks": chunks, "confidence": out["top_score"],
              "sources": list(dict.fromkeys(c["doc_id"] for c in chunks))}

    # Retrieval gates (no results / below threshold) – the chain skipped the LLM
    if out["gate"]:
        return _offer_ticket(state, domain, out["gate"], question, **common)

    # LLM unavailable after retries + fallback model: list the articles we found, offer a ticket
    if out["llm_error"]:
        return _llm_fallback(state, domain, question, chunks, **common)

    answer = out["answer"]

    # Wrong agent? hand over to the other domain
    if m := re.search(r"\[ROUTE:(IT|HR)\]", answer):
        if m.group(1) != domain:
            return {**state, **common, "reroute_to": m.group(1)}

    # LLM self-assessment
    reason = answer_gate(answer)
    if reason == "llm_no_answer":
        return _offer_ticket(state, domain, reason, question, **common)
    if reason == "partial_answer":
        return _offer_ticket(state, domain, reason, question, prefix=answer, **common)

    # Confident, grounded answer
    return _reply(state, answer, domain=domain, **common)
