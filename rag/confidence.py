"""Confidence checks that decide when the agent must NOT answer and should offer a ticket instead."""
from __future__ import annotations

from agents.prompts import NO_ANSWER
from config.settings import CONFIDENCE_THRESHOLD

_NO_ANSWER_CORE = "i don't have enough information"


def retrieval_gate(chunks: list[dict], top_score: float | None,
                   threshold: float = CONFIDENCE_THRESHOLD) -> str | None:
    """Checks before calling the LLM. Returns an escalation reason, or None if it's OK to answer."""
    if not chunks:
        return "no_results"                       # no-retrieval fallback
    if top_score is None or top_score < threshold:
        return "low_confidence"                   # retrieval-score threshold
    return None


def answer_gate(answer: str) -> str | None:
    """Checks the LLM's answer (LLM self-assessment). Returns an escalation reason or None."""
    normalized = answer.strip().lower().replace("’", "'")
    if not normalized or normalized.startswith(_NO_ANSWER_CORE) or normalized == NO_ANSWER.lower():
        return "llm_no_answer"
    if _NO_ANSWER_CORE in normalized:
        return "partial_answer"                   # answered part, flagged the rest
    return None
