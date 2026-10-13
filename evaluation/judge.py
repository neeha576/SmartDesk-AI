"""LLM-as-judge: grades one SmartDesk answer against the reference answer and the retrieved context.

The judge returns a structured verdict (pydantic, via OpenAI structured output):
  correctness   1-5  does the answer match the reference answer's facts?
  faithfulness  1-5  is every claim supported by the retrieved context? (grounding / no hallucination)
  relevance     1-5  does it answer what was asked, without padding?
  hallucination bool any claim that is NOT in the context?

Judge model: EVAL_JUDGE_MODEL in .env (optional). Defaults to LLM_MODEL. A stronger or different model than the
one being graded reduces self-preference bias. Temperature is 0, so grades are repeatable.
"""
from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

JUDGE_PROMPT = """You are a strict evaluator of an enterprise IT/HR help-desk assistant.
Grade the ASSISTANT ANSWER using only the REFERENCE ANSWER and the RETRIEVED CONTEXT given to you.

Scores (integers 1-5):
- correctness: 5 = every key fact of the reference is present and correct; 4 = minor omission, nothing wrong;
  3 = partly correct or missing an important fact; 2 = mostly wrong or misleading; 1 = wrong or no answer.
  Extra correct detail from the context is fine. Different wording is fine.
- faithfulness: 5 = every claim is supported by the retrieved context; 3 = some unsupported detail;
  1 = mostly unsupported. If the context is empty, judge against the reference answer.
- relevance: 5 = directly answers the question, concise; 3 = answers with noticeable padding or drift;
  1 = does not address the question.
- hallucination: true if the answer states any fact (number, name, link, policy rule) that is not in the context
  or the reference answer.
Citations like "Source: IT-001" are expected; do not penalise them. Give a one-sentence reason."""


class JudgeVerdict(BaseModel):
    correctness: int = Field(ge=1, le=5, description="1-5, agreement with the reference answer")
    faithfulness: int = Field(ge=1, le=5, description="1-5, claims supported by the retrieved context")
    relevance: int = Field(ge=1, le=5, description="1-5, answers the question directly")
    hallucination: bool = Field(description="true if any claim is not supported by context or reference")
    reasoning: str = Field(description="one short sentence explaining the grades")


MAX_CONTEXT_CHARS = 6000


def get_judge():
    """Structured-output judge with the app's retry behaviour."""
    from config import settings
    from agents.llm import _chat_model, with_resilience
    model = os.getenv("EVAL_JUDGE_MODEL") or settings.llm_model()
    return with_resilience(_chat_model(model).with_structured_output(JudgeVerdict))


def judge_answer(judge, question: str, reference: str, context: str, answer: str) -> dict:
    """Grade one answer. Returns the verdict as a dict, or {'error': ...} if the judge call fails."""
    human = (f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\n"
             f"RETRIEVED CONTEXT:\n{(context or '(none)')[:MAX_CONTEXT_CHARS]}\n\nASSISTANT ANSWER:\n{answer}")
    try:
        verdict = judge.invoke([SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=human)])
        return verdict.model_dump() if hasattr(verdict, "model_dump") else dict(verdict)
    except Exception as e:  # noqa: BLE001 – one failed grade must not stop the run
        return {"error": str(e)}
