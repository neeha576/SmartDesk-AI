"""Tests for the IT/HR LCEL RAG pipeline: every escalation path, LLM fallback, rerouting and HITL.

A fake retriever and fake chat model are plugged into the real chain (agents/rag_chain.py).
"""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import agents.kb_agent as kb
import rag.lc_retriever as lcr
from agents import prompts
from agents.hr_agent import hr_agent
from agents.it_agent import it_agent

GOOD = [Document(page_content="Go to the portal...", metadata={
    "doc_id": "IT-001", "title": "Password Reset", "chunk_type": "policy", "dense_score": 0.71})]


def state(*user_msgs):
    return {"messages": [{"role": "user", "content": m} for m in user_msgs]}


def last(s):
    return s["messages"][-1]["content"]


@pytest.fixture
def search(monkeypatch):
    box = {"docs": GOOD}

    def make(domain):
        box["domain"] = domain

        def run(query):
            box["query"] = query
            return box["docs"]
        return RunnableLambda(run)
    monkeypatch.setattr(kb, "_get_retriever", make)
    return box


@pytest.fixture
def llm(monkeypatch):
    box = {"answer": "Reset it at the portal.\nSource: IT-001", "calls": 0}

    def run(prompt_value):
        box["calls"] += 1
        box["messages"] = prompt_value.to_messages()
        if isinstance(box["answer"], Exception):
            raise box["answer"]
        return AIMessage(content=box["answer"])
    monkeypatch.setattr(kb, "_get_llm", lambda: RunnableLambda(run))
    return box


def test_confident_answer_is_returned_directly(search, llm):
    out = it_agent(state("How do I reset my password?"))
    assert last(out).startswith("Reset it at the portal")
    assert not out["needs_escalation"]
    assert search["domain"] == "IT" and out["confidence"] == 0.71
    assert llm["messages"][0].content == prompts.IT_AGENT_PROMPT
    assert "<context>" in llm["messages"][-1].content and "[IT-001]" in llm["messages"][-1].content


def test_no_docs_escalates_without_calling_llm(search, llm):
    search["docs"] = []
    out = it_agent(state("My monitor flickers"))
    assert out["escalation_reason"] == "no_results"
    assert out["needs_escalation"] and llm["calls"] == 0
    assert "(yes / no)" in last(out)


def test_low_score_escalates_without_calling_llm(search, llm):
    search["docs"] = [Document(page_content="x", metadata={**GOOD[0].metadata, "dense_score": 0.2})]
    out = hr_agent(state("Where can I park?"))
    assert out["escalation_reason"] == "low_confidence" and llm["calls"] == 0
    assert "People Operations" in last(out)


def test_llm_self_assessment_escalates(search, llm):
    llm["answer"] = prompts.NO_ANSWER
    out = it_agent(state("How do I add the printer?"))
    assert out["escalation_reason"] == "llm_no_answer"
    assert prompts.NO_ANSWER not in last(out)  # replaced by the friendly offer


def test_partial_answer_keeps_answer_and_offers_ticket(search, llm):
    llm["answer"] = "Use GlobalProtect.\nI don't have enough information about Linux ARM.\nSource: IT-002"
    out = it_agent(state("VPN on Linux ARM?"))
    assert out["escalation_reason"] == "partial_answer"
    assert last(out).startswith("Use GlobalProtect") and "(yes / no)" in last(out)


def test_llm_failure_falls_back_gracefully(search, llm):
    llm["answer"] = TimeoutError("LLM down")
    out = it_agent(state("How do I reset my password?"))
    assert out["escalation_reason"] == "llm_unavailable"
    assert "Password Reset (IT-001)" in last(out) and out["needs_escalation"]


def test_retriever_failure_returns_no_docs(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("qdrant down")
    monkeypatch.setattr(lcr, "kb_search", boom)
    assert lcr.SmartDeskRetriever(domain="HR").invoke("How much PTO?") == []


def test_retriever_returns_documents_with_scores(monkeypatch):
    monkeypatch.setattr(lcr, "kb_search", lambda q, domain=None, k=None: {
        "chunks": [{"text": "PTO is 15 days", "doc_id": "HR-001", "dense_score": 0.6}], "top_score": 0.6})
    docs = lcr.SmartDeskRetriever(domain="HR").invoke("PTO?")
    assert docs[0].page_content == "PTO is 15 days" and docs[0].metadata["dense_score"] == 0.6


def test_reroute_to_other_domain(search, llm):
    llm["answer"] = "[ROUTE:HR]"
    out = it_agent(state("When is payday?"))
    assert out["reroute_to"] == "HR" and len(out["messages"]) == 1  # no reply added


def test_follow_up_uses_history(search, llm):
    s = {"messages": [{"role": "user", "content": "How do I connect to the VPN?"},
                      {"role": "assistant", "content": "Open GlobalProtect..."},
                      {"role": "user", "content": "And on a Mac?"}]}
    it_agent(s)
    assert search["query"] == "How do I connect to the VPN? And on a Mac?"
    assert [m.content for m in llm["messages"][1:3]] == ["How do I connect to the VPN?", "Open GlobalProtect..."]
