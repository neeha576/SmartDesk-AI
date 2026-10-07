"""Semantic cache (memory/semantic_cache.py): lookups, safety rules and the hook in the IT/HR agents.

Uses an in-memory Qdrant and a tiny fake embedder – no OpenAI calls.
"""
import math

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient

import agents.kb_agent as kb
import memory.semantic_cache as sc
from agents.hr_agent import hr_agent
from agents.it_agent import it_agent

pytestmark = pytest.mark.semantic_cache

# Questions with the same "topic" get almost the same vector; different topics are orthogonal.
TOPICS = {"sick": 0, "password": 1, "vpn": 2, "pto": 3}


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        t = text.lower()
        v = [0.0] * 8
        for word, i in TOPICS.items():
            if word in t:
                v[i] = 1.0
        v[7] = 0.05 * (len(t) % 3)                # small wording differences
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


@pytest.fixture
def cache():
    return sc.SemanticCache(QdrantClient(":memory:"), FakeEmbedder(), collection="test_cache",
                            threshold=0.93, ttl_hours=24)


def test_paraphrase_hits_same_domain_only(cache):
    cache.store("How many sick days do I get?", "HR", "You get 10 sick days.\nSource: HR-002", ["HR-002"])
    hit = cache.lookup("what is my sick days allowance per year", "HR")
    assert hit and hit["answer"].startswith("You get 10 sick days") and hit["sources"] == ["HR-002"]
    assert hit["score"] >= 0.93
    assert cache.lookup("what is my sick days allowance per year", "IT") is None     # never across domains
    assert cache.lookup("How do I reset my password?", "HR") is None                 # different question


def test_empty_or_missing_cache_is_a_miss(cache):
    assert cache.lookup("How many sick days do I get?", "HR") is None


def test_knowledge_base_change_invalidates(cache, monkeypatch):
    cache.store("How many sick days do I get?", "HR", "10 days", ["HR-002"])
    monkeypatch.setattr(sc, "kb_version", lambda: "new-version")
    assert cache.lookup("How many sick days do I get?", "HR") is None


def test_entries_expire(cache, monkeypatch):
    cache.store("How many sick days do I get?", "HR", "10 days", ["HR-002"])
    real = sc.time.time
    monkeypatch.setattr(sc.time, "time", lambda: real() + 25 * 3600)              # TTL is 24 h
    assert cache.lookup("How many sick days do I get?", "HR") is None


def test_same_question_overwrites_instead_of_duplicating(cache):
    cache.store("How many sick days do I get?", "HR", "old", [])
    cache.store("how many sick days do i get", "HR", "new", [])
    assert cache.client.count("test_cache").count == 1
    assert cache.lookup("How many sick days do I get?", "HR")["answer"] == "new"


def test_cache_errors_are_misses(cache):
    class Broken:
        def collection_exists(self, name):
            raise ConnectionError("qdrant down")
    cache.client = Broken()
    assert cache.lookup("How many sick days do I get?", "HR") is None
    cache.store("How many sick days do I get?", "HR", "x", [])                      # no exception


def test_clear(cache):
    cache.store("How many sick days do I get?", "HR", "10 days", [])
    cache.clear()
    assert cache.lookup("How many sick days do I get?", "HR") is None


# ----------------------------------------------------------------------------- inside the IT / HR agents
GOOD = [Document(page_content="Go to the portal...", metadata={
    "doc_id": "IT-001", "title": "Password Reset", "chunk_type": "policy", "dense_score": 0.71})]


@pytest.fixture
def agent(monkeypatch, cache):
    box = {"answer": "Reset it at the portal.\nSource: IT-001", "llm_calls": 0, "searches": 0, "docs": GOOD}

    def make_retriever(domain):
        def run(query):
            box["searches"] += 1
            return box["docs"]
        return RunnableLambda(run)

    def run_llm(prompt_value):
        box["llm_calls"] += 1
        return AIMessage(content=box["answer"])

    monkeypatch.setattr(kb, "_get_retriever", make_retriever)
    monkeypatch.setattr(kb, "_get_llm", lambda: RunnableLambda(run_llm))
    monkeypatch.setattr(kb, "_get_cache", lambda: cache)
    return box


def ask(fn, *msgs):
    return fn({"messages": [{"role": "user", "content": m} for m in msgs]})


def test_second_similar_question_is_served_from_cache(agent):
    first = ask(it_agent, "How do I reset my password?")
    assert first["cache_hit"] is False and agent["llm_calls"] == 1
    second = ask(it_agent, "how can I reset my password")
    assert second["cache_hit"] is True and second["messages"][-1]["content"] == first["messages"][-1]["content"]
    assert agent["llm_calls"] == 1 and agent["searches"] == 1                       # no retrieval, no LLM
    assert second["sources"] == ["IT-001"] and not second["needs_escalation"]


def test_escalations_are_not_cached(agent):
    agent["docs"] = []                                                              # nothing found -> ticket offer
    assert ask(it_agent, "How do I reset my password?")["needs_escalation"]
    agent["docs"] = GOOD
    assert ask(it_agent, "How do I reset my password?")["cache_hit"] is False       # asked the LLM again


def test_no_answer_and_partial_answers_are_not_cached(agent):
    agent["answer"] = "I don't have enough information to answer that."
    ask(it_agent, "How do I reset my password?")
    agent["answer"] = "Reset it at the portal.\nSource: IT-001"
    assert ask(it_agent, "How do I reset my password?")["cache_hit"] is False


def test_follow_ups_skip_the_cache(agent):
    ask(it_agent, "How do I reset my password?")
    out = ask(it_agent, "How do I reset my password?", "and on vpn?")                # short follow-up
    assert out["cache_hit"] is False and agent["llm_calls"] == 2


def test_it_answer_never_served_to_hr(agent):
    ask(it_agent, "How do I reset my password?")
    assert ask(hr_agent, "How do I reset my password?")["cache_hit"] is False
