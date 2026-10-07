"""LLM retry (.with_retry) and fallback model (.with_fallbacks) behaviour."""
import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import agents.llm as llm


def timeout():
    return openai.APITimeoutError(request=httpx.Request("POST", "https://x"))


def flaky(name, fail_times, calls):
    def run(_):
        calls.append(name)
        if calls.count(name) <= fail_times:
            raise timeout()
        return AIMessage(content=f"ok from {name}")
    return RunnableLambda(run)


@pytest.fixture(autouse=True)
def retries(monkeypatch):
    monkeypatch.setattr(llm.settings, "LLM_MAX_RETRIES", 3)


def test_retries_transient_errors_then_succeeds():
    calls = []
    out = llm.with_resilience(flaky("primary", 2, calls)).invoke("hi")
    assert out.content == "ok from primary" and calls == ["primary"] * 3


def test_does_not_retry_non_transient_errors():
    calls = []

    def bad(_):
        calls.append(1)
        raise ValueError("bad request")
    with pytest.raises(ValueError):
        llm.with_resilience(RunnableLambda(bad)).invoke("hi")
    assert len(calls) == 1


def test_fallback_model_used_after_retries():
    calls = []
    chain = llm.with_resilience(flaky("primary", 99, calls)).with_fallbacks(
        [llm.with_resilience(flaky("backup", 0, calls))])
    assert chain.invoke("hi").content == "ok from backup"
    assert calls == ["primary"] * 3 + ["backup"]


def test_chat_raises_llm_unavailable(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "get_llm", lambda: llm.with_resilience(flaky("primary", 99, calls)))
    with pytest.raises(llm.LLMUnavailable):
        llm.chat([("human", "hi")])
