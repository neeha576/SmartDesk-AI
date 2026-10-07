"""LangChain chat model with retries and a graceful fallback.

get_llm() returns ChatOpenAI wrapped with:
  .with_retry(...)      – retries transient errors (timeouts, connection errors, rate limits, 5xx)
                          with exponential backoff + jitter, up to LLM_MAX_RETRIES attempts
  .with_fallbacks(...)  – if LLM_FALLBACK_MODEL is set, tries it after the primary gives up
If everything fails the exception propagates; agents catch it and show a friendly message.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from config import settings


class LLMUnavailable(Exception):
    """Raised by chat() when the LLM can't be reached after retries and fallback."""


def transient_errors() -> tuple[type[Exception], ...]:
    import openai
    return (openai.APITimeoutError, openai.APIConnectionError,
            openai.RateLimitError, openai.InternalServerError)


def with_resilience(model: Runnable) -> Runnable:
    """Add retry-with-backoff for transient API errors."""
    return model.with_retry(retry_if_exception_type=transient_errors(),
                            wait_exponential_jitter=True,
                            stop_after_attempt=settings.LLM_MAX_RETRIES)


def _chat_model(model_name: str) -> Runnable:
    provider = settings.llm_provider()
    if provider != "openai":
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Only 'openai' is implemented.")
    from langchain_openai import ChatOpenAI
    settings.required("OPENAI_API_KEY")
    # max_retries=0: .with_retry() owns retrying so attempts are counted in one place
    return ChatOpenAI(model=model_name, temperature=settings.LLM_TEMPERATURE,
                      timeout=settings.LLM_TIMEOUT_SECONDS, max_retries=0)


@lru_cache(maxsize=1)
def get_llm() -> Runnable:
    """The chat model used by every agent (model names come from .env)."""
    llm = with_resilience(_chat_model(settings.llm_model()))
    if fallback := settings.llm_fallback_model():
        llm = llm.with_fallbacks([with_resilience(_chat_model(fallback))])
    return llm


def chat(messages: list[BaseMessage | tuple[str, str]]) -> str:
    """Convenience: invoke the LLM on messages; return text or raise LLMUnavailable."""
    try:
        return (get_llm().invoke(messages).content or "").strip()
    except Exception as e:  # noqa: BLE001
        raise LLMUnavailable(str(e)) from e


@lru_cache(maxsize=8)
def get_structured_llm(schema: type) -> Runnable:
    """LLM that returns an instance of `schema` (pydantic) via OpenAI tool/function calling,
    with the same retry + fallback-model behaviour as get_llm()."""
    llm = with_resilience(_chat_model(settings.llm_model()).with_structured_output(schema))
    if fallback := settings.llm_fallback_model():
        llm = llm.with_fallbacks([with_resilience(_chat_model(fallback).with_structured_output(schema))])
    return llm


_tool_llms: dict[tuple[str, ...], Runnable] = {}


def get_tool_llm(tools) -> Runnable:
    """LLM with `tools` bound via bind_tools (the model may propose tool calls; it never executes
    them), with the same retry + fallback-model behaviour as get_llm().

    Cached by tool name – LangChain tool objects aren't hashable, so they can't be an lru_cache key."""
    tools = list(tools)
    key = tuple(getattr(t, "name", None) or t.__name__ for t in tools)   # tools or pydantic schemas
    if key not in _tool_llms:
        llm = with_resilience(_chat_model(settings.llm_model()).bind_tools(tools))
        if fallback := settings.llm_fallback_model():
            llm = llm.with_fallbacks([with_resilience(_chat_model(fallback).bind_tools(tools))])
        _tool_llms[key] = llm
    return _tool_llms[key]
