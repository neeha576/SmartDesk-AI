"""Unit tests for ticket-node helpers (the conversation itself is tested in test_graph.py)."""
import pytest
from pydantic import ValidationError

import agents.ticket_agent as ta


def test_clean_args_overrides_email_and_trims_summary():
    state = {"employee_email": "jane@nmtech.com", "domain": "HR"}
    args = ta._clean_args({"email": "x@evil.com", "summary": "word " * 40, "description": "Details of the issue.",
                           "category": "HR", "priority": "Low"}, state)
    assert args["email"] == "jane@nmtech.com" and len(args["summary"]) <= ta.SUMMARY_MAX


def test_clean_args_defaults_category_and_priority_from_state():
    args = ta._clean_args({"summary": "Payroll question", "description": "Question about my W-2 form."},
                          {"employee_email": "jane@nmtech.com", "domain": "HR"})
    assert args["category"] == "HR" and args["priority"] == "Medium"


def test_clean_args_rejects_invalid_values():
    with pytest.raises(ValidationError):
        ta._clean_args({"summary": "VPN broken", "description": "VPN fails on connect.", "category": "Finance"},
                       {"employee_email": "jane@nmtech.com"})


def test_fallback_args_use_pending_question_and_user_messages():
    state = {"employee_email": "jane@nmtech.com", "domain": "IT",
             "pending_question": "My monitor flickers",
             "messages": [{"role": "user", "content": "My monitor flickers"},
                          {"role": "assistant", "content": "...(yes / no)"},
                          {"role": "user", "content": "yes"},
                          {"role": "user", "content": "jane@nmtech.com"}]}
    args = ta._fallback_args(state)
    assert args["summary"] == "My monitor flickers" and args["category"] == "IT"
    assert args["description"] == "The employee reported: My monitor flickers"


def test_fallback_args_apply_change_request_to_existing_proposal():
    current = {"email": "jane@nmtech.com", "summary": "Monitor flickers", "description": "Flickers.",
               "category": "IT", "priority": "Medium"}
    args = ta._fallback_args({"employee_email": "jane@nmtech.com", "proposed_ticket": {"args": current},
                              "change_request": "Both monitors", "messages": []})
    assert args["description"].endswith("Additional details: Both monitors")


@pytest.mark.parametrize("text,email", [("sure, it's Jane.Doe@NMTech.com", "jane.doe@nmtech.com"),
                                        ("no email here", None)])
def test_email_extraction(text, email):
    assert ta._email_in(text) == email


# --------------------------------------------------------------------------- real LLM wiring (no network)
def _find_tools(o, depth=0):
    if depth > 8 or o is None:
        return None
    kw = getattr(o, "kwargs", None)
    if isinstance(kw, dict) and "tools" in kw:
        return kw["tools"]
    for attr in ("bound", "runnable", "last"):
        if found := _find_tools(getattr(o, attr, None), depth + 1):
            return found
    return None


@pytest.fixture
def llm_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")       # building the client makes no API call


def test_real_ticket_llm_builds_with_create_ticket_tool_bound(llm_env):
    """Regression: get_tool_llm used lru_cache on unhashable StructuredTool -> TypeError -> silent fallback."""
    tools = _find_tools(ta._get_ticket_llm())
    fn = tools[0]["function"]
    assert fn["name"] == "create_ticket"
    assert set(fn["parameters"]["properties"]) == {"email", "summary", "description", "category", "priority"}
    assert ta._get_ticket_llm() is not None                 # second call hits the cache without error


def test_real_router_llm_builds(llm_env):
    from agents.orchestrator import _get_router
    assert _get_router() is not None


def test_fallback_ignores_generic_ticket_requests_and_unrelated_messages():
    state = {"employee_email": "jane@nmtech.com", "pending_question": None, "domain": "HR", "messages": [
        {"role": "user", "content": "maternity leave"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "how to get my id"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "create new ticket"}]}
    args = ta._fallback_args(state)
    assert args["summary"] == "how to get my id"
    assert args["description"] == "The employee reported: how to get my id"


def test_fallback_returns_none_when_no_issue_described():
    state = {"employee_email": "jane@nmtech.com", "messages": [
        {"role": "user", "content": "I want to raise a ticket"}, {"role": "user", "content": "jane@nmtech.com"}]}
    assert ta._fallback_args(state) is None
