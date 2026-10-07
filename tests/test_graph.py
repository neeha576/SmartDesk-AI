"""End-to-end tests of the LangGraph orchestrator: routing, interrupts (human-in-the-loop),
bind_tools ticket proposals, session memory and error handling.

Fakes replace the three LLM calls (router, KB answer, ticket proposal) and the retriever; the real
compiled graph, checkpointer, interrupts, create_ticket tool and mock Jira are used.
"""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import agents.kb_agent as kb
import agents.orchestrator as orch
import agents.smalltalk_agent as sm
import agents.status_agent as sa
import agents.ticket_agent as ta
import tools.create_ticket as ct
from agents import prompts
from agents.graph import SmartDeskChat, build_graph
from agents.orchestrator import RouteDecision, SubTask, keyword_route
from tools import ticket_store
from tools.ticketing_client import TicketingUnavailable

PROPOSAL = {"summary": "Monitor flickering for two days", "category": "IT", "priority": "Medium",
            "description": "The employee reports their monitor has flickered for two days.",
            "email": "someone-else@evil.com"}          # LLM-written email must be overridden


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    box = {"route": "it_agent", "docs": [], "answer": "Answer.\nSource: IT-001",
           "proposal": PROPOSAL, "ticket_inputs": [], "router_inputs": []}

    def router(inputs):
        box["router_inputs"].append(inputs)
        if isinstance(box["route"], Exception):
            raise box["route"]
        if isinstance(box["route"], list):                       # multi-part: [(agent, request), ...]
            tasks = [SubTask(agent=a, request=r) for a, r in box["route"]]
            return RouteDecision(agent=tasks[0].agent, reason="test", tasks=tasks)
        return RouteDecision(agent=box["route"], reason="test")

    def ticket_llm(inputs):
        box["ticket_inputs"].append(inputs)
        p = box["proposal"]
        if isinstance(p, Exception):
            raise p
        if p is None:
            return AIMessage(content="What is the ticket about?")
        return AIMessage(content="", tool_calls=[{"name": "create_ticket", "args": dict(p), "id": "call_1"}])

    monkeypatch.setattr(orch, "_get_router", lambda: RunnableLambda(router))
    monkeypatch.setattr(kb, "_get_retriever", lambda d: RunnableLambda(lambda q: box["docs"]))
    monkeypatch.setattr(kb, "_get_llm", lambda: RunnableLambda(lambda pv: AIMessage(content=box["answer"])))
    monkeypatch.setattr(sm, "get_llm", lambda: RunnableLambda(lambda pv: AIMessage(content="Hello! How can I help?")))
    monkeypatch.setattr(ta, "_get_ticket_llm", lambda: RunnableLambda(ticket_llm))
    # real create_ticket tool against the in-memory mock Jira
    monkeypatch.setenv("JIRA_IT_PROJECT_KEY", "IT")
    monkeypatch.setenv("JIRA_HR_PROJECT_KEY", "HR")
    monkeypatch.setattr(ct.settings, "USE_MOCK_TICKETING", True)
    monkeypatch.setattr(ct.settings, "TICKET_DB_PATH", str(tmp_path / "t.db"))
    from tools import mock_ticketing
    monkeypatch.setattr(mock_ticketing, "_mock", None)          # fresh mock Jira per test
    # status agent LLM: off by default (-> no-LLM fallback); agentic tests set a policy
    box["status_policy"] = None

    def build():
        if box["status_policy"] is None:
            raise RuntimeError("status LLM disabled in this test")
        box.setdefault("status_calls", [])
        return sa.build_status_agent(model=PolicyModel(policy=box["status_policy"], box=box))
    monkeypatch.setattr(sa, "_get_status_agent", build)
    return box


from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class PolicyModel(BaseChatModel):
    """Stand-in chat model for create_agent: a policy function decides each response."""
    policy: object
    box: object            # test box (not a list field: pydantic would copy a list)

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "policy"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.box["status_calls"].append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.policy(messages))])


@pytest.fixture
def chat(fakes):
    return SmartDeskChat(build_graph())


GOOD_DOC = [Document(page_content="Reset at the portal", metadata={
    "doc_id": "IT-001", "title": "Password Reset", "chunk_type": "policy", "dense_score": 0.8})]


# ----------------------------------------------------------------------------- routing
def test_kb_answer_routes_to_it_agent(chat, fakes):
    fakes["docs"] = GOOD_DOC
    assert chat.send("How do I reset my password?").startswith("Answer.")
    assert chat.state["route"] == "it_agent" and chat.waiting_for() is None


def test_router_gets_history_and_message(chat, fakes):
    fakes["docs"] = GOOD_DOC
    chat.send("How do I connect to the VPN?")
    chat.send("And on a Mac?")
    inp = fakes["router_inputs"][-1]
    assert inp["message"] == "And on a Mac?" and inp["history"][0] == ("human", "How do I connect to the VPN?")


def test_smalltalk_route(chat, fakes):
    fakes["route"] = "smalltalk_agent"
    assert chat.send("hi") == "Hello! How can I help?"


def test_reroute_it_question_to_hr(chat, fakes, monkeypatch):
    fakes["docs"] = GOOD_DOC
    answers = iter(["[ROUTE:HR]", "Payday is every other Friday.\nSource: PAY-001"])
    monkeypatch.setattr(kb, "_get_llm", lambda: RunnableLambda(lambda pv: AIMessage(content=next(answers))))
    assert chat.send("When is payday?").startswith("Payday is every other Friday")
    assert chat.state["domain"] == "HR"


def test_router_llm_failure_uses_keywords(chat, fakes):
    fakes["route"] = TimeoutError("down")
    fakes["docs"] = GOOD_DOC
    chat.send("I forgot my password")
    assert chat.state["route"] == "it_agent"


@pytest.mark.parametrize("text,agent", [
    ("I want to raise a ticket", "ticket_agent"), ("any update on IT-42?", "status_agent"),
    ("What is the status of my tickets?", "status_agent"), ("Did anyone look at my VPN issue?", "status_agent"),
    ("any news on my request", "status_agent"), ("is my ticket still open", "status_agent"),
    ("my vpn keeps dropping", "it_agent"), ("how much PTO do I get", "hr_agent"),
    ("thanks!", "smalltalk_agent"), ("what's the weather", "smalltalk_agent")])
def test_keyword_route(text, agent):
    assert keyword_route(text).agent == agent


# ----------------------------------------------------------------------------- escalation + interrupts
def test_full_flow_flow_b(chat, fakes):
    """Capstone Flow B: no KB answer -> offer -> yes -> email -> summary -> yes -> ticket ID."""
    reply = chat.send("My monitor has been flickering for two days.")
    assert "(yes / no)" in reply and chat.waiting_for() == "ticket_offer"

    assert chat.send("yes") == prompts.TICKET_ASK_EMAIL and chat.waiting_for() == "email"

    reply = chat.send("jane.doe@nmtech.com")
    assert chat.waiting_for() == "ticket_confirmation"
    for part in ("Monitor flickering for two days", "IT Support", "Medium", "jane.doe@nmtech.com", "Shall I go ahead?"):
        assert part in reply
    assert ticket_store.tickets_for_email("jane.doe@nmtech.com") == []   # nothing created yet

    reply = chat.send("yes")
    key = chat.state["last_ticket"]["key"]
    assert key.startswith("IT-") and key in reply and "browse/" + key in reply
    assert chat.waiting_for() is None
    assert ticket_store.tickets_for_email("jane.doe@nmtech.com")[0]["ticket_key"] == key
    # the LLM's proposed tool call is what gets executed – with the employee's real email
    assert chat.state["employee_email"] == "jane.doe@nmtech.com"
    assert "evil.com" not in str(ticket_store.tickets_for_email("jane.doe@nmtech.com"))
    # the ticket LLM saw the unanswered question and the email
    assert fakes["ticket_inputs"][0]["pending_question"].startswith("My monitor")
    assert fakes["ticket_inputs"][0]["email"] == "jane.doe@nmtech.com"


def test_email_remembered_for_second_ticket(chat, fakes):
    chat.send("My monitor flickers"); chat.send("yes"); chat.send("jane@nmtech.com"); chat.send("yes")
    reply = chat.send("My keyboard is broken")
    assert "(yes / no)" in reply
    reply = chat.send("yes")
    assert chat.waiting_for() == "ticket_confirmation" and "jane@nmtech.com" in reply   # no email question


def test_offer_declined(chat, fakes):
    chat.send("My monitor flickers")
    assert chat.send("no thanks") == prompts.ESCALATION_DECLINED
    assert chat.waiting_for() is None and fakes["ticket_inputs"] == []


def test_offer_unclear_reply_asks_again(chat, fakes):
    chat.send("My monitor flickers")
    assert chat.send("hmm maybe") == prompts.ESCALATION_UNCLEAR
    assert chat.waiting_for() == "ticket_offer"
    assert chat.send("yes") == prompts.TICKET_ASK_EMAIL


def test_new_question_during_offer_is_routed_not_treated_as_unclear(chat, fakes):
    chat.send("My monitor flickers")
    assert chat.waiting_for() == "ticket_offer"
    fakes["route"], fakes["docs"] = "hr_agent", GOOD_DOC
    reply = chat.send("whats the maternity policy")
    assert reply.startswith("Answer.") and chat.waiting_for() is None
    assert fakes["router_inputs"][-1]["message"] == "whats the maternity policy"
    assert chat.state["route"] == "hr_agent" and not chat.state.get("needs_escalation")


def test_invalid_email_reasked(chat, fakes):
    chat.send("My monitor flickers"); chat.send("yes")
    assert chat.send("jane at nmtech") == prompts.TICKET_BAD_EMAIL
    assert chat.waiting_for() == "email"


def test_cancel_at_confirmation_creates_nothing(chat, fakes):
    chat.send("My monitor flickers"); chat.send("yes"); chat.send("jane@nmtech.com")
    assert chat.send("no") == prompts.TICKET_CANCELLED
    assert ticket_store.tickets_for_email("jane@nmtech.com") == []
    assert chat.state["proposed_ticket"] is None


def test_edit_redrafts_with_change_request(chat, fakes):
    chat.send("My monitor flickers"); chat.send("yes"); chat.send("jane@nmtech.com")
    fakes["proposal"] = {**PROPOSAL, "priority": "High", "description": PROPOSAL["description"] + " Both monitors."}
    reply = chat.send("Both monitors are affected, make it high priority")
    assert "Existing proposal" in fakes["ticket_inputs"][-1]["revision"]
    assert "Both monitors are affected" in fakes["ticket_inputs"][-1]["revision"]
    assert chat.waiting_for() == "ticket_confirmation" and "High" in reply
    chat.send("yes")
    assert ticket_store.tickets_for_email("jane@nmtech.com")[0]["priority"] == "High"


def test_yes_with_changes_is_an_edit(chat, fakes):
    chat.send("My monitor flickers"); chat.send("yes"); chat.send("jane@nmtech.com")
    chat.send("yes but mention it started after the Windows update")
    assert chat.waiting_for() == "ticket_confirmation"
    assert ticket_store.tickets_for_email("jane@nmtech.com") == []


def test_direct_ticket_request_asks_for_details(fakes):
    fakes["route"] = "ticket_agent"
    fakes["proposal"] = None                                     # LLM doesn't call the tool: issue unknown
    chat2 = SmartDeskChat(build_graph())
    assert chat2.send("I want to raise a ticket, my email is jane@nmtech.com") == prompts.TICKET_ASK_DETAILS
    assert chat2.waiting_for() == "ticket_details"
    fakes["proposal"] = PROPOSAL
    reply = chat2.send("My laptop battery drains in an hour")
    assert chat2.waiting_for() == "ticket_confirmation" and "Shall I go ahead?" in reply
    assert chat2.state["pending_question"] == "My laptop battery drains in an hour"


def test_jira_down_then_try_again(chat, fakes, monkeypatch):
    chat.send("My monitor flickers"); chat.send("yes"); chat.send("jane@nmtech.com")
    calls = {"n": 0}
    real = ta._run_create_tool

    def flaky(args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TicketingUnavailable("down")
        return real(args)
    monkeypatch.setattr(ta, "_run_create_tool", flaky)
    assert chat.send("yes") == prompts.TICKET_UNAVAILABLE
    assert chat.waiting_for() == "ticket_confirmation"
    reply = chat.send("try again")
    assert chat.state["last_ticket"]["key"] in reply and calls["n"] == 2


def test_ticket_llm_failure_uses_fallback_proposal(chat, fakes):
    fakes["proposal"] = TimeoutError("LLM down")
    chat.send("My monitor has been flickering."); chat.send("yes"); reply = chat.send("jane@nmtech.com")
    assert chat.waiting_for() == "ticket_confirmation"
    assert chat.state["proposed_ticket"]["id"] == "fallback"
    assert "My monitor has been flickering." in reply


def test_invalid_llm_arguments_fall_back(chat, fakes):
    fakes["proposal"] = {**PROPOSAL, "category": "Finance"}
    chat.send("My monitor flickers"); chat.send("yes"); chat.send("jane@nmtech.com")
    assert chat.state["proposed_ticket"]["args"]["category"] == "IT"


def test_separate_threads_are_isolated(fakes):
    g = build_graph()
    a, b = SmartDeskChat(g, "a"), SmartDeskChat(g, "b")
    a.send("My monitor flickers"); a.send("yes"); a.send("jane@nmtech.com")
    assert b.state.get("employee_email") is None and b.waiting_for() is None


def test_graph_structure():
    g = build_graph()
    main = set(g.get_graph().nodes)
    assert {"router", "it_agent", "hr_agent", "ticket_agent", "status_agent", "smalltalk_agent",
            "offer_ticket"} <= main
    # the ticket flow is a subgraph: its nodes are not in the main graph, only inside "ticket_agent"
    assert not {"draft_ticket", "confirm_ticket", "create_ticket"} & main
    xray = set(g.get_graph(xray=True).nodes)
    assert {f"ticket_agent:{n}" for n in ("collect_email", "draft_ticket", "ask_details",
                                          "confirm_ticket", "create_ticket")} <= xray


def test_messages_are_not_duplicated_by_the_subgraph(chat, fakes):
    chat.send("My monitor flickers"); chat.send("yes"); chat.send("jane@nmtech.com"); chat.send("yes")
    msgs = chat.state["messages"]
    ids = [m["id"] for m in msgs]
    assert len(ids) == len(set(ids))                                   # every message stored once
    assert [m["content"] for m in msgs].count("My monitor flickers") == 1
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 4      # strict alternation, nothing doubled


def test_ticket_subgraph_runs_standalone(fakes):
    """The ticket subgraph is a complete graph on its own: email -> draft -> confirm -> create."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    from agents.ticket_graph import build_ticket_graph
    sub = build_ticket_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "sub"}}
    out = sub.invoke({"messages": [{"role": "user", "content": "My monitor flickers"}],
                      "pending_question": "My monitor flickers", "domain": "IT"}, cfg)
    assert out["__interrupt__"][0].value["type"] == "email"
    out = sub.invoke(Command(resume="jane@nmtech.com"), cfg)
    assert out["__interrupt__"][0].value["type"] == "ticket_confirmation"
    out = sub.invoke(Command(resume="yes"), cfg)
    assert out["last_ticket"]["key"].startswith("IT-") and "__interrupt__" not in out


# ----------------------------------------------------------------------------- ticket status (read)
# Tests below without an agentic policy exercise the deterministic fallback (status LLM unavailable).
def _raise(email, summary, category="IT"):
    return ct.create_ticket(email=email, summary=summary, description=f"The employee reports: {summary}.",
                            category=category, priority="Medium")


def _mock():
    from tools.mock_ticketing import get_mock_client
    return get_mock_client()


def test_status_asks_email_then_reports_none_found(chat, fakes):
    fakes["route"] = "status_agent"
    assert chat.send("What is the status of my ticket?") == prompts.STATUS_ASK_EMAIL
    assert chat.waiting_for() == "status_email"
    reply = chat.send("jane@nmtech.com")
    assert reply == prompts.STATUS_NONE.format(email="jane@nmtech.com") and chat.waiting_for() is None


def test_status_single_ticket_shows_details_and_comments(chat, fakes):
    t = _raise("jane@nmtech.com", "Monitor flickering for two days")
    _mock().set_status(t["key"], "In Progress")
    _mock().add_comment(t["key"], "Replacement monitor ordered, delivery by Thursday.", author="Sam (IT)")
    fakes["route"] = "status_agent"
    chat.send("any updates on my monitor ticket?")
    reply = chat.send("jane@nmtech.com")
    assert t["key"] in reply and "In Progress" in reply
    assert "Sam (IT)" in reply and "Replacement monitor ordered" in reply
    assert chat.waiting_for() is None


def test_status_no_comments_message(chat, fakes):
    _raise("jane@nmtech.com", "VPN keeps dropping")
    fakes["route"] = "status_agent"
    chat.send("status of my ticket")
    reply = chat.send("jane@nmtech.com")
    assert "No updates from the support team yet." in reply and "To Do (Open)" in reply


def test_status_multiple_tickets_list_then_choose_by_number(chat, fakes):
    a = _raise("jane@nmtech.com", "VPN keeps dropping")
    b = _raise("jane@nmtech.com", "Need a W-2 copy", "HR")
    fakes["route"] = "status_agent"
    chat.send("What is the status of my tickets?")
    reply = chat.send("jane@nmtech.com")
    assert chat.waiting_for() == "choose_ticket"
    assert a["key"] in reply and b["key"] in reply and "Reply with a ticket key" in reply
    import re
    listed = re.findall(r"\d\. \*\*([A-Z]+-\d+)\*\*", reply)          # keys in the order shown
    reply = chat.send("2")
    assert reply.startswith(f"**{listed[1]}") and chat.waiting_for() is None


def test_status_choose_by_key_and_by_words(chat, fakes):
    a = _raise("jane@nmtech.com", "VPN keeps dropping")
    b = _raise("jane@nmtech.com", "Need a W-2 copy", "HR")
    fakes["route"] = "status_agent"
    chat.send("ticket status please"); chat.send("jane@nmtech.com")
    assert b["key"] in chat.send(b["key"].lower())
    chat.send("ticket status please")                                 # email remembered: straight to the list
    assert chat.waiting_for() == "choose_ticket"
    assert a["key"] in chat.send("the vpn one")


def test_status_auto_picks_ticket_named_in_question(chat, fakes):
    _raise("jane@nmtech.com", "VPN keeps dropping")
    b = _raise("jane@nmtech.com", "Need a W-2 copy", "HR")
    fakes["route"] = "status_agent"
    chat.send(f"what's the status of {b['key']}?")
    reply = chat.send("jane@nmtech.com")
    assert b["key"] in reply and chat.waiting_for() is None             # no list, no question


def test_status_auto_picks_by_topic(chat, fakes):
    a = _raise("jane@nmtech.com", "VPN keeps dropping")
    _raise("jane@nmtech.com", "Need a W-2 copy", "HR")
    fakes["route"] = "status_agent"
    reply = chat.send("Did anyone look at my VPN issue? my email is jane@nmtech.com")
    assert a["key"] in reply and chat.waiting_for() is None                    # no list, straight to details


def test_status_never_shows_someone_elses_ticket(chat, fakes):
    t = _raise("jane@nmtech.com", "VPN keeps dropping")
    fakes["route"] = "status_agent"
    chat.send(f"status of {t['key']}")
    reply = chat.send("bob@nmtech.com")
    assert reply == prompts.STATUS_NONE.format(email="bob@nmtech.com") and t["key"] not in reply


def test_status_unclear_choice_reasked_and_new_question_routed(chat, fakes):
    _raise("jane@nmtech.com", "VPN keeps dropping"); _raise("jane@nmtech.com", "Need a W-2 copy", "HR")
    fakes["route"] = "status_agent"
    chat.send("my tickets?"); chat.send("jane@nmtech.com")
    assert "Your tickets" in chat.send("hmm")                         # list shown again
    assert chat.waiting_for() == "choose_ticket"
    fakes["route"], fakes["docs"] = "it_agent", GOOD_DOC
    assert chat.send("actually how do I reset my password?").startswith("Answer.")
    assert chat.waiting_for() is None


def test_status_jira_unreachable_is_polite(chat, fakes, monkeypatch):
    import tools.get_ticket_status as gst

    def down(email, client=None):
        raise TicketingUnavailable("down")
    monkeypatch.setattr(gst, "get_tickets_by_email", down)
    fakes["route"] = "status_agent"
    chat.send("status of my ticket")
    assert chat.send("jane@nmtech.com") == prompts.STATUS_UNAVAILABLE and chat.waiting_for() is None


def test_ticket_created_then_status_uses_remembered_email(chat, fakes):
    chat.send("My monitor flickers"); chat.send("yes"); chat.send("jane@nmtech.com"); chat.send("yes")
    key = chat.state["last_ticket"]["key"]
    fakes["route"] = "status_agent"
    reply = chat.send("any update on my ticket?")
    assert key in reply and prompts.STATUS_ASK_EMAIL not in reply      # no email question


# ----------------------------------------------------------------------------- status agent (create_agent + tools)
import json as _json  # noqa: E402

from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402


def _call(name, args=None, i=[0]):
    i[0] += 1
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": f"call_{i[0]}"}])


def scripted_status_llm(other_key=None, inject_email=None):
    """Plays the LLM inside create_agent: uses the tools the way STATUS_AGENT_PROMPT instructs."""
    def policy(msgs):
        last = msgs[-1]
        text = last.content if isinstance(last, HumanMessage) else ""
        if isinstance(last, ToolMessage) and last.content.startswith("Employee replied:"):
            text = last.content.split(":", 1)[1]
        if text:
            if "password" in text.lower():
                return _call("end_status_check", {"reason": "new IT question"})
            return _call("get_my_tickets", {"email": inject_email} if inject_email else {})
        if last.content == "NOT_YOUR_TICKET":
            return AIMessage(content="I couldn't find that ticket among yours.")
        if last.content.startswith(("ERROR:", "EMAIL_NOT_PROVIDED")):
            return AIMessage(content="Sorry, I couldn't reach the ticketing system. Please try again in a few minutes."
                             if last.content.startswith("ERROR:") else "I need your work email to look up tickets.")
        data = _json.loads(last.content)
        if isinstance(data, dict):                                   # details
            comments = " | ".join(f"{c['author']}: {c['text']}" for c in data["comments"]) or "in the queue"
            return AIMessage(content=f"{data['key']} is {data['status']}. {comments}")
        if not data:
            return AIMessage(content="No tickets found.")
        if other_key:
            return _call("get_ticket_details", {"ticket_key": other_key})
        asked = [m for m in msgs if isinstance(m, ToolMessage) and m.content.startswith("Employee replied:")]
        query = asked[-1].content.split(":", 1)[1] if asked else next(m.content for m in msgs if isinstance(m, HumanMessage))
        pick = sa._match(query, data) if len(data) > 1 else data[0]
        if pick:
            return _call("get_ticket_details", {"ticket_key": pick["key"]})
        listing = "\n".join(f"{n}. {t['key']} – {t['summary']} – {t['status']}" for n, t in enumerate(data, 1))
        return _call("ask_employee", {"message": f"Which one?\n{listing}"})
    return policy


def _tool_results(fakes):
    return [m.content for m in fakes["status_calls"][-1] if isinstance(m, ToolMessage)]


def test_status_tools_hide_email_from_the_llm():
    schemas = {t.name: t.tool_call_schema.model_json_schema() for t in sa.STATUS_TOOLS}
    assert set(schemas) == {"get_my_tickets", "get_ticket_details", "ask_employee", "end_status_check"}
    assert schemas["get_my_tickets"].get("properties", {}) == {}
    assert set(schemas["get_ticket_details"]["properties"]) == {"ticket_key"}
    for sch in schemas.values():                                   # injected args never reach the LLM
        assert not {"email", "employee_email", "state", "tool_call_id"} & set(sch.get("properties", {}))


def test_status_agent_single_ticket(chat, fakes):
    t = _raise("jane@nmtech.com", "Monitor flickering for two days")
    _mock().add_comment(t["key"], "Replacement ordered", author="Sam (IT)")
    fakes["route"], fakes["status_policy"] = "status_agent", scripted_status_llm()
    reply = chat.send("any update on my ticket? my email is jane@nmtech.com")
    assert reply == f"{t['key']} is To Do. Sam (IT): Replacement ordered"
    assert chat.waiting_for() is None
    assert len(fakes["status_calls"]) == 3                             # get_my_tickets -> get_ticket_details -> answer
    assert _json.loads(_tool_results(fakes)[0])[0]["key"] == t["key"]


def test_status_agent_asks_email_inside_tool(chat, fakes):
    t = _raise("jane@nmtech.com", "VPN keeps dropping")
    fakes["route"], fakes["status_policy"] = "status_agent", scripted_status_llm()
    assert chat.send("what's the status of my ticket?") == prompts.STATUS_ASK_EMAIL
    assert chat.waiting_for() == "status_email"
    reply = chat.send("jane@nmtech.com")
    assert reply.startswith(t["key"]) and chat.state["employee_email"] == "jane@nmtech.com"
    assert len(fakes["status_calls"]) == 3                             # the LLM was NOT re-run on resume
    msgs = [m["content"] for m in chat.state["messages"]]
    assert prompts.STATUS_ASK_EMAIL in msgs and "jane@nmtech.com" in msgs  # email Q&A kept in the chat


def test_status_agent_asks_which_ticket_then_details(chat, fakes):
    a = _raise("jane@nmtech.com", "VPN keeps dropping")
    _raise("jane@nmtech.com", "Need a W-2 copy", "HR")
    fakes["route"], fakes["status_policy"] = "status_agent", scripted_status_llm()
    reply = chat.send("what's happening with my tickets? jane@nmtech.com")
    assert chat.waiting_for() == "choose_ticket" and reply.startswith("Which one?")
    reply = chat.send("the vpn one")
    assert reply.startswith(f"{a['key']} is To Do") and chat.waiting_for() is None
    roles = [m["role"] for m in chat.state["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]          # question + choice + answer, no duplicates


def test_status_agent_blocks_other_employees_ticket(chat, fakes, monkeypatch):
    bobs = _raise("bob@nmtech.com", "Payroll error")
    _raise("jane@nmtech.com", "VPN keeps dropping")
    looked_up = []
    real = sa.status_tool.get_ticket_details
    monkeypatch.setattr(sa.status_tool, "get_ticket_details", lambda k, **kw: (looked_up.append(k), real(k))[1])
    fakes["route"], fakes["status_policy"] = "status_agent", scripted_status_llm(other_key=bobs["key"])
    reply = chat.send(f"show me {bobs['key']}, my email is jane@nmtech.com")
    assert reply == "I couldn't find that ticket among yours."
    assert looked_up == []                                            # Jira never queried for Bob's ticket
    assert "NOT_YOUR_TICKET" in _tool_results(fakes) and "Payroll error" not in str(_tool_results(fakes))


def test_status_agent_ignores_email_written_by_llm(chat, fakes):
    _raise("bob@nmtech.com", "Payroll error")
    mine = _raise("jane@nmtech.com", "VPN keeps dropping")
    fakes["route"], fakes["status_policy"] = "status_agent", scripted_status_llm(inject_email="bob@nmtech.com")
    reply = chat.send("status please, jane@nmtech.com")
    assert "Payroll" not in reply and "bob@" not in str(_tool_results(fakes))
    assert reply.startswith(mine["key"])


def test_status_agent_hands_back_unrelated_message(chat, fakes):
    _raise("jane@nmtech.com", "VPN keeps dropping"); _raise("jane@nmtech.com", "Need a W-2 copy", "HR")
    fakes["route"], fakes["status_policy"] = "status_agent", scripted_status_llm()
    fakes["docs"] = GOOD_DOC
    chat.send("my tickets please jane@nmtech.com")
    assert chat.waiting_for() == "choose_ticket"
    # router LLM (fake) would keep saying status_agent – the hand-back guard must send it elsewhere
    reply = chat.send("actually, how do I reset my password?")
    assert reply.startswith("Answer.") and chat.state["route"] == "it_agent" and chat.waiting_for() is None


def test_status_agent_no_tickets(chat, fakes):
    fakes["route"], fakes["status_policy"] = "status_agent", scripted_status_llm()
    assert chat.send("status of my ticket, jane@nmtech.com") == "No tickets found."


def test_status_agent_jira_unreachable_is_explained_by_llm(chat, fakes, monkeypatch):
    def down(email, client=None):
        raise TicketingUnavailable("down")
    monkeypatch.setattr(sa.status_tool, "get_tickets_by_email", down)
    fakes["route"], fakes["status_policy"] = "status_agent", scripted_status_llm()
    reply = chat.send("status of my ticket, jane@nmtech.com")
    assert reply == "Sorry, I couldn't reach the ticketing system. Please try again in a few minutes."
    assert _tool_results(fakes) == ["ERROR: the ticketing system (Jira) is unreachable right now."]


def test_status_agent_call_limit_falls_back(chat, fakes):
    t = _raise("jane@nmtech.com", "VPN keeps dropping")
    fakes["route"], fakes["status_policy"] = "status_agent", lambda msgs: _call("get_my_tickets")
    reply = chat.send("status please jane@nmtech.com")
    assert len(fakes["status_calls"]) == sa.MAX_MODEL_CALLS and t["key"] in reply   # no-LLM fallback answered


# ----------------------------------------------------------------------------- multi-part: Send() + synthesizer
import time as _time  # noqa: E402

import agents.synthesizer as syn  # noqa: E402


@pytest.fixture
def synth(monkeypatch):
    """Stand-in for the synthesizer's llm.invoke([SystemMessage, HumanMessage])."""
    box = {"inputs": [], "fail": False}

    def run(messages):
        system, human = messages
        assert system.content == prompts.SYNTHESIZER_PROMPT
        box["inputs"].append(human.content)
        if box["fail"]:
            raise TimeoutError("LLM down")
        return AIMessage(content="SYN|" + human.content)
    monkeypatch.setattr(syn, "get_llm", lambda: RunnableLambda(run))
    return box


def test_parallel_smalltalk_and_it_question(chat, fakes, synth):
    fakes["route"] = [("smalltalk_agent", "hi"), ("it_agent", "How do I reset my password?")]
    fakes["docs"] = GOOD_DOC
    before = len(chat.state.get("messages", []))
    reply = chat.send("hi! how do I reset my password?")
    assert reply.startswith("SYN|") and chat.state["route"] == "parallel" and chat.waiting_for() is None
    sent = synth["inputs"][0]
    assert sent.startswith("Employee query: hi! how do I reset my password?\n\nAgent outputs:\n")
    assert sent.index("[Smalltalk Agent] (request: hi)") < sent.index("[IT Agent] (request: How do I reset my password?)")
    assert sent.index("Hello! How can I help?") < sent.index("Answer.")   # order of the message kept
    assert len(chat.state["messages"]) == before + 2                     # one user + ONE combined reply
    assert chat.state["final_answer"] == reply
    assert len(synth["inputs"]) == 1                                     # synthesizer ran exactly once


def test_parallel_agents_feed_synthesizer_directly(chat):
    """No extra 'parallel' node: Send() targets the real agents, and they hand over to the synthesizer."""
    g = chat.graph.get_graph()
    assert "parallel_agent" not in g.nodes
    into_synth = {e.source for e in g.edges if e.target == "synthesizer"}
    assert {"it_agent", "hr_agent", "smalltalk_agent", "status_agent"} <= into_synth


def test_parallel_branches_run_concurrently(chat, fakes, synth, monkeypatch):
    def slow_kb(pv):
        _time.sleep(0.5)
        return AIMessage(content="Answer.\nSource: IT-001")
    monkeypatch.setattr(kb, "_get_llm", lambda: RunnableLambda(slow_kb))
    fakes["docs"] = GOOD_DOC
    fakes["route"] = [("it_agent", "How do I reset my password?"), ("hr_agent", "How many sick days do I get?"),
                      ("it_agent", "How do I connect to the VPN?")]
    start = _time.perf_counter()
    chat.send("password reset? sick days? VPN?")
    elapsed = _time.perf_counter() - start
    assert len(chat.state["partial_results"]) == 3 and len(synth["inputs"]) == 1
    assert elapsed < 1.2, f"branches ran sequentially ({elapsed:.2f}s for 3 x 0.5s)"


def test_parallel_kb_answer_plus_ticket_status(chat, fakes, synth):
    t = _raise("jane@nmtech.com", "VPN keeps dropping")
    _mock().add_comment(t["key"], "Engineer assigned", author="Sam (IT)")
    chat.graph.update_state(chat.config, {"employee_email": "jane@nmtech.com"})
    fakes["docs"] = GOOD_DOC
    fakes["route"] = [("hr_agent", "How many sick days do I get?"), ("status_agent", "any update on my VPN ticket?")]
    reply = chat.send("how many sick days do I get, and any update on my VPN ticket?")
    assert "Answer." in reply and t["key"] in reply and "Engineer assigned" in reply
    assert chat.waiting_for() is None


def test_parallel_unanswered_part_offers_ticket_then_flow(chat, fakes, synth):
    fakes["route"] = [("smalltalk_agent", "thanks"), ("it_agent", "My monitor has been flickering")]
    fakes["docs"] = []                                                   # KB has nothing on monitors
    reply = chat.send("thanks! also my monitor has been flickering")
    assert "(yes / no)" in reply and "My monitor has been flickering" in reply
    assert chat.waiting_for() == "ticket_offer"
    assert chat.send("yes") == prompts.TICKET_ASK_EMAIL
    chat.send("jane@nmtech.com")
    assert fakes["ticket_inputs"][0]["pending_question"] == "My monitor has been flickering"


def test_parallel_status_needing_email_asks_after_synthesis(chat, fakes, synth):
    t = _raise("jane@nmtech.com", "VPN keeps dropping")
    fakes["docs"] = GOOD_DOC
    fakes["route"] = [("it_agent", "How do I reset my password?"), ("status_agent", "status of my VPN ticket")]
    reply = chat.send("how do I reset my password and what's the status of my VPN ticket?")
    assert reply == prompts.STATUS_ASK_EMAIL and chat.waiting_for() == "status_email"
    combined = chat.state["messages"][-1]["content"]   # synthesized answer; the email question is the paused prompt
    assert combined.startswith("SYN|") and prompts.PARALLEL_STATUS_NEEDS_EMAIL in combined
    reply = chat.send("jane@nmtech.com")
    assert t["key"] in reply and chat.waiting_for() is None


def test_parallel_ticket_request_starts_ticket_flow(chat, fakes, synth):
    fakes["docs"] = GOOD_DOC
    fakes["route"] = [("it_agent", "How do I reset my password?"), ("ticket_agent", "raise a ticket for my broken keyboard")]
    reply = chat.send("how do I reset my password? also raise a ticket for my broken keyboard")
    assert reply == prompts.TICKET_ASK_EMAIL and chat.waiting_for() == "email"
    assert "Answer." in chat.state["messages"][-1]["content"]   # synthesized answer before the email question
    assert chat.state["pending_question"] == "raise a ticket for my broken keyboard"


def test_synthesizer_llm_failure_joins_answers(chat, fakes, synth):
    synth["fail"] = True
    fakes["docs"] = GOOD_DOC
    fakes["route"] = [("smalltalk_agent", "hi"), ("it_agent", "How do I reset my password?")]
    assert chat.send("hi! password reset?") == "Hello! How can I help?\n\nAnswer.\nSource: IT-001"


def test_one_failing_branch_does_not_sink_the_others(chat, fakes, synth, monkeypatch):
    def boom(email, request):
        raise RuntimeError("bug")
    monkeypatch.setattr(sa, "status_summary", boom)
    chat.graph.update_state(chat.config, {"employee_email": "jane@nmtech.com"})
    fakes["docs"] = GOOD_DOC
    fakes["route"] = [("it_agent", "How do I reset my password?"), ("status_agent", "my tickets")]
    reply = chat.send("password reset? and my tickets")
    assert "Answer." in reply and "couldn't handle “my tickets”" in reply


def test_partial_results_reset_each_turn(chat, fakes, synth):
    fakes["docs"] = GOOD_DOC
    fakes["route"] = [("smalltalk_agent", "hi"), ("it_agent", "How do I reset my password?")]
    chat.send("hi! password?")
    fakes["route"] = [("smalltalk_agent", "thanks"), ("hr_agent", "How many sick days?")]
    chat.send("thanks! sick days?")
    assert [p["request"] for p in chat.state["partial_results"]] == ["thanks", "How many sick days?"]


def test_single_task_list_uses_normal_path(chat, fakes, synth):
    fakes["docs"] = GOOD_DOC
    fakes["route"] = [("it_agent", "How do I reset my password?")]
    assert chat.send("How do I reset my password?").startswith("Answer.")
    assert chat.state["route"] == "it_agent" and synth["inputs"] == []
