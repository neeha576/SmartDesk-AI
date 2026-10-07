"""Orchestrator: decides which agent handles the employee's latest message.

The LLM returns a Pydantic RouteDecision (OpenAI structured output). The prompt lists every
available agent and the routing rules (agents/prompts.py ROUTER_PROMPT). If the LLM is
unavailable, a keyword fallback keeps the help desk working.
"""
from __future__ import annotations

import logging
import re

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from agents import prompts
from agents.llm import get_structured_llm
from agents.state import AgentName, AgentState
from langgraph.types import Send
from config import settings

log = logging.getLogger(__name__)


class SubTask(BaseModel):
    """One independent request inside a multi-part message."""
    agent: AgentName = Field(description="Agent that handles this part")
    request: str = Field(description="This part of the message, rewritten as a self-contained request")


class RouteDecision(BaseModel):
    """Which agent(s) should handle the employee's latest message."""
    agent: AgentName = Field(description="it_agent, hr_agent, ticket_agent, status_agent or smalltalk_agent "
                                         "– for a multi-part message, the agent of the first part")
    reason: str = Field(description="Short reason for the choice (max 15 words).")
    tasks: list[SubTask] = Field(default_factory=list, description=(
        "ONLY when the message contains 2-3 independent requests for different agents (or two unrelated IT/HR "
        "questions): one entry per request, in the order asked. Leave empty for a single request."))


_router_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=prompts.ROUTER_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{message}"),
])


def _get_router():
    """prompt | ChatOpenAI.with_structured_output(RouteDecision), with retry + fallback model."""
    return _router_prompt | get_structured_llm(RouteDecision)


# ---------------------------------------------------------------------------- keyword fallback
_TICKET = re.compile(r"\b(raise|open|create|log|file|submit|new)\b.*\bticket\b|\b(talk|speak) to (a |an )?(human|person|agent|someone)\b", re.I)
_STATUS = re.compile(
    r"\b(status|updates?|progress|news)\b.*\b(tickets?|issues?|requests?)\b"
    r"|\b(tickets?|issues?|requests?)\b.*\b(status|updates?|progress)\b"
    r"|\b(any(one|body)?|did|has|have)\b.*\b(look(ed)?|check(ed)?|work(ed)?|respond(ed)?|repl(y|ied))\b.*\b(tickets?|issues?|requests?)\b"
    r"|\b[A-Z][A-Z0-9]+-\d+\b|\bmy tickets?\b", re.I)
_GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks?( you)?|thank you|bye|goodbye)\b", re.I)
_IT = re.compile(r"\b(password|login|log in|account|locked|mfa|authenticator|vpn|wi-?fi|wifi|network|internet|"
                 r"software|install|license|laptop|computer|monitor|printer|email|outlook|teams|phone|device|"
                 r"keyboard|mouse|headset|it desk|service desk)\b", re.I)
_HR = re.compile(r"\b(pto|leave|vacation|sick|holiday|benefit|insurance|401k|401\(k\)|hsa|pay|payday|salary|"
                 r"paycheck|bonus|w-?4|w-?2|tax|expense|reimburs|stipend|remote|hybrid|wfh|harass|"
                 r"discriminat|conduct|ethics|review|promotion|onboarding|probation|introductory|stock|vesting|hr)\b", re.I)


def keyword_route(text: str) -> RouteDecision:
    """Deterministic fallback used when the router LLM is unavailable."""
    if _TICKET.search(text):
        return RouteDecision(agent="ticket_agent", reason="keyword: ticket request")
    if _STATUS.search(text):
        return RouteDecision(agent="status_agent", reason="keyword: ticket status")
    it, hr = len(_IT.findall(text)), len(_HR.findall(text))
    if it or hr:
        return RouteDecision(agent="it_agent" if it >= hr else "hr_agent", reason="keyword: topic match")
    if _GREETING.search(text):
        return RouteDecision(agent="smalltalk_agent", reason="keyword: greeting")
    return RouteDecision(agent="smalltalk_agent", reason="keyword: no IT/HR topic found")


# ---------------------------------------------------------------------------- graph node
MAX_PARALLEL_TASKS = 3


def route_message(state: AgentState) -> dict:
    """Graph node: pick the agent(s) for the latest message and reset per-turn fields."""
    messages = state.get("messages", [])
    text = messages[-1]["content"] if messages else ""
    history = [("human" if m["role"] == "user" else "ai", m["content"])
               for m in messages[:-1]][-settings.HISTORY_TURNS:]
    try:
        decision = _get_router().invoke({"history": history, "message": text})
    except Exception as e:  # noqa: BLE001 – LLM down: keyword routing keeps SmartDesk usable
        log.warning("Router LLM failed, using keyword routing: %s", e)
        decision = keyword_route(text)
    if state.get("status_handed_back") and decision.agent == "status_agent":
        # the status agent just said this message isn't about tickets – don't send it straight back
        fallback = keyword_route(text)
        decision = fallback if fallback.agent != "status_agent" else RouteDecision(
            agent="smalltalk_agent", reason="handed back by status agent")
    tasks = [t.model_dump() for t in decision.tasks][:MAX_PARALLEL_TASKS]
    if state.get("status_handed_back"):
        tasks = [t for t in tasks if t["agent"] != "status_agent"]
    parallel = len(tasks) >= 2
    log.info("Route -> %s (%s)", [t["agent"] for t in tasks] if parallel else decision.agent, decision.reason)

    domain = {"it_agent": "IT", "hr_agent": "HR"}.get(decision.agent, state.get("domain"))
    return {
        "route": "parallel" if parallel else decision.agent, "route_reason": decision.reason, "domain": domain,
        "route_tasks": tasks if parallel else [], "partial_results": None, "status_query": None,
        "final_answer": None, "cache_hit": False,
        # per-turn resets
        "reroute_to": None, "rerouted": False, "needs_escalation": False, "escalation_reason": None,
        "change_request": None, "email_invalid": False, "ticket_error": None,
        "status_handed_back": False,
    }


def pick_agent(state: AgentState):
    """Conditional edge: the chosen agent's node – or, for a multi-part message, one Send() per part straight
    to that part's agent, so the parts run in PARALLEL; each branch ends at the synthesizer.
    Ticket requests aren't sent (creating a ticket needs the employee) – the synthesizer starts them after."""
    if state.get("route") == "parallel":
        history = state.get("messages", [])[:-1]
        shared = {k: state.get(k) for k in ("employee_email", "last_ticket", "domain")}
        sends = [Send(task["agent"], {"task": task, "index": i, "history": history, **shared})
                 for i, task in enumerate(state["route_tasks"]) if task["agent"] != "ticket_agent"]
        return sends or "synthesizer"
    return state.get("route", "smalltalk_agent")
