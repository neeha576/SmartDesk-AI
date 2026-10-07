"""Ticket agent – nodes of the ticket-creation subgraph (agents/ticket_graph.py) + the offer node.

offer_ticket lives in the main graph (it can hand a new question back to the router); the other
nodes form the ticket subgraph, added to the main graph as the single node "ticket_agent".

    offer_ticket ──yes──► ticket_agent = [collect_email ──► draft_ticket ──► confirm_ticket ──yes──► create_ticket
        │ no                 │ interrupt: "your email?"      │ LLM + bind_tools   │ interrupt:            │ runs the approved
        ▼                    │ (skipped if already known)    │ proposes a         │ "Shall I go ahead?"   │ create_ticket call
       END                   ▼                               │ create_ticket call │ no ─► END             ▼
                         (invalid → ask again)               │ no call ─► ask_details    other ─► draft_ticket (edit)
                                                             ▼ (interrupt: "what's it about?")

- interrupt(): every point where the employee must answer pauses the graph; the checkpointer stores
  the paused state under the thread_id and the app resumes it with Command(resume=<reply>).
  Nodes are pure before interrupt() because LangGraph re-runs the node from the top on resume.
- bind_tools: the LLM is bound to create_ticket_tool and *proposes* a tool call (arguments drafted from
  the conversation). The call is executed only by create_ticket, after an explicit "yes".
"""
from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from agents import prompts
from agents.confirm import is_no, is_yes
from agents.llm import get_tool_llm
from agents.state import AgentState
from tools import create_ticket as ticket_tool
from tools.create_ticket import TicketInput
from tools.registry import TICKET_AGENT_TOOLS
from tools.ticketing_client import TicketingError, TicketingUnavailable

log = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TRANSCRIPT_TURNS = 12
SUMMARY_MAX = 80

_ticket_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=prompts.TICKET_TOOL_PROMPT),
    ("human", prompts.TICKET_TOOL_INPUT),
])


def _get_ticket_llm():
    """prompt | ChatOpenAI.bind_tools(TICKET_AGENT_TOOLS) with retry + fallback model."""
    return _ticket_prompt | get_tool_llm(TICKET_AGENT_TOOLS)


def _run_create_tool(args: dict) -> dict:
    """Execute the approved create_ticket tool call."""
    return ticket_tool.create_ticket_tool.invoke(args)


# ---------------------------------------------------------------------------- helpers
def _say(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _ask(state: AgentState, prompt: str, kind: str) -> tuple[str, list[dict]]:
    """Pause the graph and wait for the employee's reply.

    Returns (reply, messages_to_add). The prompt is added to the conversation unless the previous
    node already did (e.g. the KB agent's ticket offer)."""
    msgs = state.get("messages", [])
    already_shown = bool(msgs) and msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == prompt
    reply = interrupt({"type": kind, "message": prompt})
    reply = reply.get("text", "") if isinstance(reply, dict) else str(reply)
    return reply.strip(), ([] if already_shown else [_say(prompt)]) + [_user(reply.strip())]


def _email_in(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    return m.group(0).lower() if m else None


def _transcript(state: AgentState) -> str:
    msgs = state.get("messages", [])[-TRANSCRIPT_TURNS:]
    return "\n".join(f"{'Employee' if m['role'] == 'user' else 'SmartDesk'}: {m['content']}" for m in msgs)


def _cancel(new_messages: list[dict]) -> Command:
    return Command(goto=END, update={"messages": new_messages + [_say(prompts.TICKET_CANCELLED)],
                                     **_cleared()})


def _cleared() -> dict:
    return {"proposed_ticket": None, "change_request": None, "pending_question": None, "escalation_reason": None,
            "needs_escalation": False, "ticket_error": None, "email_invalid": False}


def _confirm_text(args: dict) -> str:
    return prompts.TICKET_CONFIRM.format(
        summary=args["summary"], description=args["description"],
        category_label=prompts.CATEGORY_LABEL[args["category"]], priority=args["priority"], email=args["email"])


# ---------------------------------------------------------------------------- nodes
def _is_new_question(text: str) -> bool:
    """A reply that's clearly a new request rather than an answer to a yes/no question."""
    return "?" in text or len(text.split()) >= 3


def offer_ticket(state: AgentState) -> Command[Literal["ticket_agent", "offer_ticket", "router", "__end__"]]:
    """HITL #1 – the KB agent couldn't answer and offered a ticket: wait for yes / no.
    If the employee asks something new instead, the offer is dropped and the message is routed normally."""
    prompt = state["messages"][-1]["content"]           # the offer / "please reply yes or no"
    reply, new = _ask(state, prompt, "ticket_offer")
    if email := _email_in(reply):
        new_email = {"employee_email": email}
    else:
        new_email = {}
    if is_no(reply):
        return Command(goto=END, update={"messages": new + [_say(prompts.ESCALATION_DECLINED)],
                                         **_cleared(), **new_email})
    if is_yes(reply) or new_email:
        return Command(goto="ticket_agent", update={"messages": new, **new_email})
    if _is_new_question(reply):                                   # e.g. "whats the maternity policy"
        return Command(goto="router", update={"messages": new, **_cleared()})
    return Command(goto="offer_ticket", update={"messages": new + [_say(prompts.ESCALATION_UNCLEAR)]})


def collect_email(state: AgentState) -> Command[Literal["draft_ticket", "collect_email", "__end__"]]:
    """Entry of the ticket flow. HITL – ask for the email only if it isn't known in this session."""
    if state.get("employee_email"):
        return Command(goto="draft_ticket")
    if email := _email_in(state["messages"][-1]["content"]) if state.get("messages") else None:
        return Command(goto="draft_ticket", update={"employee_email": email})

    prompt = prompts.TICKET_BAD_EMAIL if state.get("email_invalid") else prompts.TICKET_ASK_EMAIL
    reply, new = _ask(state, prompt, "email")
    if email := _email_in(reply):
        return Command(goto="draft_ticket", update={"messages": new, "employee_email": email, "email_invalid": False})
    if is_no(reply):
        return _cancel(new)
    return Command(goto="collect_email", update={"messages": new, "email_invalid": True})


_TICKET_REQUEST = re.compile(r"\b(raise|open|create|log|file|submit|new)\b.*\bticket\b", re.I)


def _is_filler(text: str) -> bool:
    """Messages that aren't a description of the issue: emails, yes/no, 'create a ticket'."""
    t = text.strip()
    return bool(EMAIL_RE.fullmatch(t) or is_yes(t) or is_no(t) or _TICKET_REQUEST.search(t))


def _fallback_args(state: AgentState) -> dict | None:
    """Plain proposal used only if the LLM is unavailable (the employee still reviews it).
    Uses the unanswered question (or the latest message describing the issue) – never joins
    unrelated earlier messages. Returns None if no issue has been described yet."""
    current = (state.get("proposed_ticket") or {}).get("args")
    if current and state.get("change_request"):
        return {**current, "description": f"{current['description']}\n\nAdditional details: {state['change_request']}"}
    issue = (state.get("pending_question") or "").strip()
    if not issue or _is_filler(issue):
        issue = next((m["content"].strip() for m in reversed(state.get("messages", []))
                      if m["role"] == "user" and not _is_filler(m["content"])), "")
    if not issue:
        return None
    return {"email": state["employee_email"], "summary": issue[:SUMMARY_MAX],
            "description": f"The employee reported: {issue}",
            "category": state.get("domain") or "IT", "priority": "Medium"}


def _clean_args(args: dict, state: AgentState) -> dict:
    args = {**args, "email": state["employee_email"]}            # never trust an LLM-written email
    summary = " ".join(str(args.get("summary", "")).split())
    args["summary"] = summary if len(summary) <= SUMMARY_MAX else summary[:SUMMARY_MAX - 1].rstrip() + "…"
    args.setdefault("category", state.get("domain") or "IT")
    args.setdefault("priority", "Medium")
    return TicketInput(**args).model_dump(mode="json")             # validate against the tool schema


def draft_ticket(state: AgentState) -> Command[Literal["confirm_ticket", "ask_details"]]:
    """LLM + bind_tools proposes a create_ticket call drafted from the conversation (not executed here)."""
    current = (state.get("proposed_ticket") or {}).get("args")
    change = state.get("change_request")
    revision = prompts.TICKET_REVISION_BLOCK.format(
        draft="\n".join(f"{k}: {current[k]}" for k in ("summary", "description", "category", "priority")),
        change_request=change) if current and change else ""

    args, source = None, "llm"
    try:
        ai = _get_ticket_llm().invoke({
            "email": state["employee_email"], "transcript": _transcript(state),
            "pending_question": state.get("pending_question") or "(none)",
            "category_hint": state.get("domain") or "(none)", "revision": revision})
        calls = [c for c in ai.tool_calls if c["name"] == "create_ticket"]
        if calls:
            args = _clean_args(calls[0]["args"], state)
        elif not current:
            return Command(goto="ask_details")                     # LLM: issue not described yet
    except ValidationError as e:
        log.warning("LLM proposed invalid ticket arguments, using fallback draft: %s", e)
    except Exception:  # noqa: BLE001 – LLM unavailable: plain draft, still confirmed by the employee
        log.exception("Ticket drafting LLM failed, using fallback draft")

    if args is None:
        fallback = _fallback_args(state)
        if fallback is None:
            return Command(goto="ask_details")
        args, source = _clean_args(fallback, state), "fallback"
    call = {"name": "create_ticket", "id": source, "args": args}
    return Command(goto="confirm_ticket", update={"proposed_ticket": call, "change_request": None})


def ask_details(state: AgentState) -> Command[Literal["draft_ticket", "__end__"]]:
    """HITL – the employee asked for a ticket without saying what it's about."""
    reply, new = _ask(state, prompts.TICKET_ASK_DETAILS, "ticket_details")
    if is_no(reply):
        return _cancel(new)
    return Command(goto="draft_ticket", update={"messages": new, "pending_question": reply})


def confirm_ticket(state: AgentState) -> Command[Literal["create_ticket", "draft_ticket", "__end__"]]:
    """HITL #2 – show the proposed ticket and wait for approval. Nothing is created without "yes"."""
    prompt = state.get("ticket_error") or _confirm_text(state["proposed_ticket"]["args"])
    reply, new = _ask(state, prompt, "ticket_confirmation")
    if is_no(reply):
        return _cancel(new)
    if is_yes(reply) and len(reply.split()) <= 4:                  # "yes", "yes please go ahead", "try again"
        return Command(goto="create_ticket", update={"messages": new, "ticket_error": None})
    # anything else = extra details / changes (incl. "yes, but make it High") -> redraft and re-confirm
    return Command(goto="draft_ticket", update={"messages": new, "change_request": reply, "ticket_error": None})


def create_ticket(state: AgentState) -> Command[Literal["confirm_ticket", "__end__"]]:
    """Execute the approved create_ticket tool call and report the ticket ID and link."""
    args = state["proposed_ticket"]["args"]
    team = prompts.TEAM[args["category"]]
    try:
        t = _run_create_tool(args)
    except TicketingUnavailable as e:
        log.error("Jira unavailable: %s", e)
        msg = prompts.TICKET_UNAVAILABLE
    except (TicketingError, ValueError) as e:
        log.error("Ticket creation rejected: %s", e)
        msg = prompts.TICKET_FAILED.format(team=team)
    else:
        return Command(goto=END, update={
            "messages": [_say(prompts.TICKET_CREATED.format(key=t["key"], url=t["url"], team=team))],
            "last_ticket": t, **_cleared()})
    # keep the proposal; confirm_ticket shows the error and "try again" retries
    return Command(goto="confirm_ticket", update={"messages": [_say(msg)], "ticket_error": msg})
