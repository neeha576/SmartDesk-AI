"""Tools for the status agent (bound to it in tools/registry.py -> agents/status_agent.py).

    get_my_tickets()               this employee's tickets (asks for the email first if it isn't known)
    get_ticket_details(ticket_key) status, priority, dates, link, comments – ONLY for the employee's own tickets
    ask_employee(message)          interrupt(): show a question, wait, return the employee's reply
    end_status_check(reason)       the message isn't about tickets -> hand it back to the orchestrator

Privacy: the tools take NO email argument. The verified email is read from the agent state with
LangChain's InjectedState (hidden from the tool schema the LLM sees), and get_ticket_details checks
in code that the ticket belongs to the employee (otherwise NOT_YOUR_TICKET).
Human-in-the-loop: the email question and ask_employee use interrupt() inside the tool; on resume
LangGraph re-runs only the tool step (read-only Jira calls), never the LLM call.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Annotated

from langchain.tools import InjectedState
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command, interrupt

from agents import prompts
from agents.confirm import is_no
from tools import get_ticket_status as status_tool
from tools.ticketing_client import TicketingError, TicketingUnavailable

log = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


# ---------------------------------------------------------------------------- helpers
def email_in(text: str) -> str | None:
    m = EMAIL_RE.search(text or "")
    return m.group(0).lower() if m else None


def reply_text(value) -> str:
    return (value.get("text", "") if isinstance(value, dict) else str(value)).strip()


def _ask_for_email() -> tuple[str | None, list[dict]]:
    """interrupt() until the employee gives a valid email (or declines). Returns (email|None, transcript)."""
    transcript, prompt = [], prompts.STATUS_ASK_EMAIL
    while True:
        reply = reply_text(interrupt({"type": "status_email", "message": prompt}))
        transcript += [{"role": "assistant", "content": prompt}, {"role": "user", "content": reply}]
        if email := email_in(reply):
            return email, transcript
        if is_no(reply):
            return None, transcript
        prompt = prompts.TICKET_BAD_EMAIL


DECLINED = "EMAIL_NOT_PROVIDED: the employee chose not to share their email, so their tickets can't be looked up."


def _compact(t: dict) -> dict:
    return {k: t[k] for k in ("key", "summary", "status", "state", "priority", "created", "updated", "url") if k in t}


def _tool_result(content: str, tool_call_id: str, email: str | None, transcript: list[dict]):
    """Plain string, or a Command that also stores a newly verified email + the email exchange."""
    if not transcript:
        return content
    return Command(update={"employee_email": email, "transcript": transcript,
                           "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]})


def _jira_error(e: Exception) -> str:
    log.error("Jira lookup failed: %s", e)
    if isinstance(e, TicketingUnavailable):
        return "ERROR: the ticketing system (Jira) is unreachable right now."
    return "ERROR: the ticketing system could not complete the lookup."


# ---------------------------------------------------------------------------- tools
@tool
def get_my_tickets(state: Annotated[dict, InjectedState],
                   tool_call_id: Annotated[str, InjectedToolCallId]):
    """List the support tickets raised under the employee's verified email (open tickets first):
    key, title, status, plain state, priority, dates and link."""
    email, transcript = state.get("employee_email"), []
    if not email:
        email, transcript = _ask_for_email()
        if not email:
            return _tool_result(DECLINED, tool_call_id, None, transcript)
    try:
        result = json.dumps([_compact(t) for t in status_tool.get_tickets_by_email(email)])
    except (TicketingUnavailable, TicketingError) as e:
        result = _jira_error(e)
    return _tool_result(result, tool_call_id, email, transcript)


@tool
def get_ticket_details(ticket_key: str, state: Annotated[dict, InjectedState],
                       tool_call_id: Annotated[str, InjectedToolCallId]):
    """Status, priority, dates, link and the latest support-team comments for ONE of the employee's
    own tickets. Returns NOT_YOUR_TICKET if the key isn't one of theirs."""
    email, transcript = state.get("employee_email"), []
    if not email:
        email, transcript = _ask_for_email()
        if not email:
            return _tool_result(DECLINED, tool_call_id, None, transcript)
    key = ticket_key.strip().upper()
    try:
        owned = {t["key"].upper() for t in status_tool.get_tickets_by_email(email)}
        if key not in owned:
            log.warning("Blocked details request for a ticket not owned by the employee: %s", key)
            result = "NOT_YOUR_TICKET"
        else:
            d = status_tool.get_ticket_details(key)
            result = json.dumps({**_compact(d), "comments": d["comments"]})
    except (TicketingUnavailable, TicketingError) as e:
        result = _jira_error(e)
    return _tool_result(result, tool_call_id, email, transcript)


@tool
def ask_employee(message: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Show the employee a message (e.g. a numbered list of their tickets asking which one) and wait for
    their reply. Returns the reply."""
    reply = reply_text(interrupt({"type": "choose_ticket", "message": message}))
    return Command(update={
        "transcript": [{"role": "assistant", "content": message}, {"role": "user", "content": reply}],
        "messages": [ToolMessage(content=f"Employee replied: {reply}", tool_call_id=tool_call_id)]})


@tool(return_direct=True)
def end_status_check(reason: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Use when the employee's latest message is NOT about their existing tickets (a new IT/HR question,
    a new ticket request, small talk). Hands the message back to the orchestrator."""
    return Command(update={"handed_back": True,
                           "messages": [ToolMessage(content=f"Handed back: {reason}", tool_call_id=tool_call_id)]})
