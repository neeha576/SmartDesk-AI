"""Status agent – ONE tool-calling agent (LangChain `create_agent`) for checking ticket status.

    main graph:   router ─► status_agent ─► END            (or ─► router when the message isn't about tickets)
                                 │ LLM unavailable
                                 └─► status_fallback ─► END  (no-LLM version, same Jira data)

    inside status_agent (a create_agent graph):  model ⇄ tools
        tools (defined in tools/status_tools.py, bound via tools/registry.py STATUS_AGENT_TOOLS):
        get_my_tickets()            this employee's tickets (asks for the email first if it isn't known)
        get_ticket_details(key)     status, priority, dates, link, comments – ONLY for the employee's own tickets
        ask_employee(message)       interrupt(): show a question, wait, return the employee's reply
        end_status_check(reason)    the message isn't about tickets -> back to the router

Human-in-the-loop happens inside the tools with interrupt(); on resume LangGraph re-runs only the
tool step (read-only Jira calls), never the LLM call.

Privacy – the employee can only ever see their own tickets:
  1. The tools take NO email argument. The verified email is read from the agent state through
     LangChain's InjectedState, which is hidden from the LLM's tool schema – the LLM can't set it.
  2. get_ticket_details checks in code that the key is one of the employee's tickets; otherwise it
     returns NOT_YOUR_TICKET without revealing whether the ticket exists.
  3. STATUS_AGENT_PROMPT forbids discussing anyone else's tickets or email addresses.
Resilience – LangChain middleware: ModelRetryMiddleware (transient errors), ModelFallbackMiddleware
(LLM_FALLBACK_MODEL), ModelCallLimitMiddleware (loop guard); any remaining failure -> status_fallback.
"""
from __future__ import annotations

import logging
import operator
import re
from typing import Annotated, Literal, Optional

from langchain.agents import AgentState as AgentLoopState
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END
from langgraph.types import Command, interrupt

from agents import prompts
from agents.confirm import is_no
from agents.state import AgentState
from agents.synthesizer import branch_error, branch_result, is_branch
from config import settings
from tools import get_ticket_status as status_tool
from tools.registry import STATUS_AGENT_TOOLS
from tools.status_tools import email_in, reply_text
from tools.ticketing_client import TicketingError, TicketingUnavailable

STATUS_TOOLS = STATUS_AGENT_TOOLS          # get_my_tickets, get_ticket_details, ask_employee, end_status_check

log = logging.getLogger(__name__)

KEY_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]+-\d+)\b")
MAX_MODEL_CALLS = 6
_STOP = {"status", "ticket", "tickets", "update", "updates", "any", "anyone", "the", "did", "does", "look",
         "looked", "looking", "issue", "issues", "what", "whats", "about", "with", "for", "is", "of", "my",
         "request", "requests", "check", "checking", "progress", "has", "have", "been", "there", "news",
         "one", "that", "this", "please", "can", "you", "tell", "me", "how", "where", "when", "still",
         "open", "raised", "created", "latest", "yet", "and", "are", "was", "it", "at", "on", "to", "an"}


# ---------------------------------------------------------------------------- agent state
class StatusLoopState(AgentLoopState):
    """State of the status agent's own loop (LangChain messages live here, not in the chat)."""
    employee_email: Optional[str]                      # verified email – set by code, never by the LLM
    handed_back: bool
    transcript: Annotated[list[dict], operator.add]    # questions asked via interrupt + replies, for the chat


# ---------------------------------------------------------------------------- the agent
def _chat_model(model_name: str):
    from langchain_openai import ChatOpenAI
    settings.required("OPENAI_API_KEY")
    return ChatOpenAI(model=model_name, temperature=settings.LLM_TEMPERATURE,
                      timeout=settings.LLM_TIMEOUT_SECONDS, max_retries=0)   # ModelRetryMiddleware retries


def _get_status_model():
    """The chat model the agent uses (LLM_MODEL from .env)."""
    if settings.llm_provider() != "openai":
        raise ValueError("Only LLM_PROVIDER=openai is implemented.")
    return _chat_model(settings.llm_model())


def _middleware() -> list:
    from agents.llm import transient_errors
    mw = [ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="error")]   # loop guard
    if fb := settings.llm_fallback_model():
        mw.append(ModelFallbackMiddleware(_chat_model(fb)))
    mw.append(ModelRetryMiddleware(max_retries=settings.LLM_MAX_RETRIES - 1, retry_on=transient_errors(),
                                   on_failure="error"))
    return mw


def build_status_agent(model=None, middleware=None):
    """create_agent(model, tools, system prompt, state) – the whole status agent."""
    return create_agent(model or _get_status_model(), tools=STATUS_TOOLS, system_prompt=prompts.STATUS_AGENT_PROMPT,
                        state_schema=StatusLoopState, middleware=_middleware() if middleware is None else middleware,
                        name="status_agent")


def _get_status_agent():
    return build_status_agent()


# ---------------------------------------------------------------------------- main-graph nodes
def _last_user_message(state: AgentState) -> str:
    return next((m["content"] for m in reversed(state.get("messages", [])) if m["role"] == "user"), "")


def status_agent(state: AgentState) -> Command[Literal["router", "status_fallback", "synthesizer", "__end__"]]:
    """Main-graph node: run the status agent on the employee's message and copy the result back."""
    if is_branch(state):                     # one part of a multi-part message: read-only summary, no pauses
        try:
            if not state.get("employee_email"):
                return branch_result(state, prompts.PARALLEL_STATUS_NEEDS_EMAIL, followup="status_agent")
            return branch_result(state, status_summary(state["employee_email"], state["task"]["request"]))
        except Exception:  # noqa: BLE001
            return branch_error(state)
    query = state.get("status_query") or _last_user_message(state)
    t = state.get("last_ticket")
    context = f"(Ticket created earlier in this conversation: {t['key']} – {t['summary']})\n" if t else ""
    email = state.get("employee_email") or email_in(query)
    try:
        out = _get_status_agent().invoke({"messages": [HumanMessage(content=context + query)],
                                          "employee_email": email, "handed_back": False, "transcript": []})
    except GraphBubbleUp:
        raise                                         # interrupt() inside a tool: pause the conversation
    except Exception:  # noqa: BLE001 – LLM unavailable / call limit hit: no-LLM version, same Jira data
        log.exception("Status agent failed; using the no-LLM fallback")
        return Command(goto="status_fallback")

    new = list(out.get("transcript", []))
    update = {"employee_email": out.get("employee_email") or email}
    if out.get("handed_back"):
        return Command(goto="router", update={"messages": new, "status_handed_back": True, **update})
    final = next((m for m in reversed(out["messages"]) if isinstance(m, AIMessage)), None)
    if final is not None and final.content:
        new.append({"role": "assistant", "content": final.content})
    return Command(goto=END, update={"messages": new, **update})


# ---------------------------------------------------------------------------- no-LLM fallback
def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 3 and w not in _STOP}


def _match(text: str, tickets: list[dict]) -> dict | None:
    """The single ticket the text refers to – by key, list position, or summary words – else None."""
    t = (text or "").strip()
    by_key = {x["key"].upper(): x for x in tickets}
    for k in KEY_RE.findall(t):
        if k.upper() in by_key:
            return by_key[k.upper()]
    if re.fullmatch(r"#?\s*\d{1,2}\.?", t):
        i = int(re.sub(r"\D", "", t))
        return tickets[i - 1] if 1 <= i <= len(tickets) else None
    words = _keywords(t)
    hits = [x for x in tickets if words & _keywords(x["summary"])] if words else []
    return hits[0] if len(hits) == 1 else None


def _list_message(email: str, tickets: list[dict]) -> str:
    lines = "\n".join(f"{i}. **{t['key']}** – {t['summary']} ({t['status']}, updated {t['updated']})"
                      for i, t in enumerate(tickets, 1))
    return prompts.STATUS_FALLBACK_LIST.format(email=email, lines=lines)


def _details_message(d: dict) -> str:
    comments = "\n".join(f"> {c['author']} ({c['created']}): {c['text']}" for c in d["comments"]) \
        or "No updates from the support team yet."
    status_line = d["status"] if d["status"].lower() == d["state"].lower() else f"{d['status']} ({d['state']})"
    return prompts.STATUS_FALLBACK_DETAIL.format(comments=comments, status_line=status_line,
                                                 **{k: v for k, v in d.items() if k != "comments"})


def status_fallback(state: AgentState) -> Command[Literal["router", "__end__"]]:
    """Status check without the LLM: email -> list -> (which one?) -> details, all from Jira.
    Only read-only Jira calls happen before each interrupt, so re-running on resume is harmless."""
    query = state.get("status_query") or _last_user_message(state)
    new: list[dict] = []
    email = state.get("employee_email") or email_in(query)
    if not email:
        prompt = prompts.STATUS_ASK_EMAIL
        while not email:
            reply = reply_text(interrupt({"type": "status_email", "message": prompt}))
            new += [{"role": "assistant", "content": prompt}, {"role": "user", "content": reply}]
            if is_no(reply):
                return Command(goto=END, update={"messages": new + [{"role": "assistant", "content": prompts.STATUS_CANCELLED}]})
            email, prompt = email_in(reply), prompts.TICKET_BAD_EMAIL
    try:
        tickets = status_tool.get_tickets_by_email(email)
        chosen = tickets[0] if len(tickets) == 1 else _match(query, tickets)
        while tickets and not chosen:
            prompt = _list_message(email, tickets)
            reply = reply_text(interrupt({"type": "choose_ticket", "message": prompt}))
            new += [{"role": "assistant", "content": prompt}, {"role": "user", "content": reply}]
            if is_no(reply):
                return Command(goto=END, update={"messages": new + [{"role": "assistant", "content": prompts.STATUS_CANCELLED}],
                                                 "employee_email": email})
            chosen = _match(reply, tickets)
            if not chosen and ("?" in reply or len(reply.split()) >= 4):        # a new question, not a choice
                return Command(goto="router", update={"messages": new, "employee_email": email,
                                                      "status_handed_back": True})
        text = _details_message(status_tool.get_ticket_details(chosen["key"])) if tickets \
            else prompts.STATUS_NONE.format(email=email)
    except (TicketingUnavailable, TicketingError) as e:
        log.error("Ticket status lookup failed: %s", e)
        text = prompts.STATUS_UNAVAILABLE
    return Command(goto=END, update={"messages": new + [{"role": "assistant", "content": text}], "employee_email": email})


def status_summary(email: str, request: str) -> str:
    """Read-only status answer for a parallel branch (no questions asked): the ticket the request names or
    describes, else the list of the employee's tickets. Same privacy rule: only this email's tickets."""
    try:
        tickets = status_tool.get_tickets_by_email(email)
        if not tickets:
            return prompts.STATUS_NONE.format(email=email)
        chosen = tickets[0] if len(tickets) == 1 else _match(request, tickets)
        if chosen:
            return _details_message(status_tool.get_ticket_details(chosen["key"]))
        return _list_message(email, tickets)
    except (TicketingUnavailable, TicketingError) as e:
        log.error("Ticket status lookup failed: %s", e)
        return prompts.STATUS_UNAVAILABLE
