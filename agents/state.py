"""LangGraph state for SmartDesk.

The graph state is a TypedDict (LangGraph-native and checkpoint-serializable). Every boundary with
the LLM or a tool is a Pydantic model instead: RouteDecision (orchestrator), TicketInput
(create_ticket tool schema, used with bind_tools).

`messages` uses an id-aware append reducer (like LangGraph's add_messages): each node returns only
the messages it adds; every message gets an id the first time it's added, and a message whose id is
already in the conversation is skipped. That makes the state safe to share with subgraphs – a
subgraph returns the whole conversation, and without ids the parent would append it a second time.
"""
import uuid
from typing import Annotated, Literal, NotRequired, Optional, TypedDict

Domain = Literal["IT", "HR"]
AgentName = Literal["it_agent", "hr_agent", "ticket_agent", "status_agent", "smalltalk_agent"]


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    id: NotRequired[str]                # assigned by add_chat_messages


def add_chat_messages(left: list[ChatMessage] | None, right: list[ChatMessage] | None) -> list[ChatMessage]:
    """Append new messages; skip any whose id is already present (subgraphs return the full list)."""
    merged = list(left or [])
    seen = {m.get("id") for m in merged}
    for m in right or []:
        if m.get("id") and m["id"] in seen:
            continue
        if not m.get("id"):
            m = {**m, "id": uuid.uuid4().hex}
        merged.append(m)
        seen.add(m["id"])
    return merged


def collect_results(left: list | None, right: list | None) -> list:
    """Reducer for parallel branch results: branches append; None resets (start of each turn)."""
    if right is None:
        return []
    return list(left or []) + list(right)


class AgentState(TypedDict, total=False):
    # Conversation (appended by every node via the reducer)
    messages: Annotated[list[ChatMessage], add_chat_messages]

    # Routing (orchestrator)
    route: AgentName                # agent chosen for the latest message
    route_reason: str               # short explanation from the router (for logs/debugging)
    domain: Optional[Domain]        # IT/HR context of the current issue
    reroute_to: Optional[Domain]    # set when the IT/HR agent says it's the other domain's question
    rerouted: bool                  # prevents IT <-> HR ping-pong within one turn

    # Session memory (persists across turns for the thread)
    employee_email: Optional[str]
    last_ticket: Optional[dict]     # {'key', 'url', 'category', ...} of the most recent ticket

    # RAG results of the last knowledge-base answer
    retrieved_chunks: list
    confidence: Optional[float]     # top dense similarity score (or cache similarity on a cache hit)
    cache_hit: Optional[bool]       # answer came from the semantic cache (memory/semantic_cache.py)
    sources: list                   # doc_ids retrieved

    # Escalation (KB couldn't answer -> offer a ticket, human-in-the-loop via interrupt)
    needs_escalation: bool
    escalation_reason: Optional[str]  # no_results | low_confidence | llm_no_answer | partial_answer | llm_unavailable
    pending_question: Optional[str]   # the question that couldn't be answered

    # Ticket flow
    proposed_ticket: Optional[dict]   # create_ticket tool call proposed by the LLM: {'name','args','id'}
    change_request: Optional[str]     # employee's edits to the proposed ticket
    email_invalid: bool               # last email entered was invalid -> re-ask with a hint
    ticket_error: Optional[str]       # message shown when Jira failed; "try again" retries

    # Status check (agents/status_agent.py) – the agent keeps its own loop state; only this is shared
    status_handed_back: bool          # status agent said "not about tickets" -> router must not send it back
    status_query: Optional[str]       # status request to use instead of the last message (multi-intent follow-up)

    # Multi-intent messages (agents/synthesizer.py): router splits, Send() fans out, synthesizer merges
    route_tasks: list                 # [{'agent': ..., 'request': ...}] when a message has several requests
    partial_results: Annotated[list, collect_results]   # one result per parallel branch
    final_answer: Optional[str]       # the synthesizer's combined reply
