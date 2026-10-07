"""Ticket-creation subgraph.

    START ─► collect_email ─► draft_ticket ─┬─► confirm_ticket ─┬─► create_ticket ─► END
               (interrupt)    (LLM +         │    (interrupt)    │  (approved tool call)
                              bind_tools)    └─► ask_details     ├─► draft_ticket (edits)
                                                 (interrupt)     └─► END (cancelled)

Added to the main graph (agents/graph.py) as one node, "ticket_agent". It shares AgentState with the
main graph, has no checkpointer of its own (it uses the parent's MemorySaver, so its interrupts pause
and resume the whole conversation), and only ever exits to END – back to the main graph.

Because it is a standalone compiled graph it can also be run, drawn and tested on its own:
    build_ticket_graph(checkpointer=MemorySaver()).invoke(...)
"""
from __future__ import annotations

from langgraph.graph import START, StateGraph

from agents.state import AgentState
from agents.ticket_agent import ask_details, collect_email, confirm_ticket, create_ticket, draft_ticket


def build_ticket_graph(checkpointer=None):
    """Compile the ticket subgraph. Leave checkpointer=None when embedding it in the main graph."""
    g = StateGraph(AgentState)
    g.add_node("collect_email", collect_email)
    g.add_node("draft_ticket", draft_ticket)
    g.add_node("ask_details", ask_details)
    g.add_node("confirm_ticket", confirm_ticket)
    g.add_node("create_ticket", create_ticket)
    g.add_edge(START, "collect_email")
    # every node routes itself with Command(goto=...); END returns control to the main graph
    return g.compile(checkpointer=checkpointer, name="ticket_agent")
