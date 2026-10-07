"""SmartDesk LangGraph: orchestrator + agents, state management, human-in-the-loop.

    START ─► router ─┬─► it_agent ─┐            (Command: answer → END,
                     ├─► hr_agent ─┤             can't answer → offer_ticket,
                     │             └─────────── other domain → hr_agent / it_agent)
                     ├─► smalltalk_agent ─► END
                     ├─► status_agent ─► END     (create_agent: model ⇄ tools; interrupts inside tools)
                     │      └─ LLM down ─► status_fallback ─► END
                     ├─► ticket_agent ─► END
                     └─► Send() × N ─► it/hr/smalltalk/status agents (concurrent) ─► synthesizer ─► END | one follow-up
                                ▲
                     offer_ticket ─(yes)

    ticket_agent is a SUBGRAPH (agents/ticket_graph.py):
        collect_email ─► draft_ticket ─► confirm_ticket ─► create_ticket ─► END
                             ▲   └─► ask_details   │ (edits)
                             └─────────────────────┘

- State: AgentState (agents/state.py); messages are appended via a reducer.
- Persistence: an in-memory MemorySaver checkpointer stores each conversation under its thread_id –
  session memory (email, last ticket, history, paused interrupts) survives across turns.
- Parallel: a multi-part message ("hi! + IT question + ticket status") is split by the router into
  sub-tasks, fanned out with Send() straight to those agents (running concurrently), and merged by
  the synthesizer (agents/synthesizer.py).
- Subgraph: the ticket-creation flow is its own compiled graph, embedded as the node "ticket_agent";
  it shares AgentState and the parent's checkpointer, so its interrupts pause the whole conversation.
- Human-in-the-loop: offer_ticket, ticket_agent (email), ask_details, confirm_ticket, the status
  agent's tools (email, ask_employee) and status_fallback call interrupt(); the graph pauses and resumes with
  Command(resume=<employee reply>).
- bind_tools: draft_ticket lets the LLM propose a create_ticket tool call; create_ticket runs it
  only after the employee approves. status_agent is a LangChain create_agent with the tools
  get_my_tickets / get_ticket_details / ask_employee / end_status_check (email injected, never from the LLM).

Use SmartDeskChat for a simple send-a-message / get-a-reply interface.
"""
from __future__ import annotations

import uuid
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agents import prompts
from agents.kb_agent import answer_from_kb
from agents.orchestrator import pick_agent, route_message
from agents.synthesizer import branch_error, branch_messages, branch_result, is_branch, synthesizer_node
from agents.smalltalk_agent import smalltalk_agent
from agents.state import AgentState
from agents.status_agent import status_agent, status_fallback
from agents.ticket_agent import offer_ticket
from agents.ticket_graph import build_ticket_graph

_KB_FIELDS = ("domain", "retrieved_chunks", "confidence", "sources", "cache_hit", "needs_escalation",
              "escalation_reason", "pending_question")


def _new_messages(before: AgentState, after: dict) -> list[dict]:
    return after.get("messages", [])[len(before.get("messages", [])):]


def _kb_node(domain: Literal["IT", "HR"]):
    system_prompt = prompts.IT_AGENT_PROMPT if domain == "IT" else prompts.HR_AGENT_PROMPT

    def node(state: AgentState) -> Command:
        if is_branch(state):                                     # one part of a multi-part message
            return _kb_branch(state, domain)
        out = answer_from_kb(state, domain=domain, system_prompt=system_prompt)
        update = {k: out[k] for k in _KB_FIELDS if k in out}
        update["messages"] = _new_messages(state, out)

        if other := out.get("reroute_to"):                      # question belongs to the other domain
            if state.get("rerouted"):                             # avoid IT <-> HR ping-pong
                return Command(goto=END, update={"messages": [{"role": "assistant",
                                                               "content": prompts.ROUTE_UNCLEAR}]})
            return Command(goto=f"{other.lower()}_agent", update={"rerouted": True, "domain": other})
        if out.get("needs_escalation"):
            return Command(goto="offer_ticket", update=update)  # pause there for yes / no
        return Command(goto=END, update=update)

    node.__name__ = f"{domain.lower()}_agent"
    other_node = "hr_agent" if domain == "IT" else "it_agent"
    node.__annotations__["return"] = Command[Literal["offer_ticket", other_node, "synthesizer", "__end__"]]
    return node                                                   # (annotation: for graph drawing)


def _kb_branch(state: dict, domain: str) -> Command:
    """IT / HR part of a multi-part message: answer it, or record that it couldn't be answered (no pause)."""
    try:
        messages = branch_messages(state)
        for _ in range(2):                                        # at most one IT <-> HR hand-off
            prompt = prompts.IT_AGENT_PROMPT if domain == "IT" else prompts.HR_AGENT_PROMPT
            out = answer_from_kb({"messages": messages}, domain=domain, system_prompt=prompt)
            if not out.get("reroute_to"):
                break
            domain = out["reroute_to"]
        if out.get("reroute_to"):
            return branch_result(state, prompts.ROUTE_UNCLEAR)
        if out.get("needs_escalation"):
            reason = out.get("escalation_reason")
            text = (out["messages"][-1]["content"].split("\n\n")[0] if reason == "partial_answer" else
                    prompts.ESCALATION_REASON_TEXT.get(reason, prompts.ESCALATION_REASON_TEXT["no_results"])
                    .format(domain=domain))
            return branch_result(state, text, followup="offer_ticket", domain=domain)
        return branch_result(state, out["messages"][-1]["content"], domain=domain)
    except Exception:  # noqa: BLE001 – one failing branch must not sink the others
        return branch_error(state)


def _smalltalk_node(state: AgentState) -> Command[Literal["synthesizer", "__end__"]]:
    if is_branch(state):
        try:
            out = smalltalk_agent({"messages": branch_messages(state)})
            return branch_result(state, out["messages"][-1]["content"])
        except Exception:  # noqa: BLE001
            return branch_error(state)
    return Command(goto=END, update={"messages": _new_messages(state, smalltalk_agent(state))})


def build_graph(checkpointer=None):
    """Compile the SmartDesk graph.

    Uses an in-memory MemorySaver checkpointer by default: every conversation (thread_id) keeps its
    state – messages, email, last ticket, paused interrupts – for as long as the process runs.
    Pass another checkpointer (e.g. a SQLite saver) to persist conversations across restarts.
    """
    g = StateGraph(AgentState)

    g.add_node("router", route_message)
    g.add_node("it_agent", _kb_node("IT"))
    g.add_node("hr_agent", _kb_node("HR"))
    g.add_node("smalltalk_agent", _smalltalk_node)
    # status check: ONE tool-calling agent (create_agent) + a no-LLM fallback (agents/status_agent.py)
    g.add_node("status_agent", status_agent)
    g.add_node("status_fallback", status_fallback)
    # ticket flow: offer node + the ticket-creation SUBGRAPH as a single node (agents/ticket_graph.py)
    g.add_node("offer_ticket", offer_ticket)
    g.add_node("ticket_agent", build_ticket_graph())

    g.add_edge(START, "router")
    # multi-part messages: the router Send()s each part straight to its agent; every branch ends with
    # Command(goto="synthesizer"), which runs once after all of them have finished (agents/synthesizer.py)
    g.add_node("synthesizer", synthesizer_node)

    g.add_conditional_edges("router", pick_agent, ["it_agent", "hr_agent", "ticket_agent",
                                                   "status_agent", "smalltalk_agent", "synthesizer"])
    g.add_edge("ticket_agent", END)                 # subgraph finished -> end of the turn
    # the agents route themselves with Command(goto=...): END, offer_ticket, the other agent or synthesizer

    return g.compile(checkpointer=checkpointer or MemorySaver())


class SmartDeskChat:
    """One employee conversation: send a message, get SmartDesk's reply.

    Handles both cases automatically:
      - graph idle   -> start a new turn with the message
      - graph paused -> resume the interrupt with the message as the employee's answer
    """

    def __init__(self, graph=None, thread_id: str | None = None):
        self.graph = graph or build_graph()
        self.thread_id = thread_id or str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}

    @property
    def state(self) -> dict:
        """Current conversation state. While a subgraph (e.g. the ticket flow) is paused mid-way, its
        in-progress values live in the subgraph's own checkpoint, so they're merged on top."""
        snap = self.graph.get_state(self.config, subgraphs=True)
        values = dict(snap.values)
        for task in snap.tasks:
            sub = getattr(task, "state", None)
            if sub is not None and hasattr(sub, "values"):
                values.update(sub.values)
        return values

    def waiting_for(self) -> str | None:
        """Type of question the graph is paused on (e.g. 'ticket_confirmation'), or None."""
        snap = self.graph.get_state(self.config)
        pending = [i for task in snap.tasks for i in task.interrupts]
        return pending[0].value.get("type") if pending else None

    def send(self, text: str) -> str:
        if self.waiting_for():
            result = self.graph.invoke(Command(resume=text), self.config)
        else:
            result = self.graph.invoke({"messages": [{"role": "user", "content": text}]}, self.config)
        if result.get("__interrupt__"):
            return result["__interrupt__"][0].value["message"]
        return result["messages"][-1]["content"]
