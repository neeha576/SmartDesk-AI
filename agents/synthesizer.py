"""
agents/synthesizer.py — Multi-part messages: the agents run in parallel (Send()) and the synthesizer merges them.

                       ┌─Send()─► it_agent ───────┐
    router (tasks) ────┼─Send()─► smalltalk_agent ┼──► synthesizer ─► END
                       └─Send()─► status_agent ───┘        │ at most ONE interactive follow-up:
                                                            ├─► ticket_agent  (employee asked to raise a ticket)
                                                            ├─► offer_ticket  (a question couldn't be answered)
                                                            └─► status_agent  (status needs the email)

There is no extra "parallel" node: Send() starts the real agent nodes (it_agent, hr_agent, smalltalk_agent,
status_agent), each with one part of the message. In that mode (is_branch) an agent never pauses to ask the
employee anything – several branches pausing at once would mean several questions at once – and finishes with
Command(goto="synthesizer") carrying its output (branch_result). Interactive needs are recorded as a "followup".
Ticket requests aren't sent as branches; the synthesizer records them and starts the ticket flow afterwards.

synthesizer_node runs once, after every branch has finished: labels each output by agent ([IT Agent] …), asks the
LLM to merge them into one reply (final_answer), then starts at most one follow-up.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from agents.llm import get_llm
from agents.prompts import (
    PARALLEL_OFFER,
    PARALLEL_TICKET_REQUEST,
    SYNTHESIZER_EMPTY,
    SYNTHESIZER_PROMPT,
    TEAM,
)
from agents.state import AgentState
from config.logging_config import setup_logger

logger = setup_logger("smartdesk.synthesizer")

AGENT_LABELS = {
    "it_agent": "IT Agent",
    "hr_agent": "HR Agent",
    "smalltalk_agent": "Smalltalk Agent",
    "status_agent": "Status Agent",
    "ticket_agent": "Ticket Agent",
}

# ============================================================================ 1. branch helpers (used by the agents)
def is_branch(state: dict) -> bool:
    """True when an agent node was started by Send() for one part of a multi-part message."""
    return "task" in state


def branch_messages(state: dict) -> list[dict]:
    """The conversation as this branch sees it: earlier turns + its own part of the message."""
    return state.get("history", []) + [{"role": "user", "content": state["task"]["request"]}]


def branch_result(state: dict, text: str, **extra) -> Command[Literal["synthesizer"]]:
    """Finish a branch: hand this agent's output to the synthesizer (never pauses, never ENDs)."""
    task = state["task"]
    return Command(goto="synthesizer", update={"partial_results": [
        {"index": state["index"], "agent": task["agent"], "request": task["request"], "text": text, **extra}]})


def branch_error(state: dict) -> Command[Literal["synthesizer"]]:
    """A failing branch only affects its own part of the reply."""
    logger.exception("Branch failed for %s", state["task"]["agent"])
    return branch_result(state, f"Sorry – I couldn't handle “{state['task']['request']}” just now. "
                                "Please ask me about it again.")


# ============================================================================ 2. synthesizer (fan-in)
# at most one interactive follow-up after the combined reply, in this priority order
FOLLOWUP_ORDER = ["ticket_agent", "offer_ticket", "status_agent"]


def _combine_outputs(results: list[dict]) -> str:
    """Label each agent's output, in the order of the employee's message."""
    combined = ""
    for r in results:
        label = AGENT_LABELS.get(r["agent"], r["agent"])
        combined += f"[{label}] (request: {r['request']})\n{r['text']}\n\n"
    return combined


def synthesizer_node(state: AgentState) -> Command[Literal["ticket_agent", "offer_ticket", "status_agent", "__end__"]]:
    """Combine agent outputs into a final, friendly answer."""
    logger.info("Combining agent responses...")

    results = list(state.get("partial_results") or [])
    # ticket requests aren't run as branches (creating a ticket needs the employee): record them here
    results += [{"index": i, "agent": "ticket_agent", "request": t["request"], "followup": "ticket_agent",
                 "text": PARALLEL_TICKET_REQUEST.format(request=t["request"])}
                for i, t in enumerate(state.get("route_tasks") or []) if t["agent"] == "ticket_agent"]
    results.sort(key=lambda r: r["index"])
    user_query = state["messages"][-1]["content"]

    combined = _combine_outputs(results)
    if not combined.strip():
        combined = SYNTHESIZER_EMPTY

    try:
        final = get_llm().invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(content=f"Employee query: {user_query}\n\nAgent outputs:\n{combined}"),
            ]
        )
        final_answer = final.content.strip()
    except Exception:  # noqa: BLE001 – LLM down: the agents' answers joined as they are
        logger.exception("Synthesizer LLM failed; joining the agent outputs")
        final_answer = "\n\n".join(r["text"] for r in results) or SYNTHESIZER_EMPTY
    logger.info("Final answer ready (%d chars)", len(final_answer))

    # ---- one interactive follow-up (ticket request > ticket offer > status needing an email)
    follow = next((r for kind in FOLLOWUP_ORDER for r in results if r.get("followup") == kind), None)
    if follow is None:
        return Command(goto=END, update={"final_answer": final_answer,
                                         "messages": [{"role": "assistant", "content": final_answer}]})

    kind, request = follow["followup"], follow["request"]
    logger.info("Follow-up after synthesis: %s (%s)", kind, request)
    if kind == "offer_ticket":
        offer = PARALLEL_OFFER.format(request=request, team=TEAM[follow["domain"]])
        final_answer = f"{final_answer}\n\n{offer}"
        return Command(goto="offer_ticket", update={
            "final_answer": final_answer, "messages": [{"role": "assistant", "content": final_answer}],
            "pending_question": request, "domain": follow["domain"], "needs_escalation": True})
    update = {"final_answer": final_answer, "messages": [{"role": "assistant", "content": final_answer}]}
    if kind == "ticket_agent":
        return Command(goto="ticket_agent", update={**update, "pending_question": request})
    return Command(goto="status_agent", update={**update, "status_query": request})
