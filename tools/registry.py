"""Which tools are bound to which agent.

Every LangChain tool lives in the tools/ package; agents import their tool list from here and bind it:
  - ticket agent: llm.bind_tools(TICKET_AGENT_TOOLS)      (agents/ticket_agent.py)
  - status agent: create_agent(model, STATUS_AGENT_TOOLS)  (agents/status_agent.py)

The IT/HR agents use tools/kb_search.py through a fixed retrieval chain (agents/rag_chain.py) rather than
tool calling, so the knowledge base is always searched before the LLM answers.
"""
from tools.create_ticket import create_ticket_tool
from tools.status_tools import ask_employee, end_status_check, get_my_tickets, get_ticket_details

TICKET_AGENT_TOOLS = [create_ticket_tool]
STATUS_AGENT_TOOLS = [get_my_tickets, get_ticket_details, ask_employee, end_status_check]

AGENT_TOOLS = {
    "ticket_agent": TICKET_AGENT_TOOLS,
    "status_agent": STATUS_AGENT_TOOLS,
}
