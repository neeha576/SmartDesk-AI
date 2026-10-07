"""IT sub-agent: answers IT questions from the IT slice of the knowledge base."""
from agents.kb_agent import answer_from_kb
from agents.prompts import IT_AGENT_PROMPT
from agents.state import AgentState


def it_agent(state: AgentState) -> AgentState:
    """Retrieve IT chunks, check confidence, answer only from context, or offer a ticket."""
    return answer_from_kb(state, domain="IT", system_prompt=IT_AGENT_PROMPT)
