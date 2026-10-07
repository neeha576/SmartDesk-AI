"""HR sub-agent: answers HR questions from the HR slice of the knowledge base."""
from agents.kb_agent import answer_from_kb
from agents.prompts import HR_AGENT_PROMPT
from agents.state import AgentState


def hr_agent(state: AgentState) -> AgentState:
    """Retrieve HR chunks, check confidence, answer only from context, or offer a ticket."""
    return answer_from_kb(state, domain="HR", system_prompt=HR_AGENT_PROMPT)
