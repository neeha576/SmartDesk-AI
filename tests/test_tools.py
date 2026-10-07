"""Tool registry: every LangChain tool lives in tools/, and each agent binds exactly its registered tools."""
import pytest
from langchain_core.tools import BaseTool

from tools.registry import AGENT_TOOLS, STATUS_AGENT_TOOLS, TICKET_AGENT_TOOLS


def test_all_registered_tools_live_in_the_tools_package():
    for tools in AGENT_TOOLS.values():
        for t in tools:
            assert isinstance(t, BaseTool)
            assert (t.func or t.coroutine).__module__.startswith("tools."), t.name


def test_registry_contents():
    assert [t.name for t in TICKET_AGENT_TOOLS] == ["create_ticket"]
    assert [t.name for t in STATUS_AGENT_TOOLS] == ["get_my_tickets", "get_ticket_details",
                                                     "ask_employee", "end_status_check"]


@pytest.fixture
def llm_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")       # building the client makes no API call


def test_ticket_agent_binds_registry_tools(llm_env):
    import agents.ticket_agent as ta
    from tests.test_ticket_agent import _find_tools
    bound = [t["function"]["name"] for t in _find_tools(ta._get_ticket_llm())]
    assert bound == [t.name for t in TICKET_AGENT_TOOLS]


def test_status_agent_binds_registry_tools(llm_env):
    import agents.status_agent as sa
    agent = sa.build_status_agent()
    tools_node = agent.get_graph().nodes["tools"].data
    assert set(tools_node.tools_by_name) == {t.name for t in STATUS_AGENT_TOOLS}
