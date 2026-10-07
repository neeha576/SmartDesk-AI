"""Handles greetings, thanks, small talk and out-of-scope questions politely (LangChain chain)."""
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.llm import get_llm
from agents.prompts import SMALLTALK_FALLBACK, SMALLTALK_PROMPT
from agents.state import AgentState
from config import settings

_prompt = ChatPromptTemplate.from_messages([SystemMessage(content=SMALLTALK_PROMPT),
                                            MessagesPlaceholder("history")])


def smalltalk_agent(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    history = [("human" if m["role"] == "user" else "ai", m["content"])
               for m in messages if m["role"] in ("user", "assistant")][-settings.HISTORY_TURNS:]
    try:
        text = (_prompt | get_llm() | StrOutputParser()).invoke({"history": history}).strip()
    except Exception:  # noqa: BLE001 – LLM down: canned, safe greeting
        text = SMALLTALK_FALLBACK
    return {**state, "messages": messages + [{"role": "assistant", "content": text}]}
