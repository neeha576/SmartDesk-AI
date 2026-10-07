"""Smoke tests for the Streamlit UI (ui/streamlit_app.py) with a stand-in chat – no LLM, Qdrant or Jira."""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import agents.graph as graph_module  # noqa: E402

from pathlib import Path  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py")


class FakeChat:
    """Records what the UI sends; pretends to ask for a confirmation when a ticket is requested."""
    sent: list[str] = []
    updates: list[dict] = []

    def __init__(self, graph=None):
        self.graph, self.config, self._waiting = self, {"configurable": {"thread_id": "t"}}, None

    def send(self, text):
        FakeChat.sent.append(text)
        if "ticket:" in text:
            self._waiting = "ticket_confirmation"
            return "Here's the ticket I'll raise… Shall I go ahead? (yes / no)"
        self._waiting = None
        return f"Answer to: {text}"

    def waiting_for(self):
        return self._waiting

    def update_state(self, config, values):          # stands in for graph.update_state
        FakeChat.updates.append(values)

    @property
    def state(self):
        return {}


@pytest.fixture
def app(monkeypatch):
    FakeChat.sent, FakeChat.updates = [], []
    monkeypatch.setattr(graph_module, "SmartDeskChat", FakeChat)
    monkeypatch.setattr(graph_module, "build_graph", lambda: object())
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_starts_with_welcome_and_actions(app):
    assert "SmartDesk" in app.chat_message[0].markdown[0].value
    labels = [b.label for b in app.button]
    assert {"Create ticket", "Check ticket status", "How do I reset my password?", "How much PTO do I get?"} <= set(labels)


def test_faq_button_asks_the_question(app):
    _button(app, "How do I reset my password?").click().run()
    assert FakeChat.sent == ["How do I reset my password?"]
    assert app.chat_message[-1].markdown[0].value == "Answer to: How do I reset my password?"


def test_status_button(app):
    _button(app, "Check ticket status").click().run()
    assert FakeChat.sent == ["What's the status of my tickets?"]


def test_chat_input_and_yes_no_buttons(app):
    app.chat_input[0].set_value("Please raise an IT ticket: laptop broken").run()
    assert FakeChat.sent[-1] == "Please raise an IT ticket: laptop broken"
    _button(app, "Yes").click().run()                # confirmation buttons appear while SmartDesk waits
    assert FakeChat.sent[-1] == "yes"
    assert not [b for b in app.button if b.label == "Yes"]


def test_email_is_given_to_the_graph(app):
    app.sidebar.text_input[0].input("Jane@NMTech.com").run()
    assert FakeChat.updates == [{"employee_email": "jane@nmtech.com"}]
