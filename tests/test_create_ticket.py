"""Independent tests for the Jira ticket-creation tool (no network: a fake HTTP session plays Jira)."""
import pytest
import requests
from pydantic import ValidationError

import tools.create_ticket as ct
from tools import ticket_store
from tools.ticketing_client import JiraClient, TicketingError, TicketingUnavailable


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}
        self.content = b"x" if body is not None else b""
        self.text = str(body)

    def json(self):
        return self._body


class FakeJira(requests.Session):
    """Returns queued responses (or raises queued exceptions) and records every request."""

    def __init__(self, *responses):
        super().__init__()
        self.queue, self.calls = list(responses), []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        nxt = self.queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


CREATED = FakeResponse(201, {"id": "10001", "key": "IT-42", "self": "https://x/rest/api/3/issue/10001"})
TICKET = dict(email="Jane.Doe@nmtech.com", summary="Monitor flickering",
              description="Monitor has been flickering for two days.\n\nTried a different cable.",
              category="IT", priority="Critical")


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("JIRA_BASE_URL", "https://nmtech.atlassian.net/")
    monkeypatch.setenv("JIRA_EMAIL", "bot@nmtech.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "token")
    monkeypatch.setenv("JIRA_IT_PROJECT_KEY", "IT")
    monkeypatch.setenv("JIRA_HR_PROJECT_KEY", "HR")
    monkeypatch.setattr(ct.settings, "USE_MOCK_TICKETING", False)
    monkeypatch.setattr(ct.settings, "TICKET_DB_PATH", str(tmp_path / "tickets.db"))
    monkeypatch.setattr(ct.settings, "JIRA_MAX_RETRIES", 3)


def client(*responses):
    return JiraClient(session=FakeJira(*responses))


def test_creates_ticket_with_all_fields():
    c = client(CREATED)
    out = ct.create_ticket(**TICKET, client=c)
    assert out["key"] == "IT-42"
    assert out["url"] == "https://nmtech.atlassian.net/browse/IT-42"

    call = c.session.calls[0]
    assert call["method"] == "POST" and call["url"].endswith("/rest/api/3/issue")
    assert call["timeout"] == ct.settings.JIRA_TIMEOUT_SECONDS
    f = call["json"]["fields"]
    assert f["project"] == {"key": "IT"}
    assert f["issuetype"] == {"name": "Task"}
    assert f["summary"] == "Monitor flickering"
    assert f["priority"] == {"name": "Highest"}                       # Critical -> Highest
    assert set(f["labels"]) == {"smartdesk", "it", "priority-critical", "requester-jane.doe_nmtech.com"}
    adf = f["description"]
    assert adf["type"] == "doc" and adf["version"] == 1
    text = str(adf)
    assert "jane.doe@nmtech.com" in text.lower() and "Tried a different cable." in text
    assert c.session.auth == ("bot@nmtech.com", "token")


def test_saves_ticket_locally_for_status_lookup():
    ct.create_ticket(**TICKET, client=client(CREATED))
    rows = ticket_store.tickets_for_email("JANE.DOE@nmtech.com")
    assert [r["ticket_key"] for r in rows] == ["IT-42"]
    assert rows[0]["url"].endswith("/browse/IT-42")


def test_hr_ticket_goes_to_hr_project():
    c = client(FakeResponse(201, {"id": "1", "key": "HR-7"}))
    out = ct.create_ticket(**{**TICKET, "category": "hr", "priority": "low"}, client=c)
    assert out["key"] == "HR-7" and out["category"] == "HR"
    assert c.session.calls[0]["json"]["fields"]["project"] == {"key": "HR"}


@pytest.mark.parametrize("bad", [
    {"email": "not-an-email"},
    {"summary": "hi"},
    {"description": "   "},
    {"category": "Finance"},
    {"priority": "Urgent"},
])
def test_invalid_input_is_rejected_before_calling_jira(bad):
    c = client()
    with pytest.raises((ValidationError, ValueError)):
        ct.create_ticket(**{**TICKET, **bad}, client=c)
    assert c.session.calls == []


def test_multiline_summary_is_flattened():
    c = client(CREATED)
    ct.create_ticket(**{**TICKET, "summary": "VPN\nnot connecting"}, client=c)
    assert c.session.calls[0]["json"]["fields"]["summary"] == "VPN not connecting"


def test_retries_without_priority_when_project_does_not_support_it():
    rejected = FakeResponse(400, {"errorMessages": [], "errors": {
        "priority": "Field 'priority' cannot be set. It is not on the appropriate screen, or unknown."}})
    c = client(rejected, CREATED)
    out = ct.create_ticket(**TICKET, client=c)
    assert out["key"] == "IT-42"
    assert "priority" in c.session.calls[0]["json"]["fields"]
    assert "priority" not in c.session.calls[1]["json"]["fields"]
    assert "priority-critical" in c.session.calls[1]["json"]["fields"]["labels"]


def test_bad_auth_raises_clear_error():
    with pytest.raises(TicketingError, match="Authentication failed"):
        ct.create_ticket(**TICKET, client=client(FakeResponse(401, {})))


def test_other_validation_errors_are_not_retried():
    c = client(FakeResponse(400, {"errorMessages": [], "errors": {"project": "valid project is required"}}))
    with pytest.raises(TicketingError, match="project"):
        ct.create_ticket(**TICKET, client=c)
    assert len(c.session.calls) == 1


def test_transient_errors_are_retried():
    c = client(FakeResponse(503, {}), requests.ConnectionError("reset"), CREATED)
    assert ct.create_ticket(**TICKET, client=c)["key"] == "IT-42"
    assert len(c.session.calls) == 3


def test_jira_down_raises_ticketing_unavailable():
    c = client(*[requests.ConnectionError("down")] * 3)
    with pytest.raises(TicketingUnavailable):
        ct.create_ticket(**TICKET, client=c)
    assert len(c.session.calls) == 3
    assert ticket_store.tickets_for_email(TICKET["email"]) == []   # nothing saved on failure


def test_mock_mode(monkeypatch):
    monkeypatch.setattr(ct.settings, "USE_MOCK_TICKETING", True)
    out = ct.create_ticket(**TICKET)
    assert out["key"].startswith("IT-") and out["url"].startswith("https://mock-jira.local/browse/")


def test_langchain_tool_interface(monkeypatch):
    monkeypatch.setattr(ct.settings, "USE_MOCK_TICKETING", True)
    assert ct.create_ticket_tool.name == "create_ticket"
    assert set(ct.create_ticket_tool.args) == {"email", "summary", "description", "category", "priority"}
    out = ct.create_ticket_tool.invoke({**TICKET, "category": "HR"})
    assert out["key"].startswith("HR-")
