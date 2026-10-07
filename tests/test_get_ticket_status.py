"""Independent tests for the ticket-status read tool (fake HTTP session plays Jira)."""
import pytest
import requests

import tools.create_ticket as ct
from tools import get_ticket_status as gs
from tools.ticketing_client import JiraClient, TicketingUnavailable


class R:
    def __init__(self, status, body):
        self.status_code, self._b, self.content, self.text = status, body, b"x", str(body)

    def json(self):
        return self._b


class FakeJira(requests.Session):
    def __init__(self, *responses):
        super().__init__()
        self.queue, self.calls = list(responses), []

    def request(self, method, url, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        nxt = self.queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def issue(key, summary, status, cat, updated):
    return {"key": key, "fields": {"summary": summary, "status": {"name": status, "statusCategory": {"key": cat}},
                                   "priority": {"name": "Medium"}, "created": "2026-10-01T09:00:00.000-0600",
                                   "updated": updated}}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL", "https://nmtech.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "bot@nmtech.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "t")
    monkeypatch.setenv("JIRA_IT_PROJECT_KEY", "IT")
    monkeypatch.setenv("JIRA_HR_PROJECT_KEY", "HR")
    monkeypatch.setattr(ct.settings, "USE_MOCK_TICKETING", False)
    monkeypatch.setattr(ct.settings, "JIRA_MAX_RETRIES", 2)


def test_searches_jira_by_requester_label_and_lists_open_first():
    c = JiraClient(session=FakeJira(R(200, {"issues": [
        issue("IT-3", "Laptop battery", "Done", "done", "2026-10-06T10:00:00.000-0600"),
        issue("IT-2", "VPN drops", "In Progress", "indeterminate", "2026-10-05T10:00:00.000-0600"),
        issue("HR-1", "W-2 copy", "To Do", "new", "2026-10-04T10:00:00.000-0600")]})))
    tickets = gs.get_tickets_by_email("Jane.Doe@nmtech.com", client=c)
    call = c.session.calls[0]
    assert call["url"].endswith("/rest/api/3/search/jql")
    assert call["params"]["jql"] == 'labels = "requester-jane.doe_nmtech.com" AND project in (IT, HR) ORDER BY updated DESC'
    assert [t["key"] for t in tickets] == ["IT-2", "HR-1", "IT-3"]          # open first, newest first
    assert tickets[0]["state"] == "In Progress" and tickets[1]["state"] == "Open"
    assert tickets[2]["state"] == "Resolved / Closed" and not tickets[2]["is_open"]
    assert tickets[0]["url"] == "https://nmtech.atlassian.net/browse/IT-2" and tickets[0]["updated"] == "2026-10-05"


def test_falls_back_to_legacy_search_endpoint():
    c = JiraClient(session=FakeJira(R(410, {"errorMessages": ["gone"]}), R(200, {"issues": []})))
    assert gs.get_tickets_by_email("jane@nmtech.com", client=c) == []
    assert c.session.calls[1]["url"].endswith("/rest/api/3/search")


def test_details_include_latest_comments_as_text():
    adf = {"type": "doc", "version": 1, "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "Replacement monitor ordered, "},
                                          {"type": "text", "text": "delivery by Thursday."}]}]}
    c = JiraClient(session=FakeJira(
        R(200, issue("IT-2", "Monitor flickering", "In Progress", "indeterminate", "2026-10-06T10:00:00.000-0600")),
        R(200, {"comments": [{"author": {"displayName": "Sam (IT)"}, "created": "2026-10-06T11:00:00.000-0600", "body": adf}]})))
    d = gs.get_ticket_details("IT-2", client=c)
    assert d["status"] == "In Progress" and d["comments"][0] == {
        "author": "Sam (IT)", "created": "2026-10-06", "text": "Replacement monitor ordered, delivery by Thursday."}
    assert c.session.calls[1]["params"]["orderBy"] == "-created"


def test_jira_down_raises_unavailable():
    c = JiraClient(session=FakeJira(*[requests.ConnectionError("down")] * 2))
    with pytest.raises(TicketingUnavailable):
        gs.get_tickets_by_email("jane@nmtech.com", client=c)


def test_mock_client_search_and_comments(monkeypatch):
    from tools import mock_ticketing
    monkeypatch.setattr(ct.settings, "USE_MOCK_TICKETING", True)
    monkeypatch.setattr(mock_ticketing, "_mock", None)
    m = mock_ticketing.get_mock_client()
    k1 = m.create_issue({"project": {"key": "IT"}, "summary": "VPN", "labels": ["requester-jane_nmtech.com"]})["key"]
    m.create_issue({"project": {"key": "IT"}, "summary": "Other", "labels": ["requester-bob_nmtech.com"]})
    m.add_comment(k1, "Looking into it")
    m.set_status(k1, "In Progress")
    [t] = gs.get_tickets_by_email("jane@nmtech.com")
    assert t["key"] == k1 and t["state"] == "In Progress"
    assert gs.get_ticket_details(k1)["comments"][0]["text"] == "Looking into it"
