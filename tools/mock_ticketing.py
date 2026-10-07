"""In-memory Jira stand-in (USE_MOCK_TICKETING=true) – same interface as JiraClient.

Lets the whole app run and be demoed without a Jira account. Data lives for the process only.
Supports create, search by requester label, comments and status changes.
"""
from __future__ import annotations

import itertools
import re
from datetime import datetime, timezone

_CATEGORY = {"To Do": "new", "Open": "new", "In Progress": "indeterminate", "Waiting on Employee": "indeterminate",
             "Done": "done", "Resolved": "done", "Closed": "done"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MockJiraClient:
    base_url = "https://mock-jira.local"

    def __init__(self):
        self.issues: dict[str, dict] = {}
        self.comments: dict[str, list[dict]] = {}
        self._counters: dict[str, itertools.count] = {}

    def myself(self) -> dict:
        return {"displayName": "Mock User", "emailAddress": "mock@nmtech.com"}

    def get_project(self, key: str) -> dict:
        return {"key": key, "name": f"{key} (mock)"}

    def create_issue(self, fields: dict) -> dict:
        project = fields["project"]["key"]
        counter = self._counters.setdefault(project, itertools.count(1001))
        key = f"{project}-{next(counter)}"
        self.issues[key] = {"key": key, "fields": {
            **fields, "status": {"name": "To Do", "statusCategory": {"key": "new"}},
            "created": _now(), "updated": _now()}}
        return {"id": str(len(self.issues)), "key": key, "self": f"{self.base_url}/rest/api/3/issue/{key}"}

    def get_issue(self, key: str, fields: str = "") -> dict:
        return self.issues[key]

    def search_issues(self, jql: str, fields: str = "", max_results: int = 20) -> list[dict]:
        labels = re.findall(r'labels\s*=\s*"([^"]+)"', jql)
        hits = [i for i in self.issues.values() if all(lb in i["fields"].get("labels", []) for lb in labels)]
        return sorted(hits, key=lambda i: i["fields"]["updated"], reverse=True)[:max_results]

    def get_comments(self, key: str, max_results: int = 5) -> list[dict]:
        return list(reversed(self.comments.get(key, [])))[:max_results]

    def add_comment(self, key: str, text: str, author: str = "IT Service Desk") -> dict:
        c = {"author": {"displayName": author}, "created": _now(),
             "body": {"type": "doc", "version": 1,
                      "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}]}}
        self.comments.setdefault(key, []).append(c)
        self.issues[key]["fields"]["updated"] = _now()
        return c

    def set_status(self, key: str, status: str) -> None:
        """Demo helper: move a ticket to another status (e.g. 'In Progress', 'Done')."""
        self.issues[key]["fields"]["status"] = {"name": status, "statusCategory": {"key": _CATEGORY.get(status, "indeterminate")}}
        self.issues[key]["fields"]["updated"] = _now()

    def browse_url(self, key: str) -> str:
        return f"{self.base_url}/browse/{key}"


_mock = None


def get_mock_client() -> MockJiraClient:
    global _mock
    if _mock is None:
        _mock = MockJiraClient()
    return _mock
