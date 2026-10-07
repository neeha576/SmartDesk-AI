"""Read operation: look up an employee's tickets in Jira and their status + support-team comments.

Tickets are linked to the employee by the `requester-<email>` label that create_ticket adds
(Jira can't search by an external requester's email). The search is done live in Jira, so status
changes and comments made by the support team are always current.

    get_tickets_by_email("jane.doe@nmtech.com")
    # -> [{'key': 'IT-42', 'summary': ..., 'status': 'In Progress', 'state': 'In Progress', 'is_open': True,
    #      'priority': 'Medium', 'updated': '2026-10-06', 'url': ...}, ...]   open tickets first, newest first
    get_ticket_details("IT-42")
    # -> {... same fields ..., 'comments': [{'author': 'Sam (IT)', 'created': '2026-10-06', 'text': '...'}]}

Raises TicketingUnavailable (Jira unreachable) or TicketingError (Jira rejected the request).
"""
from __future__ import annotations

from config import settings
from tools.create_ticket import _requester_label, get_client

# Jira status category -> plain-English state shown to employees
STATE = {"new": "Open", "indeterminate": "In Progress", "done": "Resolved / Closed"}
MAX_TICKETS = 10
FIELDS = "summary,status,priority,labels,created,updated"


def _date(value: str | None) -> str:
    return (value or "")[:10]


def _adf_to_text(node) -> str:
    """Flatten Atlassian Document Format (comment bodies) to plain text."""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return node.get("text", "")
    parts = [_adf_to_text(c) for c in node.get("content", [])]
    sep = "\n" if node.get("type") in ("doc", "bulletList", "orderedList") else ""
    return sep.join(p for p in parts if p).strip()


def _summarize(issue: dict, client) -> dict:
    f = issue.get("fields", {})
    status = f.get("status") or {}
    category = (status.get("statusCategory") or {}).get("key", "new")
    return {
        "key": issue["key"],
        "summary": f.get("summary", ""),
        "status": status.get("name", "Unknown"),
        "state": STATE.get(category, "In Progress"),
        "is_open": category != "done",
        "priority": (f.get("priority") or {}).get("name", "–"),
        "created": _date(f.get("created")),
        "updated": _date(f.get("updated")),
        "url": client.browse_url(issue["key"]),
    }


def _projects() -> list[str]:
    keys = []
    for cat in ("IT", "HR"):
        try:
            keys.append(settings.jira_project_key(cat))
        except RuntimeError:
            pass
    return keys


def get_tickets_by_email(email: str, client=None) -> list[dict]:
    """All tickets SmartDesk raised for this email: open first, then most recently updated."""
    client = client or get_client()
    jql = f'labels = "{_requester_label(email)}"'
    if projects := _projects():
        jql += f" AND project in ({', '.join(projects)})"
    jql += " ORDER BY updated DESC"
    tickets = [_summarize(i, client) for i in client.search_issues(jql, fields=FIELDS, max_results=MAX_TICKETS)]
    return sorted(tickets, key=lambda t: not t["is_open"])          # stable: keeps newest-first within groups


def get_ticket_details(ticket_key: str, client=None, max_comments: int = 3) -> dict:
    """Status, priority, dates, link and the latest support-team comments for one ticket."""
    client = client or get_client()
    details = _summarize(client.get_issue(ticket_key, fields=FIELDS), client)
    details["comments"] = [{"author": (c.get("author") or {}).get("displayName", "Support team"),
                            "created": _date(c.get("created")),
                            "text": _adf_to_text(c.get("body"))}
                           for c in client.get_comments(ticket_key, max_results=max_comments)]
    return details
