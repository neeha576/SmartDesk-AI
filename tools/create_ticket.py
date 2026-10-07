"""Write operation: create a support ticket in Jira.

    from tools.create_ticket import create_ticket
    result = create_ticket(email="jane.doe@nmtech.com", summary="Monitor flickering",
                           description="Monitor has flickered for two days...", category="IT",
                           priority="Medium")
    # -> {'key': 'IT-42', 'url': 'https://<site>.atlassian.net/browse/IT-42', 'category': 'IT', ...}

- Validates every field before calling Jira (TicketInput).
- IT tickets go to JIRA_IT_PROJECT_KEY, HR tickets to JIRA_HR_PROJECT_KEY.
- Requester email, category, priority and the conversation summary go into the description
  (Atlassian Document Format), plus labels for filtering in Jira.
- If the Jira project doesn't allow setting Priority, the ticket is created without it (priority
  still recorded in the description and a label).
- Every created ticket is saved locally (tools/ticket_store.py) for the status-check flow.
- USE_MOCK_TICKETING=true uses an in-memory Jira stand-in.

Raises TicketingUnavailable (Jira down/unreachable) or TicketingError (Jira rejected the request);
ValueError / pydantic.ValidationError for invalid input.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, EmailStr, Field, field_validator

from config import settings
from tools import ticket_store
from tools.ticketing_client import JiraClient, TicketingError

log = logging.getLogger(__name__)

Priority = Literal["Low", "Medium", "High", "Critical"]
Category = Literal["IT", "HR"]

# SmartDesk priority -> Jira's default priority scheme
JIRA_PRIORITY = {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Highest"}


class TicketInput(BaseModel):
    """Fields required to create a SmartDesk support ticket."""
    email: EmailStr = Field(description="Employee's email address – links the ticket to the employee.")
    summary: str = Field(min_length=5, max_length=120,
                         description="Short one-line title, e.g. 'Unable to connect to VPN from home'.")
    description: str = Field(min_length=10, max_length=20000,
                             description="Detailed description summarising the conversation and what the employee needs.")
    category: Category = Field(description="'IT' or 'HR' – inferred from the conversation.")
    priority: Priority = Field(default="Medium", description="Low, Medium, High or Critical.")

    @field_validator("summary", "description")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("summary")
    @classmethod
    def _single_line(cls, v: str) -> str:
        return " ".join(v.split())  # Jira summaries can't contain newlines


# --------------------------------------------------------------------------- helpers
def _requester_label(email: str) -> str:
    return "requester-" + re.sub(r"[^a-z0-9._-]", "_", email.lower())


def _adf_description(t: TicketInput) -> dict:
    """Atlassian Document Format body: requester details, then the description paragraphs."""
    def field_line(label: str, value: str) -> dict:
        return {"type": "paragraph", "content": [
            {"type": "text", "text": f"{label}: ", "marks": [{"type": "strong"}]},
            {"type": "text", "text": value}]}

    paragraphs = [{"type": "paragraph", "content": [{"type": "text", "text": p}]}
                  for p in re.split(r"\n\s*\n", t.description) if p.strip()]
    return {"type": "doc", "version": 1, "content": [
        field_line("Requester", str(t.email)),
        field_line("Category", t.category),
        field_line("Priority", t.priority),
        field_line("Source", "SmartDesk AI help desk agent"),
        {"type": "rule"},
        *paragraphs,
    ]}


def _build_fields(t: TicketInput, include_priority: bool = True) -> dict:
    fields = {
        "project": {"key": settings.jira_project_key(t.category)},
        "issuetype": {"name": settings.JIRA_ISSUE_TYPE},
        "summary": t.summary,
        "description": _adf_description(t),
        "labels": ["smartdesk", t.category.lower(), f"priority-{t.priority.lower()}",
                   _requester_label(str(t.email))],
    }
    if include_priority:
        fields["priority"] = {"name": JIRA_PRIORITY[t.priority]}
    return fields


def get_client():
    if settings.USE_MOCK_TICKETING:
        from tools.mock_ticketing import get_mock_client
        return get_mock_client()
    return JiraClient()


# --------------------------------------------------------------------------- tool
def create_ticket(email: str, summary: str, description: str,
                  category: str = "IT", priority: str = "Medium", client=None) -> dict:
    """Create a Jira ticket and return {'key', 'url', 'category', 'priority', 'summary', 'email'}."""
    t = TicketInput(email=email, summary=summary, description=description,
                    category=category.upper(), priority=priority.capitalize())
    client = client or get_client()

    try:
        created = client.create_issue(_build_fields(t))
    except TicketingError as e:
        # Team-managed / simplified projects often don't have Priority on the create screen
        if e.status == 400 and "priority" in e.details:
            log.warning("Jira project doesn't accept Priority; creating without it (%s)", e)
            created = client.create_issue(_build_fields(t, include_priority=False))
        else:
            raise

    key = created["key"]
    url = client.browse_url(key)
    ticket_store.save_ticket(str(t.email), key, t.category, t.summary, t.priority, url)
    log.info("Created Jira ticket %s for %s", key, t.email)
    return {"key": key, "url": url, "category": t.category, "priority": t.priority,
            "summary": t.summary, "email": str(t.email)}


create_ticket_tool = StructuredTool.from_function(
    func=create_ticket,
    name="create_ticket",
    description=("Create an IT or HR support ticket in Jira. Only call this AFTER the employee has "
                 "confirmed the ticket summary. Returns the ticket key and link."),
    args_schema=TicketInput,
)
