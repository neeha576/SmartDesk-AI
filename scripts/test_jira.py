"""Check the live Jira connection and (optionally) create a real test ticket.

    python scripts/test_jira.py              # read-only checks: auth, projects, issue type, priority field
    python scripts/test_jira.py --create     # also create a real test ticket in the IT project
    python scripts/test_jira.py --create --category HR --email you@nmtech.com

Uses JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_IT_PROJECT_KEY, JIRA_HR_PROJECT_KEY from .env.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from tools.create_ticket import create_ticket  # noqa: E402
from tools.ticketing_client import JiraClient, TicketingError, TicketingUnavailable  # noqa: E402

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def check_project(client: JiraClient, category: str) -> bool:
    try:
        key = settings.jira_project_key(category)
    except RuntimeError as e:
        print(f"{BAD} {e}")
        return False
    try:
        project = client.get_project(key)
    except TicketingError as e:
        print(f"{BAD} {category} project '{key}': {e}")
        return False
    print(f"{OK} {category} project '{key}' found: {project.get('name')}")

    resp = client._request("GET", f"/rest/api/3/issue/createmeta/{key}/issuetypes")
    types = resp.get("issueTypes") or resp.get("values", [])
    names = {t["name"]: t["id"] for t in types}
    if settings.JIRA_ISSUE_TYPE not in names:
        print(f"{BAD}   issue type '{settings.JIRA_ISSUE_TYPE}' not available. "
              f"Available: {', '.join(names)}. Set JIRA_ISSUE_TYPE in .env.")
        return False
    print(f"{OK}   issue type '{settings.JIRA_ISSUE_TYPE}' available")

    meta = client._request("GET", f"/rest/api/3/issue/createmeta/{key}/issuetypes/{names[settings.JIRA_ISSUE_TYPE]}")
    fields = {f.get("fieldId") or f.get("key") for f in meta.get("fields", meta.get("values", []))}
    if "priority" in fields:
        print(f"{OK}   Priority can be set on create")
    else:
        print(f"{WARN}   Priority isn't on this project's create screen – tickets will be created without it "
              f"(priority still recorded in the description and a label)")
    return True


def main():
    p = argparse.ArgumentParser(description="Check the live Jira integration.")
    p.add_argument("--create", action="store_true", help="Create a real test ticket.")
    p.add_argument("--category", choices=["IT", "HR"], default="IT")
    p.add_argument("--email", default=None, help="Requester email for the test ticket (default: JIRA_EMAIL).")
    args = p.parse_args()

    if settings.USE_MOCK_TICKETING:
        print(f"{WARN} USE_MOCK_TICKETING=true – set it to false in .env to test the real Jira.")
        return 1

    try:
        client = JiraClient()
        me = client.myself()
        print(f"{OK} Authenticated to {client.base_url} as {me.get('displayName')} ({me.get('emailAddress', 'email hidden')})")
        ok = all([check_project(client, "IT"), check_project(client, "HR")])
        from tools.get_ticket_status import get_tickets_by_email
        found = get_tickets_by_email(settings.jira_email(), client=client)
        print(f"{OK} Ticket search works ({len(found)} SmartDesk ticket(s) for {settings.jira_email()})")
    except (TicketingError, TicketingUnavailable, RuntimeError) as e:
        print(f"{BAD} {e}")
        return 1

    if args.create:
        if not ok:
            print(f"{BAD} Fix the problems above before creating a ticket.")
            return 1
        try:
            t = create_ticket(email=args.email or settings.jira_email(),
                              summary="SmartDesk test ticket – please ignore",
                              description="Created by scripts/test_jira.py to verify the SmartDesk Jira integration.",
                              category=args.category, priority="Low")
        except (TicketingError, TicketingUnavailable) as e:
            print(f"{BAD} Ticket creation failed: {e}")
            return 1
        print(f"{OK} Created {t['key']}: {t['url']}")
        print(f"{OK} Saved locally for {t['email']} in {settings.TICKET_DB_PATH}")
    else:
        print("\nAll read-only checks done. Run with --create to create a real test ticket.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
