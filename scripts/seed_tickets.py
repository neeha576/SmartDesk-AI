"""Create sample tickets for an employee so the status-check flow can be tested.

    python scripts/seed_tickets.py jane.doe@nmtech.com            # real Jira (USE_MOCK_TICKETING=false)

Creates three tickets (2 IT, 1 HR) through the same create_ticket tool the agent uses – so they carry
the requester label the status search relies on – and adds a support-team comment to the first.
Move a ticket to "In Progress" or "Done" in Jira to see status changes in the chat.

With USE_MOCK_TICKETING=true the mock Jira only lives inside one process, so seed it from the chat
instead:  python main.py --seed jane.doe@nmtech.com
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from tools.create_ticket import create_ticket, get_client  # noqa: E402

SAMPLES = [
    ("Monitor flickering for two days", "The employee reports their office monitor has been flickering for two days. "
     "SmartDesk could not find a troubleshooting guide in the knowledge base.", "IT", "Medium"),
    ("VPN disconnects every few minutes", "The employee reports GlobalProtect VPN disconnects every few minutes when "
     "working from home. They already reconnected and restarted the laptop.", "IT", "High"),
    ("Question about relocation allowance", "The employee asked whether NMTech offers a relocation allowance for "
     "moving to Albuquerque. The topic is not covered in the HR knowledge base.", "HR", "Low"),
]
FIRST_COMMENT = "Replacement monitor ordered – delivery expected by Thursday."


def seed(email: str) -> list[dict]:
    created = [create_ticket(email=email, summary=s, description=d, category=c, priority=p) for s, d, c, p in SAMPLES]
    get_client().add_comment(created[0]["key"], FIRST_COMMENT)
    if settings.USE_MOCK_TICKETING:
        from tools.mock_ticketing import get_mock_client
        get_mock_client().set_status(created[0]["key"], "In Progress")
    return created


def main():
    p = argparse.ArgumentParser(description="Create sample tickets for status-check testing.")
    p.add_argument("email")
    args = p.parse_args()
    if settings.USE_MOCK_TICKETING:
        print("USE_MOCK_TICKETING=true – mock tickets would vanish when this script exits.\n"
              "Use:  python main.py --seed", args.email)
        return 1
    for t in seed(args.email):
        print(f"Created {t['key']} ({t['category']}, {t['priority']}): {t['summary']}  {t['url']}")
    print(f"Added a support-team comment to the first ticket. Now ask SmartDesk: 'What is the status of my tickets?'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
