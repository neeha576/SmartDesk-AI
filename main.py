"""SmartDesk AI – command-line chat with the full multi-agent graph.

    python main.py            # chat
    python main.py --debug    # also show routing, scores and what the graph is waiting for
    python main.py --seed jane.doe@nmtech.com   # mock mode: start with 3 sample tickets for that email

Type 'exit' to quit, 'new' to start a new conversation (new thread_id).
"""
from __future__ import annotations

import argparse
import logging

from config import settings  # loads .env
from agents.graph import SmartDeskChat, build_graph


def main():
    p = argparse.ArgumentParser(description="Chat with SmartDesk AI.")
    p.add_argument("--debug", action="store_true", help="Show routing and retrieval details.")
    p.add_argument("--seed", metavar="EMAIL", help="Mock mode only: create 3 sample tickets for EMAIL at start-up.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.debug else logging.WARNING,
                        format="   %(levelname)s %(name)s: %(message)s")

    if args.seed:
        if not settings.USE_MOCK_TICKETING:
            print("--seed is for mock mode. For real Jira run:  python scripts/seed_tickets.py", args.seed)
            return
        from scripts.seed_tickets import seed
        for t in seed(args.seed):
            print(f"(seeded {t['key']}: {t['summary']})")
        print()

    graph = build_graph()
    chat = SmartDeskChat(graph)
    print("SmartDesk AI – NMTech IT & HR help desk. Type 'exit' to quit, 'new' for a new conversation.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("exit", "quit"):
            break
        if text.lower() == "new":
            chat = SmartDeskChat(graph)
            print("(new conversation)\n")
            continue
        if not text:
            continue
        print(f"\nSmartDesk: {chat.send(text)}\n")
        if args.debug:
            s = chat.state
            score = s.get("confidence")
            print(f"   [route={s.get('route')} ({s.get('route_reason')}) domain={s.get('domain')} "
                  f"score={round(score, 3) if isinstance(score, float) else '-'} "
                  f"escalation={s.get('escalation_reason') or 'none'} cache_hit={bool(s.get('cache_hit'))} "
                  f"waiting_for={chat.waiting_for()} "
                  f"email={s.get('employee_email')}]\n")


if __name__ == "__main__":
    main()
