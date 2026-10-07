"""Debug the IT / HR knowledge-base agents directly, without the orchestrator or ticket flow.
For the full conversation (routing, tickets, confirmations) use:  python main.py

Interactive chat with one agent (prints score, threshold, escalation reason, sources):
    python scripts/try_agent.py --domain it
    python scripts/try_agent.py --domain hr

Run every question in evaluation/eval_queries.json and report answer vs. escalation:
    python scripts/try_agent.py --eval

Needs a built index (python scripts/build_index.py) and the LLM/embedding settings in .env.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config.settings as settings  # noqa: E402  (loads .env)
from agents.hr_agent import hr_agent  # noqa: E402
from agents.it_agent import it_agent  # noqa: E402
from agents.smalltalk_agent import smalltalk_agent  # noqa: E402


AGENTS = {"IT": it_agent, "HR": hr_agent}


def debug_line(s: dict) -> str:
    score = s.get("confidence")
    score = f"{score:.3f}" if isinstance(score, float) else "-"
    line = (f"   [domain={s.get('domain', '-')} score={score} threshold={settings.CONFIDENCE_THRESHOLD} "
            f"escalation={s.get('escalation_reason') or 'none'} sources={s.get('sources', [])[:4]}")
    return line + "]"


def run_turn(state: dict, domain: str) -> dict:
    """One knowledge-base turn (with IT <-> HR hand-off). Ticket offers are shown but not acted on."""
    out = AGENTS[domain](state)
    if out.get("reroute_to"):
        print(f"   [rerouted {domain} -> {out['reroute_to']}]")
        out = AGENTS[out["reroute_to"]]({**out, "reroute_to": None})
    return out


def interactive(domain: str):
    print(f"SmartDesk {domain} agent – type 'exit' to quit, 'reset' to clear history.\n")
    state = {"messages": []}
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("exit", "quit"):
            break
        if text.lower() == "reset":
            state = {"messages": []}
            continue
        if not text:
            continue
        state["messages"].append({"role": "user", "content": text})
        state = run_turn(state, domain)
        print(f"\nSmartDesk: {state['messages'][-1]['content']}")
        print(debug_line(state) + "\n")


def evaluate():
    queries = json.loads((ROOT / "evaluation" / "eval_queries.json").read_text(encoding="utf-8"))
    passed = 0
    rows = []
    for q in queries:
        state = {"messages": [{"role": "user", "content": q["question"]}]}
        if q["expected"] == "out_of_scope":
            out = smalltalk_agent(state)
            ok, got = True, "smalltalk"
        else:
            out = run_turn(state, q["domain"])
            got = "escalate" if out.get("needs_escalation") else "answer"
            ok = got == q["expected"]
            if ok and got == "answer" and q.get("source_doc"):
                ok = q["source_doc"] in out.get("sources", [])
        passed += ok
        score = out.get("confidence")
        rows.append((ok, q["expected"], got, score, q["question"], out["messages"][-1]["content"]))
        print(f"{'PASS' if ok else 'FAIL'}  expected={q['expected']:<12} got={got:<9} "
              f"score={score if score is None else round(score, 3)!s:<6} {q['question']}")
    print(f"\n{passed}/{len(queries)} passed  (CONFIDENCE_THRESHOLD={settings.CONFIDENCE_THRESHOLD})")
    ans = [r[3] for r in rows if r[1] == "answer" and r[3] is not None]
    esc = [r[3] for r in rows if r[1] == "escalate" and r[3] is not None]
    if ans and esc:
        print(f"Answerable questions score {min(ans):.3f}–{max(ans):.3f}; gap questions score "
              f"{min(esc):.3f}–{max(esc):.3f}. Set CONFIDENCE_THRESHOLD between the two ranges.")
    fails = [r for r in rows if not r[0]]
    if fails:
        print("\nFailed replies:")
        for _, exp, got, _, q, reply in fails:
            print(f"- {q}\n  expected {exp}, got {got}: {reply[:200]}")


def main():
    p = argparse.ArgumentParser(description="Try the SmartDesk IT/HR agents.")
    p.add_argument("--domain", choices=["it", "hr"], default="it")
    p.add_argument("--eval", action="store_true", help="Run evaluation/eval_queries.json")
    args = p.parse_args()
    evaluate() if args.eval else interactive(args.domain.upper())


if __name__ == "__main__":
    main()
