"""Evaluate SmartDesk AI on the golden dataset: retrieval (MRR), F1 scores and LLM-as-judge.

    python evaluation/run_eval.py                     # everything (retrieval + agent + judge)
    python evaluation/run_eval.py --retrieval-only    # MRR / Hit@k / Recall@k only (embeddings, no chat LLM)
    python evaluation/run_eval.py --no-judge          # skip the LLM judge
    python evaluation/run_eval.py --limit 10          # quick smoke run
    python evaluation/run_eval.py --ids IT-01 GAP-03  # selected items

What is measured
  1. Retrieval (items with relevant_doc_ids): the question goes through the same hybrid search the agents use
     (Qdrant dense + BM25, domain filter). Ranked document ids → MRR@k, Hit@1/3/k, Recall@k.
  2. Behaviour (all items): each question is sent to the FULL graph (router → agent) in a fresh conversation.
     Predicted behaviour: answer | escalate (ticket offered) | decline (small talk / out of scope).
     → accuracy, precision / recall / F1 per class, macro-F1, confusion matrix.
  3. Answer quality (answerable items the agent answered):
     - token F1 against the reference answer (SQuAD-style; "Source:" lines removed),
     - key-fact recall (share of must-have facts present),
     - LLM-as-judge: correctness, faithfulness (grounding), relevance (1-5) and a hallucination flag.
  4. Latency per question (p50 / p95).

The semantic cache is OFF during evaluation (so every answer is generated); add --with-cache to include it.
No tickets are created: escalation pauses at the "raise a ticket?" question and the run moves on.

Output: evaluation/results/<timestamp>/ with results.json (every item), summary.json and report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import metrics as M  # noqa: E402

DATASET = ROOT / "evaluation" / "golden_dataset.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
LABELS = ["answer", "escalate", "decline"]
WAITING_ESCALATE = {"ticket_offer", "email", "ticket_details", "ticket_confirmation"}


# ----------------------------------------------------------------------------- loading
def load_items(path: Path = DATASET, ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    if ids:
        items = [i for i in items if i["id"] in set(ids)]
    return items[:limit] if limit else items


# ----------------------------------------------------------------------------- 1. retrieval
def evaluate_retrieval(item: dict, search, k: int) -> dict | None:
    """search(question, domain, k) -> {'chunks': [...], 'top_score': float}. None for items without gold docs."""
    if not item.get("relevant_doc_ids"):
        return None
    try:
        res = search(item["question"], domain=item.get("domain"), k=k)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    ranked = M.dedupe(c["doc_id"] for c in res["chunks"])
    rel = item["relevant_doc_ids"]
    return {"ranked_doc_ids": ranked, "top_score": res.get("top_score"),
            "rr": M.reciprocal_rank(ranked, rel), "hit@1": M.hit_at_k(ranked, rel, 1),
            "hit@3": M.hit_at_k(ranked, rel, 3), f"hit@{k}": M.hit_at_k(ranked, rel, k),
            f"recall@{k}": M.recall_at_k(ranked, rel, k)}


# ----------------------------------------------------------------------------- 2. behaviour
def predicted_behavior(state: dict, waiting: str | None) -> str:
    if waiting in WAITING_ESCALATE or state.get("needs_escalation") or state.get("route") == "ticket_agent":
        return "escalate"
    if state.get("route") == "smalltalk_agent":
        return "decline"
    if state.get("route") in ("it_agent", "hr_agent", "parallel"):
        return "answer"
    return "other"                                     # e.g. routed to the status agent


def run_agent(item: dict, make_chat) -> dict:
    """Send the question to a fresh conversation; return reply, behaviour, context and latency."""
    chat = make_chat()
    t0 = time.perf_counter()
    try:
        reply = chat.send(item["question"])
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "predicted": "error", "latency_s": time.perf_counter() - t0}
    latency = time.perf_counter() - t0
    state, waiting = chat.state, chat.waiting_for()
    chunks = state.get("retrieved_chunks") or []
    return {
        "reply": reply, "predicted": predicted_behavior(state, waiting), "latency_s": latency,
        "route": state.get("route"), "waiting_for": waiting, "confidence": state.get("confidence"),
        "escalation_reason": state.get("escalation_reason"), "sources": state.get("sources") or [],
        "context": "\n\n".join(f"[{c.get('doc_id')}] {c.get('text', '')}" for c in chunks),
    }


# ----------------------------------------------------------------------------- 3. answer quality
def evaluate_answer(item: dict, run: dict, judge=None) -> dict | None:
    if item["expected_behavior"] != "answer" or run.get("predicted") != "answer":
        return None
    out = {"token": M.token_f1(run["reply"], item["reference_answer"]),
           "key_fact_recall": M.key_fact_recall(run["reply"], item.get("key_facts", []))}
    if judge is not None:
        from evaluation.judge import judge_answer
        out["judge"] = judge_answer(judge, item["question"], item["reference_answer"],
                                    run.get("context", ""), M.strip_sources(run["reply"]))
    return out


# ----------------------------------------------------------------------------- run + summary
def evaluate(items: list[dict], search=None, make_chat=None, judge=None, k: int = 5, log=print) -> list[dict]:
    rows = []
    for n, item in enumerate(items, 1):
        row = {"id": item["id"], "question": item["question"], "domain": item.get("domain"),
               "category": item["category"], "expected": item["expected_behavior"],
               "relevant_doc_ids": item.get("relevant_doc_ids", [])}
        if search is not None:
            row["retrieval"] = evaluate_retrieval(item, search, k)
        if make_chat is not None:
            row["agent"] = run_agent(item, make_chat)
            row["answer_quality"] = evaluate_answer(item, row["agent"], judge)
        log(_progress_line(n, len(items), row, k))
        rows.append(row)
    return rows


def summarize(rows: list[dict], k: int) -> dict:
    s: dict = {"n_items": len(rows), "k": k}

    ret = [r["retrieval"] for r in rows if r.get("retrieval") and "error" not in r["retrieval"]]
    if ret:
        s["retrieval"] = {"n": len(ret), f"mrr@{k}": M.mean(x["rr"] for x in ret),
                          "hit@1": M.mean(x["hit@1"] for x in ret), "hit@3": M.mean(x["hit@3"] for x in ret),
                          f"hit@{k}": M.mean(x[f"hit@{k}"] for x in ret),
                          f"recall@{k}": M.mean(x[f"recall@{k}"] for x in ret)}
        for dom in ("IT", "HR"):
            sub = [r["retrieval"] for r in rows if r.get("retrieval") and "error" not in r["retrieval"]
                   and r["domain"] == dom]
            if sub:
                s["retrieval"][f"mrr@{k}_{dom}"] = M.mean(x["rr"] for x in sub)

    runs = [r for r in rows if r.get("agent")]
    if runs:
        y_true = [r["expected"] for r in runs]
        y_pred = [r["agent"]["predicted"] for r in runs]
        s["behavior"] = M.classification_report(y_true, y_pred, LABELS)
        lat = [r["agent"]["latency_s"] for r in runs]
        s["latency_s"] = {"mean": M.mean(lat), "p50": M.percentile(lat, 50), "p95": M.percentile(lat, 95)}

    aq = [r["answer_quality"] for r in rows if r.get("answer_quality")]
    if aq:
        s["answer"] = {"n": len(aq),
                       "token_precision": M.mean(a["token"]["precision"] for a in aq),
                       "token_recall": M.mean(a["token"]["recall"] for a in aq),
                       "token_f1": M.mean(a["token"]["f1"] for a in aq),
                       "key_fact_recall": M.mean(a["key_fact_recall"] for a in aq)}
        verdicts = [a["judge"] for a in aq if a.get("judge") and "error" not in a["judge"]]
        if verdicts:
            s["judge"] = {"n": len(verdicts),
                          "correctness": M.mean(v["correctness"] for v in verdicts),
                          "faithfulness": M.mean(v["faithfulness"] for v in verdicts),
                          "relevance": M.mean(v["relevance"] for v in verdicts),
                          "pass_rate_correctness>=4": M.mean(float(v["correctness"] >= 4) for v in verdicts),
                          "hallucination_rate": M.mean(float(v["hallucination"]) for v in verdicts),
                          "errors": sum(1 for a in aq if a.get("judge") and "error" in a["judge"])}
    return s


# ----------------------------------------------------------------------------- report
def _pct(v):
    return "–" if v is None else f"{v * 100:.1f}%"


def _num(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def _progress_line(n, total, row, k):
    parts = [f"[{n:>2}/{total}] {row['id']:<7}"]
    if row.get("retrieval") and "rr" in row["retrieval"]:
        parts.append(f"RR={row['retrieval']['rr']:.2f}")
    if row.get("agent"):
        a = row["agent"]
        ok = "✓" if a["predicted"] == row["expected"] else "✗"
        parts.append(f"{ok} expected={row['expected']} got={a['predicted']} ({a['latency_s']:.1f}s)")
    aq = row.get("answer_quality")
    if aq:
        parts.append(f"F1={aq['token']['f1']:.2f}")
        j = aq.get("judge") or {}
        if "correctness" in j:
            parts.append(f"judge C/F/R={j['correctness']}/{j['faithfulness']}/{j['relevance']}"
                         + (" HALLUC" if j["hallucination"] else ""))
    return "  ".join(parts)


def render_report(summary: dict, rows: list[dict], meta: dict) -> str:
    k = summary["k"]
    L = ["# SmartDesk AI – evaluation report", "",
         f"Run: {meta['timestamp']} · dataset: `{meta['dataset']}` ({summary['n_items']} items) · "
         f"k={k} · judge: {meta['judge_model'] or 'off'} · semantic cache: {'on' if meta['with_cache'] else 'off'}", ""]

    if r := summary.get("retrieval"):
        L += ["## Retrieval (hybrid search, domain-filtered)", "",
              "| Metric | Value |", "|---|---|",
              f"| MRR@{k} | **{_num(r[f'mrr@{k}'])}** |",
              f"| MRR@{k} – IT / HR | {_num(r.get(f'mrr@{k}_IT'))} / {_num(r.get(f'mrr@{k}_HR'))} |",
              f"| Hit@1 | {_pct(r['hit@1'])} |", f"| Hit@3 | {_pct(r['hit@3'])} |",
              f"| Hit@{k} | {_pct(r[f'hit@{k}'])} |", f"| Recall@{k} | {_pct(r[f'recall@{k}'])} |",
              f"| Queries | {r['n']} |", ""]

    if b := summary.get("behavior"):
        L += ["## Behaviour: answer vs. escalate vs. decline", "",
              f"Accuracy **{_pct(b['accuracy'])}** · macro-F1 **{_num(b['macro_f1'])}**", "",
              "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for lab, m in b["per_class"].items():
            L.append(f"| {lab} | {_pct(m['precision'])} | {_pct(m['recall'])} | **{_num(m['f1'])}** | {m['support']} |")
        seen = {p for row in b["confusion"].values() for p in row}
        cols = [c for c in LABELS if c in seen] + sorted(seen - set(LABELS))
        L += ["", "Confusion matrix (rows = expected, columns = predicted):", "",
              "| expected \\ predicted | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for t, row in b["confusion"].items():
            L.append(f"| {t} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
        L.append("")

    if a := summary.get("answer"):
        L += ["## Answer quality (answerable questions that were answered)", "",
              "| Metric | Value |", "|---|---|",
              f"| Token F1 vs. reference | **{_num(a['token_f1'])}** (P {_num(a['token_precision'])} / "
              f"R {_num(a['token_recall'])}) |",
              f"| Key-fact recall | {_pct(a['key_fact_recall'])} |", f"| Answers graded | {a['n']} |"]
        if j := summary.get("judge"):
            L += [f"| LLM judge – correctness (1-5) | **{_num(j['correctness'], 2)}** |",
                  f"| LLM judge – faithfulness / grounding (1-5) | **{_num(j['faithfulness'], 2)}** |",
                  f"| LLM judge – relevance (1-5) | {_num(j['relevance'], 2)} |",
                  f"| Correct (judge score ≥ 4) | {_pct(j['pass_rate_correctness>=4'])} |",
                  f"| Hallucination rate | {_pct(j['hallucination_rate'])} |"]
        L.append("")

    if lat := summary.get("latency_s"):
        L += [f"**Latency per question:** mean {_num(lat['mean'], 2)} s · p50 {_num(lat['p50'], 2)} s · "
              f"p95 {_num(lat['p95'], 2)} s", ""]

    fails = []
    for r in rows:
        why = []
        if r.get("agent") and r["agent"]["predicted"] != r["expected"]:
            why.append(f"expected {r['expected']}, got {r['agent']['predicted']}"
                       + (f" ({r['agent'].get('escalation_reason')})" if r["agent"].get("escalation_reason") else ""))
        if r.get("retrieval") and r["retrieval"].get("rr") == 0:
            why.append(f"relevant doc not in top {k} (got {', '.join(r['retrieval']['ranked_doc_ids'][:3]) or 'nothing'})")
        j = ((r.get("answer_quality") or {}).get("judge") or {})
        if j.get("correctness", 5) <= 2 or j.get("hallucination"):
            why.append(f"judge: {j.get('reasoning', '')}")
        if why:
            fails.append(f"- **{r['id']}** – {r['question']}  \n  " + "; ".join(why))
    L += ["## Items to review", ""] + (fails or ["None 🎉"]) + [""]
    L += ["## How to read this",
          "- **MRR** (mean reciprocal rank): 1.0 = the right document is always ranked first; 0.5 = second on average.",
          "- **Behaviour F1**: the *escalate* F1 shows how reliably SmartDesk knows when it doesn't know; the "
          "*answer* recall shows how often it answers what it should.",
          "- **Token F1** compares words with the reference; full-sentence LLM answers score lower than their "
          "real quality, so read it alongside key-fact recall and the judge's correctness.",
          "- **Faithfulness** and the **hallucination rate** come from the LLM judge checking each claim against "
          "the retrieved context."]
    return "\n".join(L)


# ----------------------------------------------------------------------------- pre-flight
def preflight(checks: dict) -> list[str]:
    """Run each check (name -> callable); return error messages. Infrastructure problems (missing .env value,
    no index, API down) must stop the run – otherwise they show up as wrong answers and distort the metrics."""
    errors = []
    for name, check in checks.items():
        try:
            check()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    return errors


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate SmartDesk AI on the golden dataset.")
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--k", type=int, default=None, help="retrieval depth (default RETRIEVAL_TOP_K)")
    p.add_argument("--limit", type=int, help="only the first N items")
    p.add_argument("--ids", nargs="*", help="only these item ids")
    p.add_argument("--retrieval-only", action="store_true", help="MRR / Hit@k / Recall@k only")
    p.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge")
    p.add_argument("--with-cache", action="store_true", help="keep the semantic cache on")
    p.add_argument("--out", default=str(RESULTS_DIR))
    p.add_argument("--skip-preflight", action="store_true", help="don't check embeddings / LLM / index first")
    args = p.parse_args(argv)
    try:                                   # Windows consoles / redirected output: never crash on ✓ ✗
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    from config import settings
    from tools.kb_search import kb_search
    k = args.k or settings.RETRIEVAL_TOP_K
    if not args.with_cache:
        settings.SEMANTIC_CACHE_ENABLED = False

    items = load_items(Path(args.dataset), args.ids, args.limit)
    if not args.skip_preflight:
        checks = {"Knowledge-base search (embeddings + Qdrant index)":
                  lambda: kb_search("How do I reset my password?", domain="IT", k=1)["chunks"][0]}
        if not args.retrieval_only:
            from agents.llm import get_llm
            checks["Chat LLM"] = lambda: get_llm().invoke("Reply with OK.")
        print("Pre-flight checks…")
        if errors := preflight(checks):
            print("\nCannot evaluate – fix these first (nothing was scored):")
            for e in errors:
                print(f"  ✗ {e}")
            print("\nTip: check .env, run `python scripts/build_index.py`, and make sure the Streamlit app "
                  "isn't holding the local Qdrant folder open.")
            raise SystemExit(2)
    make_chat = judge = None
    judge_model = None
    if not args.retrieval_only:
        from agents.graph import SmartDeskChat, build_graph
        graph = build_graph()
        make_chat = lambda: SmartDeskChat(graph)   # noqa: E731 – fresh thread per question
        if not args.no_judge:
            import os
            from evaluation.judge import get_judge
            judge, judge_model = get_judge(), os.getenv("EVAL_JUDGE_MODEL") or settings.llm_model()

    print(f"Evaluating {len(items)} items (k={k}, judge={'on' if judge else 'off'}, "
          f"cache={'on' if args.with_cache else 'off'})\n")
    rows = evaluate(items, search=kb_search, make_chat=make_chat, judge=judge, k=k)
    summary = summarize(rows, k)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / stamp
    out.mkdir(parents=True, exist_ok=True)
    meta = {"timestamp": stamp, "dataset": Path(args.dataset).name, "judge_model": judge_model,
            "with_cache": args.with_cache}
    (out / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({**meta, **summary}, indent=2), encoding="utf-8")
    report = render_report(summary, rows, meta)
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
