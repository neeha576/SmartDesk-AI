"""Evaluation pipeline: metric maths, golden-dataset integrity and an end-to-end run with stand-ins."""
import json
import re
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from evaluation import metrics as M
from evaluation import run_eval as R
from evaluation.judge import JudgeVerdict, judge_answer

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- metrics
def test_reciprocal_rank_and_hits():
    ranked = ["IT-004", "IT-001", "IT-003"]
    assert M.reciprocal_rank(ranked, ["IT-001"]) == 0.5
    assert M.reciprocal_rank(ranked, ["HR-001"]) == 0.0
    assert M.reciprocal_rank(ranked, ["IT-003", "IT-001"]) == 0.5          # first relevant counts
    assert M.hit_at_k(ranked, ["IT-001"], 1) == 0.0 and M.hit_at_k(ranked, ["IT-001"], 3) == 1.0
    assert M.recall_at_k(ranked, ["IT-001", "IT-007"], 3) == 0.5
    assert M.dedupe(["A", "B", "A", "C"]) == ["A", "B", "C"]


def test_token_f1_ignores_case_punctuation_articles_and_sources():
    ref = "Every other Friday – 26 paychecks a year."
    assert M.token_f1("every other friday, 26 paychecks a year\nSource: PAY-001", ref)["f1"] == 1.0
    partial = M.token_f1("Every other Friday.", ref)
    assert partial["precision"] == 1.0 and 0 < partial["recall"] < 1 and 0 < partial["f1"] < 1
    assert M.token_f1("", ref)["f1"] == 0.0 and M.token_f1("bananas", ref)["f1"] == 0.0


def test_key_fact_recall_is_format_tolerant():
    assert M.key_fact_recall("You'd get $10,800 (1-year cliff)", ["10800", "1 year cliff"]) == 1.0
    assert M.key_fact_recall("Ask HR.", ["30 days", "manager"]) == 0.0
    assert M.key_fact_recall("anything", []) is None


def test_classification_report():
    rep = M.classification_report(["answer", "answer", "escalate", "decline"],
                                  ["answer", "escalate", "escalate", "decline"],
                                  ["answer", "escalate", "decline"])
    assert rep["accuracy"] == 0.75
    assert rep["per_class"]["answer"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2}
    assert rep["per_class"]["escalate"]["precision"] == 0.5 and rep["per_class"]["escalate"]["recall"] == 1.0
    assert rep["confusion"]["answer"]["escalate"] == 1
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


# ----------------------------------------------------------------------------- golden dataset integrity
def _kb_doc_ids():
    ids = set()
    for p in (ROOT / "knowledge_base").rglob("*.md"):
        if p.name in ("README.md", "GAPS.md"):
            continue
        m = re.search(r"^doc_id:\s*([A-Z]+-\d+)", p.read_text(encoding="utf-8"), re.M)
        if m:
            ids.add(m.group(1))
    return ids


def test_golden_dataset_is_consistent():
    data = json.loads((ROOT / "evaluation" / "golden_dataset.json").read_text(encoding="utf-8"))
    items, kb = data["items"], _kb_doc_ids()
    assert len(items) >= 50 and len({i["id"] for i in items}) == len(items)
    for it in items:
        assert it["expected_behavior"] in R.LABELS
        if it["expected_behavior"] == "answer":
            assert it["relevant_doc_ids"] and set(it["relevant_doc_ids"]) <= kb, it["id"]
            assert it["key_facts"], it["id"]
            for f in it["key_facts"]:
                assert M.normalize_fact(f) in M.normalize_fact(it["reference_answer"]), (it["id"], f)
        else:
            assert not it["relevant_doc_ids"]
    behaviours = {i["expected_behavior"] for i in items}
    assert behaviours == {"answer", "escalate", "decline"}
    assert {i["domain"] for i in items if i["expected_behavior"] == "answer"} == {"IT", "HR"}


# ----------------------------------------------------------------------------- end to end with stand-ins
ITEMS = [
    {"id": "A1", "question": "When is payday?", "domain": "HR", "category": "factual", "expected_behavior": "answer",
     "relevant_doc_ids": ["PAY-001"], "reference_answer": "Every other Friday – 26 paychecks a year.",
     "key_facts": ["every other friday", "26"]},
    {"id": "G1", "question": "Where can I park?", "domain": "HR", "category": "kb_gap",
     "expected_behavior": "escalate", "relevant_doc_ids": [], "reference_answer": "-", "key_facts": []},
    {"id": "O1", "question": "Weather?", "domain": None, "category": "out_of_scope",
     "expected_behavior": "decline", "relevant_doc_ids": [], "reference_answer": "-", "key_facts": []},
]


class FakeChat:
    SCRIPT = {
        "When is payday?": ("Payday is every other Friday – 26 paychecks a year.\nSource: PAY-001",
                            {"route": "hr_agent", "retrieved_chunks": [{"doc_id": "PAY-001", "text": "Every other Friday"}]},
                            None),
        "Where can I park?": ("I couldn't find that… raise an HR ticket? (yes / no)",
                              {"route": "hr_agent", "needs_escalation": True, "escalation_reason": "low_confidence"},
                              "ticket_offer"),
        "Weather?": ("I can only help with IT and HR questions.", {"route": "smalltalk_agent"}, None),
    }

    def send(self, text):
        self._reply, self.state, self._waiting = self.SCRIPT[text]
        return self._reply

    def waiting_for(self):
        return self._waiting


def fake_search(question, domain=None, k=5):
    return {"chunks": [{"doc_id": "PAY-003"}, {"doc_id": "PAY-001"}, {"doc_id": "PAY-001"}], "top_score": 0.6}


def test_end_to_end_with_stand_ins(tmp_path):
    seen = {}

    def fake_judge(messages):
        seen["prompt"] = messages[1].content
        return JudgeVerdict(correctness=5, faithfulness=5, relevance=4, hallucination=False, reasoning="ok")

    rows = R.evaluate(ITEMS, search=fake_search, make_chat=FakeChat, judge=RunnableLambda(fake_judge),
                      k=5, log=lambda *_: None)
    s = R.summarize(rows, k=5)

    assert s["retrieval"]["mrr@5"] == 0.5 and s["retrieval"]["hit@1"] == 0.0 and s["retrieval"]["hit@3"] == 1.0
    assert s["behavior"]["accuracy"] == 1.0 and s["behavior"]["macro_f1"] == 1.0
    assert s["answer"]["key_fact_recall"] == 1.0 and s["answer"]["token_f1"] > 0.7
    assert s["judge"]["correctness"] == 5 and s["judge"]["hallucination_rate"] == 0.0
    assert "Source:" not in seen["prompt"].split("ASSISTANT ANSWER:")[1]       # citations stripped for the judge
    assert "[PAY-001] Every other Friday" in seen["prompt"]                       # context passed to the judge

    report = R.render_report(s, rows, {"timestamp": "t", "dataset": "golden", "judge_model": "fake",
                                       "with_cache": False})
    assert "MRR@5 | **0.500**" in report and "Accuracy **100.0%**" in report and "None 🎉" in report


def test_wrong_behaviour_and_judge_failures_are_reported():
    class Unsure(FakeChat):
        SCRIPT = {**FakeChat.SCRIPT, "When is payday?": ("I couldn't find that… (yes / no)",
                                                        {"route": "hr_agent", "needs_escalation": True,
                                                         "escalation_reason": "low_confidence"}, "ticket_offer")}

    rows = R.evaluate(ITEMS, search=fake_search, make_chat=Unsure,
                      judge=RunnableLambda(lambda m: (_ for _ in ()).throw(TimeoutError("down"))),
                      k=5, log=lambda *_: None)
    s = R.summarize(rows, k=5)
    assert s["behavior"]["per_class"]["answer"]["recall"] == 0.0
    assert s["behavior"]["confusion"]["answer"]["escalate"] == 1
    assert "answer" not in s                                                    # nothing answered to grade
    report = R.render_report(s, rows, {"timestamp": "t", "dataset": "g", "judge_model": None, "with_cache": False})
    assert "**A1**" in report and "expected answer, got escalate (low_confidence)" in report


def test_judge_errors_do_not_stop_the_run():
    out = judge_answer(RunnableLambda(lambda m: (_ for _ in ()).throw(TimeoutError("down"))), "q", "r", "c", "a")
    assert "error" in out


def test_load_items_filters():
    assert [i["id"] for i in R.load_items(ids=["IT-01", "GAP-01"])] == ["IT-01", "GAP-01"]
    assert len(R.load_items(limit=3)) == 3


def test_preflight_collects_failures():
    def boom():
        raise RuntimeError("LLM_MODEL is not set")
    assert R.preflight({"ok": lambda: 1}) == []
    assert R.preflight({"ok": lambda: 1, "Chat LLM": boom}) == ["Chat LLM: LLM_MODEL is not set"]
