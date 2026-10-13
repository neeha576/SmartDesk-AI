"""Evaluation metrics (pure functions – no LLM, no network).

Retrieval   reciprocal_rank / MRR, hit_at_k, recall_at_k   – over ranked document ids
Answers     token_f1 (SQuAD-style precision / recall / F1), key_fact_recall
Behaviour   classification_report – precision / recall / F1 per class (answer | escalate | decline),
            accuracy, macro-F1 and a confusion matrix
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# ----------------------------------------------------------------------------- text helpers
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SOURCE_LINE = re.compile(r"^\s*(sources?|\[?source)\b.*$", re.IGNORECASE | re.MULTILINE)


def strip_sources(text: str) -> str:
    """Drop 'Source: IT-001' lines so citations don't count as answer tokens."""
    return _SOURCE_LINE.sub("", text or "").strip()


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lowercase, remove punctuation and articles, collapse whitespace."""
    text = (text or "").lower()
    # any Unicode punctuation (incl. – — ’ “) becomes a space
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def normalize_fact(text: str) -> str:
    """Looser normalisation for key facts: '$10,800' == '10800', '1-year' == '1 year'."""
    t = (text or "").lower().replace("$", "").replace(",", "").replace("–", " ").replace("‑", " ")
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)
    return re.sub(r"\s+", " ", t)


# ----------------------------------------------------------------------------- answer quality
def token_f1(prediction: str, reference: str) -> dict:
    """Token-overlap precision / recall / F1 between an answer and the reference answer."""
    pred = normalize_answer(strip_sources(prediction)).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        same = float(pred == ref)
        return {"precision": same, "recall": same, "f1": same}
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p, r = common / len(pred), common / len(ref)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r)}


def key_fact_recall(answer: str, facts: list[str]) -> float | None:
    """Share of the must-have facts found in the answer (None when the item has no key facts)."""
    if not facts:
        return None
    a = normalize_fact(answer)
    return sum(normalize_fact(f) in a for f in facts) / len(facts)


# ----------------------------------------------------------------------------- retrieval
def dedupe(seq):
    """Keep the first occurrence of each id, in rank order (several chunks can come from one document)."""
    return list(dict.fromkeys(seq))


def reciprocal_rank(ranked_ids: list[str], relevant: list[str]) -> float:
    """1 / rank of the first relevant document; 0 if none was retrieved. MRR = mean over queries."""
    rel = set(relevant)
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked_ids: list[str], relevant: list[str], k: int) -> float:
    return float(bool(set(ranked_ids[:k]) & set(relevant)))


def recall_at_k(ranked_ids: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    return len(set(ranked_ids[:k]) & rel) / len(rel) if rel else 0.0


# ----------------------------------------------------------------------------- behaviour (classification)
def classification_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Per-class precision / recall / F1 / support, accuracy, macro-F1 and confusion matrix."""
    per_class = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    present = [lab for lab in labels if per_class[lab]["support"]]
    confusion = {t: {p: sum(a == t and b == p for a, b in zip(y_true, y_pred))
                     for p in sorted(set(labels) | set(y_pred))} for t in labels}
    return {
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0,
        "macro_f1": sum(per_class[lab]["f1"] for lab in present) / len(present) if present else 0.0,
        "per_class": per_class,
        "confusion": confusion,
    }


def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(values, pct: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    i = min(len(vals) - 1, max(0, round(pct / 100 * (len(vals) - 1))))
    return vals[i]
