"""Load the knowledge base: Markdown policy documents (with YAML front matter) and Q&A pair JSON files."""
from __future__ import annotations

import json
import re
from pathlib import Path

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Split simple `key: value` front matter from the Markdown body."""
    match = FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def load_markdown_docs(kb_dir: str | Path) -> list[dict]:
    """Return one record per Markdown policy file: {'text', 'metadata'}.

    README.md and GAPS.md are skipped – they describe the KB, they are not KB content
    (indexing GAPS.md would make the deliberate gaps answerable).
    """
    kb_dir = Path(kb_dir)
    docs = []
    for path in sorted(kb_dir.rglob("*.md")):
        if path.parent == kb_dir:          # top-level README.md / GAPS.md
            continue
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        if not meta.get("doc_id"):
            raise ValueError(f"{path} is missing 'doc_id' in its front matter")
        docs.append({
            "text": body.strip(),
            "metadata": {
                "doc_id": meta["doc_id"],
                "title": meta.get("title", path.stem),
                "domain": meta.get("domain", "HR"),
                "category": meta.get("category", ""),
                "source": path.relative_to(kb_dir).as_posix(),
                "chunk_type": "policy",
            },
        })
    return docs


def load_qa_pairs(kb_dir: str | Path) -> list[dict]:
    """Return one record per Q&A pair; each pair is kept whole as a single chunk."""
    records = []
    for path in sorted(Path(kb_dir, "qa_pairs").glob("*.json")):
        for qa in json.loads(path.read_text(encoding="utf-8")):
            records.append({
                "text": f"Q: {qa['question']}\nA: {qa['answer']}",
                "metadata": {
                    "doc_id": qa["source_doc"],
                    "qa_id": qa["id"],
                    "title": qa["question"],
                    "domain": qa["domain"],
                    "category": qa.get("topic", ""),
                    "question_type": qa.get("question_type", ""),
                    "source": f"qa_pairs/{path.name}",
                    "chunk_type": "qa",
                },
            })
    return records
