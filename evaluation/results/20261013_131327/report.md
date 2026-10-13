# SmartDesk AI – evaluation report

Run: 20261013_131327 · dataset: `golden_dataset.json` (73 items) · k=5 · judge: off · semantic cache: off

## Retrieval (hybrid search, domain-filtered)

| Metric | Value |
|---|---|
| MRR@5 | **0.938** |
| MRR@5 – IT / HR | 0.950 / 0.931 |
| Hit@1 | 87.5% |
| Hit@3 | 100.0% |
| Hit@5 | 100.0% |
| Recall@5 | 99.1% |
| Queries | 56 |

## Items to review

None 🎉

## How to read this
- **MRR** (mean reciprocal rank): 1.0 = the right document is always ranked first; 0.5 = second on average.
- **Behaviour F1**: the *escalate* F1 shows how reliably SmartDesk knows when it doesn't know; the *answer* recall shows how often it answers what it should.
- **Token F1** compares words with the reference; full-sentence LLM answers score lower than their real quality, so read it alongside key-fact recall and the judge's correctness.
- **Faithfulness** and the **hallucination rate** come from the LLM judge checking each claim against the retrieved context.