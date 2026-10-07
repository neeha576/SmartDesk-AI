# SmartDesk Knowledge Base – NMTech

Fictional company: **NMTech**, a 500-employee SaaS company headquartered in **Albuquerque, New Mexico**. All amounts are in U.S. dollars and times are in Mountain Time (MT).
All content is synthetic and written for this project. Numbers are consistent across documents and Q&A pairs (see *Key figures* below).

## Contents
| Folder | Docs | Domain label | Topics |
|---|---|---|---|
| `it/` | 7 | IT | Password reset, VPN, MFA, software requests, Wi-Fi/guest Wi-Fi, mobile email, IT service desk |
| `hr/` | 7 | HR | Time off, remote/hybrid work, expenses, code of conduct, anti-harassment, performance review, HR contacts |
| `onboarding/` | 5 | HR (IT for `it_equipment.md`) | Day-one checklist, IT equipment, benefits, buddy program, introductory period |
| `payroll/` | 4 | HR | Payday, W-4/W-2 tax forms, compensation structure, stock option vesting |
| `qa_pairs/` | 75 pairs | IT / HR | `it_qa.json` (28), `hr_qa.json` (47) |
| `GAPS.md` | – | – | Deliberate gaps, near-misses and out-of-scope test questions |

**Totals:** 23 policy documents + 75 Q&A pairs.

## Key figures (single source of truth)
| Item | Value | Document |
|---|---|---|
| PTO | 15 days/yr, 1.25 days/month, carry over 5, cash out 5 | HR-001 |
| Sick leave | 8 days (64 hours)/yr, carry over 64 hours, never paid out | HR-001 |
| Company / floating holidays | 10 / 2 | HR-001 |
| Paid parental leave (maternity / paternity) | 12 weeks birthing / 6 weeks non-birthing | HR-001 |
| Internet stipend | $50 per month | HR-002, HR-003 |
| Home office stipend | $500 one-time, after 90 days | HR-002, HR-003, ONB-005 |
| Receipt required | Over $25 | HR-003 |
| Client meals | $75 per person | HR-003 |
| Mileage | Current IRS standard rate | HR-003 |
| Learning budget | $2,000 per year | HR-003 |
| Vendor gift limit | $100 per year | HR-004 |
| Buddy lunch | $50 per pair | ONB-004 |
| HSA employer contribution | $1,000 single / $2,000 family | ONB-003 |
| Life insurance | 2 × base salary, up to $500,000 | ONB-003 |
| 401(k) match | 100% of first 4%, vests immediately | ONB-003, PAY-003 |
| Target bonus | 10% of base, paid by March 15 | PAY-003, HR-006 |
| Payday | Every other Friday (26 per year) | PAY-001 |
| Introductory period | 90 days (+30 max extension) | ONB-005 |
| IT hours | Mon–Fri 7:00 AM – 7:00 PM MT, ext. 2020 | IT-007 |

## Document format
Each Markdown file starts with YAML front matter that the loader stores as Qdrant payload:
```yaml
doc_id: IT-002          # stable ID, cited in answers
title: VPN Setup and Troubleshooting
domain: IT              # IT | HR  -> used by the IT / HR sub-agents as a filter
category: IT Support    # IT Support | HR Policies | Onboarding | Payroll and Compensation
owner: IT Network Team
last_updated: 2026-06-15
```
Bodies use `##` headings for each sub-topic, so the chunker can split on headings and keep each chunk focused.

## Q&A pair format
```json
{
  "id": "HR-QA-003",
  "question": "Can I carry over unused sick leave and get paid for it?",
  "answer": "...",
  "domain": "HR",
  "topic": "Time Off",
  "question_type": "policy_interpretation",   // factual | procedural | policy_interpretation
  "source_doc": "HR-001"                      // doc_id the answer comes from
}
```
Each Q&A pair is indexed as one whole chunk. `source_doc` doubles as ground truth for retrieval evaluation.
