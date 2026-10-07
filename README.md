# SmartDesk AI — IT & HR Help Desk Agent

First-line help desk agent that answers IT/HR questions from a knowledge base,
creates Jira tickets when it can't answer (with confirmation), and checks ticket status.

## Project structure
```
smartdesk-ai/
├── agents/            # LangGraph graph, orchestrator, IT/HR/Ticket/Smalltalk/Status agents, prompts, LCEL RAG chain, LLM client
├── tools/             # ALL agent tools + registry (which agent binds which tools), Jira client, KB search, ticket store
├── rag/               # loader, chunker, embeddings, Qdrant store, hybrid retriever, LangChain retriever, confidence gates
├── knowledge_base/    # Markdown policies (it/, hr/, onboarding/, payroll/) + Q&A JSON
├── memory/            # Session memory, semantic cache
├── config/            # Settings from env vars, logging
├── ui/                # Streamlit app
├── evaluation/        # Eval queries + Ragas runner
├── scripts/           # build_index, try_agent, test_jira, seed_tickets
├── tests/             # Unit tests
├── docs/              # Architecture diagram, screenshots
├── data/              # Vector store + SQLite (git-ignored)
└── main.py            # CLI entry point
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # fill in keys
# Qdrant: local file mode works out of the box (QDRANT_PATH).
# For a server instead: docker run -p 6333:6333 qdrant/qdrant  and set QDRANT_URL=http://localhost:6333
python scripts/build_index.py
python main.py          # chat with SmartDesk (add --debug to see routing and scores)
streamlit run ui/streamlit_app.py   # or the web UI
```

## Web UI (Streamlit)
`streamlit run ui/streamlit_app.py` opens SmartDesk in the browser. It uses the same graph as `main.py`
(`SmartDeskChat`), so every button simply sends a message and goes through routing, the knowledge base,
confirmation and the privacy rules as usual.

| Part | What it does |
|---|---|
| **Create ticket** | A form (IT / HR + problem) that asks SmartDesk to raise a ticket; the draft is shown for confirmation first |
| **Check ticket status** | Asks for *your* tickets (needs your email: from the sidebar, or SmartDesk asks) |
| **Popular questions** | IT and HR tabs with common questions from the knowledge base – one click asks |
| **Chat window** | Conversation history and chat input; **Yes / No** buttons appear when SmartDesk asks for a confirmation |
| **Sidebar** | Your work email, new conversation, sample tickets (mock mode), routing details for debugging |

Tests: `tests/test_ui.py` runs the app headless with Streamlit's `AppTest` and a stand-in chat.

## Indexing the knowledge base
```bash
python scripts/build_index.py --dry-run   # load + chunk only, prints chunk stats (no API calls)
python scripts/build_index.py             # embed + upsert into Qdrant, then run sample queries
```
- **Sources:** every Markdown policy under `knowledge_base/*/` plus each Q&A pair in `knowledge_base/qa_pairs/`. `README.md` and `GAPS.md` are not indexed.
- **Chunking:** 500 tokens with 50-token overlap (`cl100k_base` tokenizer), splitting on `##` sections first so chunks end at natural boundaries. Each chunk is prefixed with its document title. Q&A pairs are one chunk each. Override with `CHUNK_SIZE` / `CHUNK_OVERLAP` env vars.
- **Vectors:** dense embeddings from `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`, plus BM25 sparse vectors (`SPARSE_MODEL`) for hybrid search.
- **Payload:** `text`, `doc_id`, `title`, `domain` (IT/HR), `category`, `source`, `chunk_type` (policy/qa), `chunk_index`.
- Re-running rebuilds the collection; point IDs are deterministic, so `--no-recreate` overwrites instead of duplicating.
- In local mode (`QDRANT_PATH`), only one process can open the store at a time: stop the app before re-indexing.

## Agents

SmartDesk is a multi-agent **LangGraph** application (`agents/graph.py`). An orchestrator routes every
employee message to one specialised agent; the conversation state is checkpointed per `thread_id`.

| Agent / node | File | Role |
|---|---|---|
| Orchestrator (`router`) | `agents/orchestrator.py` | Chooses the agent for each message (Pydantic `RouteDecision`) |
| IT agent | `agents/it_agent.py` → `kb_agent.py` | Answers IT questions from the IT slice of the knowledge base |
| HR agent | `agents/hr_agent.py` → `kb_agent.py` | Answers HR questions from the HR slice of the knowledge base |
| Ticket agent (subgraph) | `agents/ticket_graph.py` + `agents/ticket_agent.py` | Collects email, drafts the ticket via `bind_tools`, confirms via `interrupt`, creates it in Jira |
| Smalltalk agent | `agents/smalltalk_agent.py` | Greetings, thanks, out-of-scope questions |
| Synthesizer | `agents/synthesizer.py` | Multi-part messages: the agents answer their parts concurrently (`Send()`), then `synthesizer_node` combines the labelled outputs (`[IT Agent]`, `[Status Agent]` …) into one final answer |
| Status agent | `agents/status_agent.py` | One tool-calling agent (LangChain `create_agent`): looks up the employee's own tickets in Jira, asks which one, reports status + comments |

All prompts live in `agents/prompts.py`; the LLM client (with retries and fallback model) is `agents/llm.py`.

### Orchestrator and graph
- **Routing:** the router LLM returns a Pydantic `RouteDecision(agent, reason)` via structured output.
  `ROUTER_PROMPT` lists every available agent and the routing rules: IT/HR questions and problems always
  go to the IT/HR agent first (knowledge base before tickets); `ticket_agent` only for explicit ticket
  requests; `status_agent` for existing tickets; `smalltalk_agent` for greetings and out-of-scope; history
  is used for follow-ups. If the router LLM is down, keyword routing (`keyword_route`) takes over.
- **State management:** `AgentState` (`agents/state.py`) is the graph state; `messages` uses an append
  reducer that gives every message an id and skips ids it already has (so a subgraph returning the full
  conversation doesn't duplicate it). An in-memory `MemorySaver` checkpointer stores each conversation under its `thread_id`, so the
  employee's email, last ticket and history persist across turns.
- **Graph:** `START → router → {it_agent | hr_agent | ticket_agent | status_agent | smalltalk_agent}`.
  `ticket_agent` is a **subgraph** (`agents/ticket_graph.py`): the whole ticket-creation flow compiled as its own
  graph and added to the main graph as one node. It shares `AgentState` and the parent's checkpointer, so its
  interrupts pause and resume the whole conversation; it can also be run and tested on its own.
  The IT/HR agents route themselves with `Command(goto=...)`: answer → END, can't answer → `offer_ticket`,
  other domain's question → the other agent (once). See [docs/architecture.md](docs/architecture.md).
- **Running it:** `SmartDeskChat(build_graph()).send(text)` starts a new turn, or – if the graph is paused
  at an `interrupt()` – resumes it with `Command(resume=text)`.

### Multi-part messages – parallel agents with `Send()` + synthesizer
For a message like *"hi! how do I reset my password, and any update on my VPN ticket?"* the router's Pydantic
`RouteDecision.tasks` lists the independent parts (max 3), e.g. `[smalltalk: "hi", it_agent: "How do I reset my
password?", status_agent: "any update on my VPN ticket?"]`. The router's conditional edge returns one
**`Send(<agent>, {...})`** per part, straight to that part's agent (`it_agent`, `hr_agent`, `smalltalk_agent`,
`status_agent` – no extra "parallel" node), so the parts run **concurrently**. Each agent sees it was started as a
branch, answers its part and ends with `Command(goto="synthesizer")`, adding `{agent, request, text}` to
`partial_results`; the synthesizer runs once, after every branch has finished. Ticket requests aren't sent as
branches (creating a ticket needs the employee); the synthesizer records them and starts the ticket flow after. The **`synthesizer`** (`agents/synthesizer.py`) then labels each
output by agent (`[IT Agent] (request: …)`), sends `SystemMessage(SYNTHESIZER_PROMPT)` +
`HumanMessage("Employee query: … Agent outputs: …")` to the LLM, and stores the merged reply in `final_answer`
(and `messages`), keeping every fact, link and `Source:` line, in the order of the message. Progress is logged
via `config/logging_config.setup_logger` (`LOG_LEVEL` env, default INFO).

Agents running as parallel branches never stop to ask the employee anything (several branches pausing at once would mean several
questions at once):

| Part | In a parallel branch |
|---|---|
| IT / HR question | Normal knowledge-base answer; if it can't be answered, "couldn't find …" is recorded |
| Small talk | Normal reply |
| Ticket status | Email known: read-only status summary from Jira (same privacy rule). Unknown: flagged |
| Raise a ticket | Flagged (interactive) |

After the synthesized reply, **at most one interactive follow-up** runs, in this order: the ticket request
(`ticket_agent`), a ticket offer for an unanswered part (`offer_ticket`, yes / no), or asking for the email for the
status part (`status_agent`). A failing branch doesn't sink the others; if the synthesizer LLM is down, the answers are
joined as they are. Single-request messages are unaffected.

### Tools and which agent binds them
Every LangChain tool lives in the `tools/` package; `tools/registry.py` lists which tools each agent binds:

| Agent | Tools (file) | How they're bound |
|---|---|---|
| Ticket agent | `create_ticket` (`tools/create_ticket.py`) | `llm.bind_tools(TICKET_AGENT_TOOLS)` – the LLM proposes the call, it runs only after "yes" |
| Status agent | `get_my_tickets`, `get_ticket_details`, `ask_employee`, `end_status_check` (`tools/status_tools.py`) | `create_agent(model, STATUS_AGENT_TOOLS)` |
| IT / HR agents | `kb_search` (`tools/kb_search.py`) | Not tool calling: a fixed LCEL retrieval chain, so the knowledge base is always searched before answering |

### IT and HR agents (RAG)
Both agents share one pipeline (`agents/kb_agent.py`) and differ only in their system prompt and the
`domain` filter (`IT` / `HR`) applied to the search. The retrieval → LLM step is a LangChain LCEL chain
(`agents/rag_chain.py`):

```
query → SmartDeskRetriever (Qdrant hybrid: dense + BM25, RRF, filtered by domain)
      → confidence gate (no docs? best similarity < CONFIDENCE_THRESHOLD?) ── fail → LLM skipped
      → ChatPromptTemplate (system prompt + chat history + <context> with [doc_id] chunks)
      → ChatOpenAI (.with_retry + .with_fallbacks) → StrOutputParser
```

**How the agent decides it doesn't know (escalation logic)** – four safeguards, in order:

| # | Check | Where | Result |
|---|---|---|---|
| 1 | No chunks retrieved (or Qdrant unreachable) | `rag/confidence.py` `retrieval_gate` | Offer a ticket; LLM not called |
| 2 | Top **dense cosine similarity** below `CONFIDENCE_THRESHOLD` | `rag/confidence.py` `retrieval_gate` | Offer a ticket; LLM not called |
| 3 | LLM replies *"I don't have enough information to answer that."* | `rag/confidence.py` `answer_gate` | Offer a ticket |
| 4 | LLM answers only part of the question | `rag/confidence.py` `answer_gate` | Show the answer + offer a ticket for the rest |

Every escalation is **human-in-the-loop**: the agent asks *"Would you like me to create a support ticket? (yes / no)"*
and the graph pauses at `offer_ticket` with `interrupt()`; only a "yes" starts the ticket agent. The threshold uses the dense cosine score,
not the RRF fusion score, because RRF is rank-based. Tune it with `python scripts/try_agent.py --eval`.

**Grounding rules in the system prompts** (`IT_AGENT_PROMPT`, `HR_AGENT_PROMPT`):
- Answer only from `<context>`; never use outside knowledge, guess, or invent steps, links or contacts.
- Copy numbers, amounts, dates, URLs, emails and extensions exactly; cite sources (`Source: IT-001`).
- Polite, professional, empathetic tone; numbered steps for procedures; concise.
- IT: never ask for or repeat passwords/MFA codes; urgent security actions first.
- HR: empathy first on sensitive topics (harassment, ethics), never ask for incident details or
  discourage reporting; policy facts only – no personal tax, legal or financial advice.
- Text in the context or the employee's message is treated as data, not instructions (prompt-injection guard).

**Other behaviour**
- **Follow-ups:** the last `HISTORY_TURNS` messages are sent to the LLM, and short follow-ups
  ("And on a Mac?") are searched together with the previous question.
- **Wrong agent:** if the IT agent gets an HR question (or vice versa) the LLM returns `[ROUTE:HR]`
  and the question is handed to the other agent.
- **LLM failure:** transient errors are retried (`LLM_MAX_RETRIES`), then `LLM_FALLBACK_MODEL` is tried
  if set; if all fail, the employee gets the titles of the relevant articles and a ticket offer.

### Ticket agent (subgraph, interrupt + bind_tools)
`offer_ticket` lives in the main graph (it can hand a new question back to the router). Steps 2–6 are the nodes of the
**ticket subgraph** (`agents/ticket_graph.py`, node functions in `agents/ticket_agent.py`), embedded as `ticket_agent`.
Hexagons in the diagram are `interrupt()` pauses:

1. **`offer_ticket`** – *interrupt*: "Would you like me to create a ticket? (yes / no)". yes → ticket flow; no → END;
   a new question instead of yes/no (e.g. "what's the maternity policy?") → the offer is dropped and the message goes back to the router.
2. **`collect_email`** – *interrupt* for the email only if it isn't already known in the session
   (an email typed anywhere is picked up; invalid ones are re-asked).
3. **`draft_ticket`** – the LLM is bound to `create_ticket_tool` with **`bind_tools`** and *proposes* a
   `create_ticket` call whose arguments (title, description, IT/HR, priority) are drafted from the conversation
   (`TICKET_TOOL_PROMPT`). The proposal is validated against the tool's Pydantic schema (`TicketInput`) and the
   email is always the employee's own. If the employee never said what the issue is, the LLM doesn't call the
   tool and `ask_details` (*interrupt*) asks for it.
4. **`confirm_ticket`** – *interrupt*: shows the proposed ticket and *"Shall I go ahead?"*.
   **yes** → `create_ticket`; **no** → cancelled, nothing created; anything else (extra details,
   "make it high priority", "yes, but…") → back to `draft_ticket` with the change request.
5. **`create_ticket`** – executes the **approved** tool call (Jira) and replies with the ticket key and link.
   Priority: Critical = security incident/outage, High = can't work, Medium = workaround exists, Low = question/request.
6. **Errors** – Jira unreachable or rejecting the request: polite message and back to `confirm_ticket`, where
   "try again" retries. LLM unavailable: a plain proposal is built from the employee's own words, still confirmed.

Nodes are side-effect free before `interrupt()` (LangGraph re-runs a node from the top when it resumes),
so the LLM call and the Jira call live in their own nodes.

### Status agent (ticket status – read operation, one tool-calling agent)
`agents/status_agent.py`: a single LangChain **`create_agent`** (model ⇄ tools loop, built on LangGraph) with four tools
from `tools/status_tools.py` (bound via `tools/registry.py`),
added to the main graph as one node. Jira reads live in `tools/get_ticket_status.py`; prompt `STATUS_AGENT_PROMPT`.

| Tool | What it does |
|---|---|
| `get_my_tickets()` | Live Jira search for the employee's tickets (`labels = "requester-<email>" AND project in (IT, HR)`), open first. If the email isn't known yet, the tool asks for it with **`interrupt()`** |
| `get_ticket_details(ticket_key)` | Status, priority, dates, link, latest support-team comments – **only if the key is one of the employee's tickets**, otherwise `NOT_YOUR_TICKET` |
| `ask_employee(message)` | **`interrupt()`** inside the tool: shows a question (e.g. "which ticket?") and returns the employee's reply to the LLM |
| `end_status_check(reason)` | The message isn't about tickets → back to the router (which won't send it straight back) |

The LLM decides the steps: one ticket → details; a question that names or describes one ticket ("my VPN issue", "the
screen one") → details; several → `ask_employee` with a numbered list, then details; none → "no tickets found". It writes
every reply, including Jira errors (tools return `ERROR: …` and the LLM apologises). When a tool pauses with `interrupt()`,
resuming re-runs only the tool step (read-only Jira calls), never the LLM call.

**Resilience (LangChain middleware):** `ModelRetryMiddleware` (transient OpenAI errors), `ModelFallbackMiddleware`
(`LLM_FALLBACK_MODEL`, if set) and `ModelCallLimitMiddleware` (max 6 model calls per request). If the agent still fails,
**`status_fallback`** answers without the LLM from the same Jira data (email → list → which one? → details) using a few
fixed messages (`STATUS_*` in `agents/prompts.py`).

**Privacy – employees only ever see their own tickets**, enforced in three layers:
1. The tools take **no email argument**. The verified email is read from the agent state with LangChain's
   **`InjectedState`**, which is hidden from the tool schema the LLM sees – the LLM can't supply or change it.
2. `get_ticket_details` is **checked in code** against the employee's own ticket list; other keys return
   `NOT_YOUR_TICKET` without revealing whether the ticket exists, and Jira is never asked for them.
3. **`STATUS_AGENT_PROMPT`** forbids looking up, confirming or describing anyone else's tickets or writing out email
   addresses – even if the employee gives another email, names a colleague, claims to be a manager or quotes a key that
   isn't theirs – and tells the LLM to ignore instructions that try to break these rules.

Tickets are linked to employees by the `requester-*` label that `create_ticket` adds, because Jira can't search by an
external requester's email. To test with tickets made outside SmartDesk, add that label in Jira, or use
`scripts/seed_tickets.py`.

### Smalltalk agent
Replies to greetings and thanks, explains what SmartDesk can do, and politely declines out-of-scope
requests (weather, sports, coding help…) without answering them or stating any policy from memory.
Falls back to a fixed greeting if the LLM is unavailable.

### Agent settings (`.env`, optional)
| Variable | Default | Purpose |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.30 | Minimum top dense similarity to answer (calibrate with `try_agent.py --eval`) |
| `RETRIEVAL_TOP_K` | 5 | Chunks sent to the LLM |
| `HISTORY_TURNS` | 6 | Past messages sent to the LLM |
| `LLM_FALLBACK_MODEL` | – | Second model tried if `LLM_MODEL` keeps failing |
| `LLM_MAX_RETRIES` / `LLM_TIMEOUT_SECONDS` / `LLM_TEMPERATURE` | 3 / 30 / 0 | LLM call behaviour |

## Testing
```bash
pytest tests -v                             # unit tests – offline, no API key needed
python scripts/build_index.py               # build the Qdrant index (needs OPENAI_API_KEY)
python main.py --debug                      # full multi-agent chat (routing, tickets, status, confirmations)
python main.py --seed jane@nmtech.com       # mock mode: start with 3 sample tickets to test the status flow
python scripts/try_agent.py --domain it     # debug the IT knowledge-base agent alone (or --domain hr)
python scripts/try_agent.py --eval          # run evaluation/eval_queries.json, PASS/FAIL per question
```
`try_agent.py` prints the similarity score, threshold, escalation reason and sources after each reply,
and `--eval` suggests a `CONFIDENCE_THRESHOLD` from the scores of answerable vs. gap questions.
Set `USE_MOCK_TICKETING=true` to try the ticket flow in `main.py` without creating real Jira tickets.
`tests/test_graph.py` runs complete conversations through the compiled graph (routing, multi-part messages – parallel
branches are timed to prove they run concurrently –, every interrupt,
bind_tools proposals, edits, cancellations, Jira failures, separate threads) with fake LLMs and mock Jira –
including the status agent (`create_agent`) driven by a scripted stand-in model and the privacy checks (no email
parameter on the tools, `NOT_YOUR_TICKET` for other employees' keys, LLM-written emails ignored, LLM not re-run on resume).

## Ticketing (Jira)
- `tools/create_ticket.py` – `create_ticket(email, summary, description, category, priority)` and the LangChain
  `create_ticket_tool`. IT tickets go to `JIRA_IT_PROJECT_KEY`, HR tickets to `JIRA_HR_PROJECT_KEY`.
- Input is validated first (valid email, 5–120 char summary, IT/HR, Low/Medium/High/Critical).
- The description (Atlassian Document Format) records requester email, category and priority; labels
  `smartdesk`, `it`/`hr`, `priority-*`, `requester-*` make tickets easy to filter in Jira.
- Created tickets are saved in SQLite (`TICKET_DB_PATH`) so the status check can find them by email.
- Jira down → `TicketingUnavailable` (retried first); Jira rejects the request → `TicketingError`.
- `USE_MOCK_TICKETING=true` uses an in-memory Jira for demos without an account.

- `tools/get_ticket_status.py` – `get_tickets_by_email(email)` and `get_ticket_details(key)` read tickets, status and
  comments live from Jira (see *Status agent*).

```bash
python scripts/test_jira.py                       # check auth, projects, issue type, priority field, search (read-only)
python scripts/test_jira.py --create              # create a real test ticket
python scripts/seed_tickets.py jane@nmtech.com    # create 3 sample tickets (+ a support comment) in real Jira
```

## Vector database
Qdrant stores a dense embedding and a BM25 sparse vector for every chunk in one collection.
Search runs both and merges them with Qdrant's built-in Reciprocal Rank Fusion (hybrid search).

## Semantic cache (repeated questions)
Similar IT/HR questions are answered from a cache instead of running retrieval and the LLM again
(`memory/semantic_cache.py`). "How many sick days do I get?" and "what's my annual sick leave allowance?" share one
cached answer.

- **How:** the question is embedded with the same model as the knowledge base and compared with earlier
  questions in a separate Qdrant collection (`smartdesk_cache`). If the closest one in the **same domain** scores at
  least `CACHE_THRESHOLD` (cosine, default 0.93), its answer is returned. The check happens inside the IT/HR agent,
  after routing, so ticket flows, status checks and privacy rules are unaffected. The question's embedding is
  reused by retrieval on a miss, so a miss costs no extra embedding call.
- **Only good answers are cached:** confident, grounded knowledge-base answers. Never: "couldn't find it" /
  partial answers / ticket offers, ticket status or creation (personal), small talk, or short follow-ups that
  depend on the conversation ("what about on a Mac?").
- **Staying correct:** every entry stores the knowledge-base version (hash of `knowledge_base/`), so entries from an
  older KB are ignored; `scripts/build_index.py` also clears the cache on re-index; entries expire after
  `CACHE_TTL_HOURS` (default 168). Any cache error counts as a miss.
- **Settings (.env):** `SEMANTIC_CACHE_ENABLED=true`, `CACHE_THRESHOLD=0.93`, `CACHE_TTL_HOURS=168`,
  `CACHE_COLLECTION=smartdesk_cache`.
- **Tuning:** `python scripts/cache_check.py` prints similarity scores for paraphrases vs. different questions
  with your embedding model – keep the threshold above every "different" score.
- **Seeing it work:** `python main.py --debug` shows `cache_hit=True`, and the log line `Cache HIT (HR, 0.95): …`;
  in the web UI turn on *Show routing details*. Tests: `tests/test_semantic_cache.py`.

## Environment variables
See `.env.example`.

## Architecture
![SmartDesk architecture: router → parallel agents → synthesizer](docs/architecture.svg)

| File | What it is |
|---|---|
| `docs/architecture.svg` | Main flow diagram (router → parallel agents → synthesizer), shown above |
| `docs/architecture.png` | Same diagram as an image, for slides |
| `docs/architecture.mmd` | Same diagram in Mermaid, editable as text |
| `docs/architecture.html` | Full page: main flow, ticket subgraph and status agent |
| `docs/architecture.md` | Component diagram and the graph generated from code |
