# SmartDesk AI – Architecture

## Components and data flow

```mermaid
flowchart TD
    U[Employee] -->|message| UI[CLI main.py / Streamlit]
    UI -->|send / resume| G[LangGraph app<br/>agents/graph.py]
    G <-->|state per thread_id| CP[(MemorySaver<br/>in-memory checkpointer)]

    subgraph Graph
      R[router<br/>Pydantic RouteDecision] -->|IT question| IT[it_agent]
      R -->|HR question| HR[hr_agent]
      R -->|greeting / out of scope| SM[smalltalk_agent]
      R -->|existing ticket| ST
      subgraph ST [status_agent – LangChain create_agent]
        MD[model<br/>ChatOpenAI + retry / fallback / call-limit middleware] <-->|tool calls / results| ST2[tools<br/>get_my_tickets · get_ticket_details<br/>ask_employee · end_status_check]
        ST2 -.->|interrupt: email / which ticket?| MD
      end
      ST -->|end_status_check| R
      ST -.->|LLM down| SF{{status_fallback<br/>no-LLM: email → list → details}}
      R -->|explicit ticket request| TK
      R ==>|"multi-part: Send() × N"| IT & HR & SM & ST
      IT & HR & SM & ST ==>|branch output| SY[synthesizer<br/>agents/synthesizer.py<br/>LLM merges labelled outputs]
      SY -.->|one follow-up| TK
      SY -.->|one follow-up| OF
      SY -.->|one follow-up| ST
      IT <-->|wrong domain| HR
      IT & HR -->|can't answer| OF{{offer_ticket<br/>interrupt: yes/no}}
      OF -->|yes| TK
      subgraph TK [ticket_agent – subgraph]
        CE{{collect_email<br/>interrupt: email}} --> DR[draft_ticket<br/>LLM.bind_tools create_ticket]
        DR -->|issue unclear| AD{{ask_details<br/>interrupt}}
        AD --> DR
        DR --> CF{{confirm_ticket<br/>interrupt: Shall I go ahead?}}
        CF -->|changes| DR
        CF -->|yes| CT[create_ticket<br/>executes approved tool call]
        CT -->|Jira down| CF
      end
    end

    IT & HR --> RAG[LCEL chain: SmartDeskRetriever → gate → prompt → ChatOpenAI]
    RAG --> Q[(Qdrant<br/>dense + BM25 hybrid)]
    CT --> J[(Jira Cloud REST API)]
    ST2 & SF --> J
    CT --> DB[(SQLite<br/>email → ticket keys)]
    R & RAG & DR & MD & SY --> LLM[[OpenAI LLM<br/>retry + fallback model]]
```
Hexagons and dotted "interrupt" arrows are human-in-the-loop points: the graph pauses with `interrupt()` and resumes
with `Command(resume=<employee reply>)`. In the status agent the pauses happen inside the tools. Privacy: the tools
read the verified email via `InjectedState` (hidden from the LLM) and check ticket ownership in code.

## LangGraph graph (generated)
Generated from the compiled graph with the ticket subgraph expanded (`xray=True`):
`python -c "from agents.graph import build_graph; print(build_graph().get_graph(xray=True).draw_mermaid())"`.
Dotted edges are `Command(goto=...)` transitions chosen inside the node.

The status agent is invoked inside the `status_agent` node, so its internal model ⇄ tools loop isn't expanded here.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	router(router)
	it_agent(it_agent)
	hr_agent(hr_agent)
	smalltalk_agent(smalltalk_agent)
	status_agent(status_agent)
	status_fallback(status_fallback)
	offer_ticket(offer_ticket)
	synthesizer(synthesizer)
	__end__([<p>__end__</p>]):::last
	__start__ --> router;
	hr_agent -.-> __end__;
	hr_agent -.-> it_agent;
	hr_agent -.-> offer_ticket;
	hr_agent -.-> synthesizer;
	it_agent -.-> __end__;
	it_agent -.-> hr_agent;
	it_agent -.-> offer_ticket;
	it_agent -.-> synthesizer;
	offer_ticket -.-> __end__;
	offer_ticket -.-> router;
	offer_ticket -.-> ticket_agent\3a__start__;
	router -.-> hr_agent;
	router -.-> it_agent;
	router -.-> smalltalk_agent;
	router -.-> status_agent;
	router -.-> synthesizer;
	router -.-> ticket_agent\3a__start__;
	smalltalk_agent -.-> __end__;
	smalltalk_agent -.-> synthesizer;
	status_agent -.-> __end__;
	status_agent -.-> router;
	status_agent -.-> status_fallback;
	status_agent -.-> synthesizer;
	status_fallback -.-> __end__;
	status_fallback -.-> router;
	synthesizer -.-> __end__;
	synthesizer -.-> offer_ticket;
	synthesizer -.-> status_agent;
	synthesizer -.-> ticket_agent\3a__start__;
	ticket_agent\3a__end__ --> __end__;
	offer_ticket -.-> offer_ticket;
	subgraph ticket_agent
	ticket_agent\3a__start__(<p>__start__</p>)
	ticket_agent\3acollect_email(collect_email)
	ticket_agent\3adraft_ticket(draft_ticket)
	ticket_agent\3aask_details(ask_details)
	ticket_agent\3aconfirm_ticket(confirm_ticket)
	ticket_agent\3acreate_ticket(create_ticket)
	ticket_agent\3a__end__(<p>__end__</p>)
	ticket_agent\3a__start__ --> ticket_agent\3acollect_email;
	ticket_agent\3aask_details -.-> ticket_agent\3a__end__;
	ticket_agent\3aask_details -.-> ticket_agent\3adraft_ticket;
	ticket_agent\3acollect_email -.-> ticket_agent\3a__end__;
	ticket_agent\3acollect_email -.-> ticket_agent\3adraft_ticket;
	ticket_agent\3aconfirm_ticket -.-> ticket_agent\3a__end__;
	ticket_agent\3aconfirm_ticket -.-> ticket_agent\3acreate_ticket;
	ticket_agent\3aconfirm_ticket -.-> ticket_agent\3adraft_ticket;
	ticket_agent\3acreate_ticket -.-> ticket_agent\3a__end__;
	ticket_agent\3acreate_ticket -.-> ticket_agent\3aconfirm_ticket;
	ticket_agent\3adraft_ticket -.-> ticket_agent\3aask_details;
	ticket_agent\3adraft_ticket -.-> ticket_agent\3aconfirm_ticket;
	ticket_agent\3acollect_email -.-> ticket_agent\3acollect_email;
	end
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
