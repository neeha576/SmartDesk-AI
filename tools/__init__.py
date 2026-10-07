"""All tools used by SmartDesk agents.

LangChain tools (bound to agents in tools/registry.py):
    create_ticket.py   create_ticket_tool                      -> ticket agent (bind_tools)
    status_tools.py    get_my_tickets, get_ticket_details,
                       ask_employee, end_status_check          -> status agent (create_agent)
Plain helpers used by those tools / the RAG chain:
    kb_search.py         hybrid knowledge-base search (IT/HR agents' retriever)
    get_ticket_status.py Jira reads: tickets by requester email, ticket details + comments
    ticketing_client.py  Jira REST client (retries, errors);  mock_ticketing.py  in-memory Jira
    ticket_store.py      local SQLite record of created tickets
"""
