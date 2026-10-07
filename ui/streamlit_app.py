"""SmartDesk AI – Streamlit chat UI.

    streamlit run ui/streamlit_app.py

Everything goes through the same LangGraph app as the CLI (SmartDeskChat): the quick-action buttons and the FAQ
buttons just send a message for the employee, so routing, the knowledge base, ticket confirmation (interrupt) and
the privacy rules all work exactly as in chat.

  Sidebar        work email · new conversation · (mock mode) sample tickets · debug details
  Quick actions  Create ticket (form) · Check ticket status
  Popular FAQs   IT and HR tabs – one click asks the question
  Chat window    history + chat input; Yes / No buttons when SmartDesk asks for a confirmation
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                       # `streamlit run ui/streamlit_app.py` from the project root
    sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402  (loads .env)
from agents.graph import SmartDeskChat, build_graph  # noqa: E402

EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")

# Popular questions (all answered by the knowledge base: knowledge_base/qa_pairs)
IT_FAQS = [
    "How do I reset my password?",
    "My account is locked. What should I do?",
    "How do I connect to the VPN?",
    "How do I set up MFA?",
    "How do I request a software license?",
    "How do I set up work email on my phone?",
    "How do I get guest Wi-Fi for a visitor?",
    "What are the IT help desk hours?",
]
HR_FAQS = [
    "How much PTO do I get?",
    "How many sick days do I get per year?",
    "How much parental (maternity or paternity) leave does NMTech offer?",
    "What is the work-from-home policy?",
    "How do I submit an expense reimbursement?",
    "When is payday?",
    "Does NMTech match 401(k) contributions?",
    "How do I enroll in health insurance?",
]

# Interrupt types (agents/ticket_agent.py) that are yes / no questions
YES_NO = {"ticket_offer", "ticket_confirmation"}
WAITING_HINT = {
    "ticket_offer": "SmartDesk is asking whether to raise a ticket.",
    "ticket_confirmation": "SmartDesk is waiting for you to confirm the ticket.",
    "ticket_details": "SmartDesk needs a few more details about the issue.",
    "email": "SmartDesk needs your work email to raise the ticket.",
    "status_email": "SmartDesk needs your work email to look up your tickets.",
    "choose_ticket": "Reply with the ticket key or its number in the list.",
}

WELCOME = ("Hi! I'm **SmartDesk**, NMTech's IT & HR help desk assistant. Ask me anything about IT or HR "
           "policies, raise a support ticket, or check on a ticket you already have.")

st.set_page_config(page_title="SmartDesk AI", page_icon=":material/support_agent:", layout="wide")
st.markdown("""
<style>
  /* Pastel green buttons, compact size */
  .stButton > button, div[data-testid="stDialog"] .stButton > button {
    background: #dff3e7; color: #1f4d36; border: 1px solid #a9dbbd;
    min-height: 2rem; padding: 0.2rem 0.8rem; border-radius: 8px;
  }
  .stButton > button p { font-size: 0.85rem; }
  .stButton > button:hover { background: #c9ebd7; border-color: #7cc79c; color: #163b29; }
  .stButton > button:focus:not(:active) { border-color: #5fb784; color: #163b29; }
  .stButton > button[kind="primary"] { background: #bfe6cf; border-color: #7cc79c; font-weight: 600; }
  .stButton > button[kind="primary"]:hover { background: #a9dcbe; }
  /* FAQ buttons: left-aligned text */
  div[data-testid="stTabs"] .stButton > button { justify-content: flex-start; text-align: left; }
  div[data-testid="stTabs"] .stButton > button p { text-align: left; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- session
@st.cache_resource(show_spinner="Starting SmartDesk…")
def get_graph():
    """One compiled graph (with its MemorySaver) for the whole app; each browser session has its own thread_id."""
    return build_graph()


def new_conversation():
    st.session_state.chat = SmartDeskChat(get_graph())
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
    st.session_state.pending = None
    st.session_state.synced_email = None


if "chat" not in st.session_state:
    new_conversation()


def ask(text: str):
    """Queue a message (from a button or the chat input); it's sent on this run, below the history."""
    st.session_state.pending = text


def sync_email():
    """Give the graph the employee's email from the sidebar (only between turns, never mid-question)."""
    chat, email = st.session_state.chat, st.session_state.get("email", "").strip().lower()
    if not email or email == st.session_state.synced_email or not EMAIL_RE.match(email):
        return
    if chat.waiting_for() is None:
        chat.graph.update_state(chat.config, {"employee_email": email})
        st.session_state.synced_email = email


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title(":material/support_agent: SmartDesk AI")
    st.caption("NMTech · IT & HR help desk")

    email = st.text_input("Your work email", key="email", placeholder="name@nmtech.com",
                          help="Optional. Used to raise tickets and to show **your own** tickets only.")
    if email and not EMAIL_RE.match(email.strip()):
        st.warning("That doesn't look like a valid email address.")
    sync_email()

    st.button("New conversation", icon=":material/add_comment:", on_click=new_conversation,
              use_container_width=True)

    if settings.USE_MOCK_TICKETING:
        st.divider()
        st.caption("Mock ticketing mode (no real Jira).")
        if st.button("Add sample tickets for my email", icon=":material/dataset:", use_container_width=True,
                     disabled=not (email and EMAIL_RE.match(email.strip()))):
            from scripts.seed_tickets import seed
            keys = [t["key"] for t in seed(email.strip().lower())]
            st.success(f"Added {', '.join(keys)}")

    st.divider()
    debug = st.toggle("Show routing details", help="Route, retrieval score and what SmartDesk is waiting for.")


# ----------------------------------------------------------------------------- header + quick actions
st.header("How can I help you today?")

@st.dialog("Create a support ticket")
def ticket_form():
    st.caption("SmartDesk drafts the ticket and shows it to you before anything is created.")
    team = st.radio("Team", ["IT", "HR"], horizontal=True)
    issue = st.text_area("What's the problem?", placeholder="e.g. My laptop won't connect to the office Wi-Fi "
                                                            "since this morning.")
    if st.button("Draft ticket", type="primary"):
        if not issue.strip():
            st.warning("Please describe the problem first.")
        else:
            ask(f"Please raise an {team} ticket: {issue.strip()}")
            st.rerun()


c1, c2, _ = st.columns([1, 1.3, 3.5])
with c1:
    if st.button("Create ticket", icon=":material/confirmation_number:", type="primary", use_container_width=True):
        ticket_form()
with c2:
    st.button("Check ticket status", icon=":material/manage_search:", use_container_width=True,
              on_click=ask, args=("What's the status of my tickets?",))

with st.expander("Popular questions", icon=":material/help:", expanded=False):
    it_tab, hr_tab = st.tabs(["IT", "HR"])
    for tab, faqs, prefix in ((it_tab, IT_FAQS, "it"), (hr_tab, HR_FAQS, "hr")):
        with tab:
            cols = st.columns(2)
            for i, q in enumerate(faqs):
                cols[i % 2].button(q, key=f"faq_{prefix}_{i}", on_click=ask, args=(q,), use_container_width=True)


# ----------------------------------------------------------------------------- chat window
chat = st.session_state.chat
window = st.container(height=480, border=True)
with window:
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user"
                             else ":material/support_agent:"):
            st.markdown(m["content"])

    text = st.session_state.pending
    if text:
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": text})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(text)
        with st.chat_message("assistant", avatar=":material/support_agent:"):
            with st.spinner("Thinking…"):
                try:
                    reply = chat.send(text)
                except Exception:  # noqa: BLE001 – never show a stack trace to the employee
                    reply = ("Sorry, something went wrong on my side. Please try again in a moment, or contact "
                             "the IT Service Desk if it keeps happening.")
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        sync_email()
        st.rerun()

waiting = chat.waiting_for()
if waiting:
    hint = WAITING_HINT.get(waiting, "SmartDesk is waiting for your reply.")
    if waiting in YES_NO:
        h, y, n = st.columns([4, 1, 1])
        h.info(hint, icon=":material/pending:")
        y.button("Yes", icon=":material/check:", type="primary", use_container_width=True, on_click=ask, args=("yes",))
        n.button("No", icon=":material/close:", use_container_width=True, on_click=ask, args=("no",))
    else:
        st.info(hint, icon=":material/pending:")

if debug:
    s = chat.state
    score = s.get("confidence")
    st.caption(f"route: `{s.get('route')}` ({s.get('route_reason') or '-'}) · domain: `{s.get('domain')}` · "
               f"score: `{round(score, 3) if isinstance(score, float) else '-'}` · "
               f"escalation: `{s.get('escalation_reason') or 'none'}` · waiting for: `{waiting}` · "
               f"email: `{s.get('employee_email')}`")

if prompt := st.chat_input("Ask an IT or HR question, or describe a problem…"):
    ask(prompt)
    st.rerun()
