""" 
    Bank support assistant.
    A chat interface over a LangGraph workflow. Every assistant reply carries the
    route, evidence, and trace that produced it, because in a bank the answer is
    only half of what matters.
"""


import uuid

import streamlit as st

import src.config as config
from src.config import (
    BANK_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_APPROVAL_LIMIT,
    MAX_REVISION_ATTEMPTS,
)
from src.graph import pending_question, resume_turn, run_turn, snapshot
from src.retriever import collection_size
from src.scenarios import SCENARIOS
from src.schemas import Outcome
from src.tools import record_approval_decision
from ui.components import (
    approval_card,
    decision_panel,
    escalation_card,
    outcome_banner,
    specialist_queue_panel,
)

st.set_page_config(
    page_title="Bank Assistant Agent",
    page_icon="🏦",
    layout="wide",
)



# Session

def new_conversation() -> None:
    """Start a fresh thread.

    The thread id is what isolates one conversation's memory from another's, so
    a new conversation must never reuse one.
    """

    st.session_state.thread_id = f"THREAD-{uuid.uuid4().hex[:8].upper()}"
    st.session_state.turns = []
    st.session_state.awaiting_answer = False
    st.session_state.queued_message = None


for key, default in (
    ("thread_id", None),
    ("turns", []),
    ("awaiting_answer", False),
    ("queued_message", None),
):
    st.session_state.setdefault(key, default)

if st.session_state.thread_id is None:
    new_conversation()


def submit(message: str) -> None:
    """Queue a message so a button click and a chat entry take the same path."""

    st.session_state.queued_message = message


def last_reply(state: dict) -> str:
    replies = [m for m in state.get("messages", []) if m.type == "ai"]
    return str(replies[-1].content) if replies else ""


def handle_decision(pending, decision: str) -> None:
    record_approval_decision.invoke(
        {
            "approval_id": pending.approval_id,
            "decision": decision,
            "approver": "demo reviewer",
            "note": f"{decision} from the support console",
        }
    )
    pending.status = decision
    st.session_state.decisions = st.session_state.get("decisions", {})
    st.session_state.decisions[pending.approval_id] = decision
    st.rerun()



# Sidebar

with st.sidebar:
    st.markdown(f"### {BANK_NAME} Assistant")
    st.caption("Multi-agent support workflow")

    if GEMINI_API_KEY:
        st.success(f"Gemini: `{GEMINI_MODEL}`")
    else:
        st.warning("No GEMINI_API_KEY — deterministic fallbacks active")

    try:
        st.caption(f"Vector store: {collection_size()} chunks indexed")
    except Exception:  # noqa: BLE001 - the interface must load without an index
        st.error("No vector index. Run `python -m scripts.build_vector_db`.")

    st.caption(
        f"Automated refund limit ${MAX_APPROVAL_LIMIT:,.0f} · "
        f"max {MAX_REVISION_ATTEMPTS} revisions"
    )

    st.divider()
    st.markdown("#### Try a scenario")
    for scenario in SCENARIOS:
        if st.button(scenario.label, key=f"scn-{scenario.key}", width="stretch"):
            new_conversation()
            config.FORCE_BAD_DRAFTS = 0
            config.FORCE_TOOL_FAILURE = ""
            config.FORCE_LLM_FAILURE = False
            for field, value in scenario.injections.items():
                if field == "force_bad_drafts":
                    config.FORCE_BAD_DRAFTS = int(value)
                elif field == "force_tool_failure":
                    config.FORCE_TOOL_FAILURE = str(value)
                elif field == "force_llm_failure":
                    config.FORCE_LLM_FAILURE = bool(value)
            st.session_state.active_scenario = scenario.key
            submit(scenario.message)
            st.rerun()

    active = st.session_state.get("active_scenario")
    if active:
        scenario = next(s for s in SCENARIOS if s.key == active)
        with st.expander("What this scenario shows", expanded=True):
            st.markdown(f"**{scenario.kind}**")
            st.markdown(scenario.notes)
            st.caption("Expected route: " + " → ".join(scenario.expected_route))
            if scenario.follow_up:
                st.caption(f"Follow-up to send: {scenario.follow_up}")

    st.divider()
    if st.button("New conversation", width="stretch"):
        st.session_state.pop("active_scenario", None)
        new_conversation()
        st.rerun()

    st.caption(f"Thread `{st.session_state.thread_id}`")



# Header

st.title(f"{BANK_NAME} Assistant Agent")
st.caption(
    "Triage classifies and routes · a domain specialist retrieves policy and drafts · "
    "a critic validates · consequential actions stop for a person. "
)



message = st.session_state.queued_message
if message:
    st.session_state.queued_message = None
    resuming = st.session_state.awaiting_answer

    with st.spinner("Working through the case..."):
        try:
            if resuming:
                result = resume_turn(message, thread_id=st.session_state.thread_id)
            else:
                result = run_turn(message, thread_id=st.session_state.thread_id)
        except Exception as exc:  # noqa: BLE001 - surfaced rather than crashing the app
            st.error(f"The workflow could not complete: {exc}")
            result = None

    if result is not None:
        question = pending_question(result)
        state = snapshot(st.session_state.thread_id)

        st.session_state.awaiting_answer = bool(question)
        st.session_state.turns.append(
            {"customer": message, "question": question, "state": state}
        )
    st.rerun()



# Conversation



for index, turn in enumerate(st.session_state.turns):
    with st.chat_message("user"):
        st.markdown(turn["customer"])

    state = turn["state"]
    with st.chat_message("assistant"):
        if turn["question"]:
            st.markdown(turn["question"])
            st.caption("The workflow is paused here. Your reply goes back into triage.")
        else:
            st.markdown(last_reply(state) or "_No reply produced._")

        outcome_banner(state)

        if state.get("outcome") is Outcome.PENDING_APPROVAL:
            decided = st.session_state.get("decisions", {})
            pending = state.get("pending_approval")
            if pending and pending.approval_id in decided:
                verdict = decided[pending.approval_id]
                if verdict == "approved":
                    st.success(
                        f"Approved by a human. `{pending.action_type}` of "
                        f"${pending.amount:,.2f} would now be processed. Simulated."
                        if pending.amount is not None
                        else f"Approved by a human. `{pending.action_type}` would now "
                             "be processed. Simulated."
                    )
                else:
                    st.error(
                        f"Rejected by a human. `{pending.action_type}` will not proceed."
                    )
            elif index == len(st.session_state.turns) - 1:
                approval_card(state, handle_decision)

        if state.get("outcome") is Outcome.ESCALATED:
            escalation_card(state)

        with st.expander("How this was decided", expanded=index == len(st.session_state.turns) - 1):
            decision_panel(state)



# Input

placeholder = (
    "Answer the question above..."
    if st.session_state.awaiting_answer
    else "Describe your problem..."
)

if typed := st.chat_input(placeholder):
    submit(typed)
    st.rerun()


with st.expander("Specialist queue"):
    specialist_queue_panel()
