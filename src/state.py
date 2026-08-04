"""LangGraph state.
Roles never call each other. Everything moves through this object, so the trace
in the interface is a complete account of how a turn reached its outcome.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.schemas import (
    Critique,
    Draft,
    Outcome,
    PendingApproval,
    Source,
    ToolCall,
    TraceEvent,
    TriageResult,
)


class ChatState(TypedDict, total=False):
    """Shared state for one conversation thread."""

    # --- Conversation ---
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    customer_message: str

    # --- Triage ---
    triage: TriageResult | None
    domain: str
    clarifications: int

    # --- Evidence ---
    customer_record: dict | None
    account_records: list[dict]
    transactions: list[dict]
    cards: list[dict]
    loans: list[dict]
    credit_profile: dict | None
    fee_history: dict | None
    loan_assessment: dict | None
    fraud_risk: dict | None
    sources: list[Source]

    # --- Specialist agent ---
    draft: Draft | None
    retry_count: int
    critic_feedback: list[str]

    # --- Critic ---
    critique: Critique | None
    # Every verdict this turn, not just the last one. Without these two the
    # shared-state log shows only the draft that was finally accepted, so a
    # rejection and the reason for it disappear the moment a retry succeeds.
    critique_history: list[Critique]
    draft_history: list[Draft]

    # --- Outcome ---
    outcome: Outcome | None
    outcome_summary: str
    pending_approval: PendingApproval | None
    ticket_id: str | None
    escalation_id: str | None
    questions: list[str]

    # --- Observability ---
    trace: Annotated[list[TraceEvent], operator.add]
    tool_calls: Annotated[list[ToolCall], operator.add]
    errors: Annotated[list[str], operator.add]
    degraded: bool


def new_turn(session_id: str, customer_message: str) -> dict:
    """Input for one turn.

    Per-turn fields are reset explicitly so a follow-up never inherits the
    previous turn's draft or verdict. `messages` is intentionally absent from
    the reset list: its reducer appends, which is what carries the
    conversation forward.
    """

    return {
        "session_id": session_id,
        "customer_message": customer_message,
        "triage": None,
        "domain": "",
        "customer_record": None,
        "account_records": [],
        "transactions": [],
        "cards": [],
        "loans": [],
        "credit_profile": None,
        "fee_history": None,
        "loan_assessment": None,
        "fraud_risk": None,
        "sources": [],
        "draft": None,
        "retry_count": 0,
        "critic_feedback": [],
        "critique": None,
        "critique_history": [],
        "draft_history": [],
        "outcome": None,
        "outcome_summary": "",
        "pending_approval": None,
        "ticket_id": None,
        "escalation_id": None,
        "questions": [],
        "degraded": False,
    }
