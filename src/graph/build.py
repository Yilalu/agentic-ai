"""Graph assembly and the entry points the interface uses."""

import functools
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from .nodes import (
    account_agent_node,
    ask_user_node,
    card_agent_node,
    critic_node,
    escalated_node,
    fraud_agent_node,
    human_approval_node,
    intake_node,
    loan_agent_node,
    out_of_scope_node,
    resolved_node,
    triage_node,
)
from src.graph.routes import (
    route_after_critic,
    route_after_out_of_scope,
    route_after_triage,
)
from src.schemas import (
    Critique,
    Domain,
    Draft,
    Outcome,
    PendingApproval,
    ProposedAction,
    Source,
    ToolCall,
    TraceEvent,
    TriageResult,
    Verdict,
)
from src.state import ChatState, new_turn

SPECIALIST_NODES = ("card_agent", "loan_agent", "account_agent", "fraud_agent")

CHECKPOINTED_TYPES = [
    Critique,
    Domain,
    Draft,
    Outcome,
    PendingApproval,
    ProposedAction,
    Source,
    ToolCall,
    TraceEvent,
    TriageResult,
    Verdict,
]


def build_graph(checkpointer=None):
    """Wire the workflow.

    START -> intake -> triage
      triage -> ask_user            (missing info; capped, then escalated)
      triage -> card_agent | loan_agent | account_agent   -> critic
      triage -> fraud_agent                               -> escalated
      triage -> out_of_scope        (no agent and no policy covers it)
      out_of_scope -> escalated     (bank business, so a person can help)
      out_of_scope -> END           (nothing to do with the bank)
      ask_user -> triage            (after the customer replies)
      critic -> resolved            (approve, refund within the limit)
      critic -> human_approval      (approve, over the limit or any loan)
      critic -> <originating agent> (revise, within the retry budget)
      critic -> escalated           (escalate, or the budget is spent)
      resolved | human_approval | escalated -> END
    """

    graph = StateGraph(ChatState)

    graph.add_node("intake", intake_node)
    graph.add_node("triage", triage_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("card_agent", card_agent_node)
    graph.add_node("loan_agent", loan_agent_node)
    graph.add_node("account_agent", account_agent_node)
    graph.add_node("fraud_agent", fraud_agent_node)
    graph.add_node("critic", critic_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("resolved", resolved_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("escalated", escalated_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "triage")

    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "ask_user": "ask_user",
            "card_agent": "card_agent",
            "loan_agent": "loan_agent",
            "account_agent": "account_agent",
            "fraud_agent": "fraud_agent",
            "out_of_scope": "out_of_scope",
            "escalated": "escalated",
        },
    )

    # Declining is not always the end of it. A banking question outside these
    # four domains still goes to somebody who can answer it.
    graph.add_conditional_edges(
        "out_of_scope",
        route_after_out_of_scope,
        {"escalated": "escalated", "end": END},
    )

    # The clarification loop: the customer's answer re-enters classification.
    graph.add_edge("ask_user", "triage")

    # The three drafting specialists always hand to the critic. None of them
    # judges its own output.
    for node in ("card_agent", "loan_agent", "account_agent"):
        graph.add_edge(node, "critic")

    # Fraud bypasses the critic entirely: there is no automated resolution to
    # validate, only a case to hand to an investigator.
    graph.add_edge("fraud_agent", "escalated")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "card_agent": "card_agent",
            "loan_agent": "loan_agent",
            "account_agent": "account_agent",
            "resolved": "resolved",
            "human_approval": "human_approval",
            "escalated": "escalated",
        },
    )

    for terminal in ("resolved", "human_approval", "escalated"):
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=checkpointer or memory_checkpointer())


def memory_checkpointer() -> MemorySaver:
    """In-memory checkpointer that knows how to restore this project's schemas.

    State carries Pydantic models, and LangGraph will refuse to deserialize
    unregistered types in a future release. Declaring them makes that
    restriction a non-event rather than a breaking upgrade.
    """

    return MemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINTED_TYPES)
    )


@functools.lru_cache(maxsize=1)
def get_graph():
    """Shared compiled graph.

    The checkpointer is what gives each `thread_id` its own conversation, and
    what lets an interrupted turn resume where it paused.
    """

    return build_graph()


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}


def run_turn(message: str, thread_id: str | None = None, session_id: str | None = None) -> dict:
    """Run one turn to completion or to a pause.

    Returns the state snapshot. If the graph paused to ask a question, the
    snapshot carries `__interrupt__`; call `resume_turn` with the answer.
    """

    thread = thread_id or f"THREAD-{uuid.uuid4().hex[:8].upper()}"
    graph = get_graph()
    return graph.invoke(
        new_turn(session_id or thread, message), config=thread_config(thread)
    )


def resume_turn(answer: str, thread_id: str) -> dict:
    """Resume a paused turn with the customer's answer."""

    from langgraph.types import Command

    return get_graph().invoke(Command(resume=answer), config=thread_config(thread_id))


def snapshot(thread_id: str) -> dict:
    """Full accumulated state for a thread."""

    state = get_graph().get_state(thread_config(thread_id))
    return dict(state.values) if state and state.values else {}


def pending_question(result: dict) -> str | None:
    """The question text when a turn paused, otherwise None."""

    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None

    payload = interrupts[0].value
    if isinstance(payload, dict):
        return payload.get("prompt")
    return str(payload)
