"""Routing functions.

Routing lives here rather than in the nodes or the prompts. Each router is a
pure function of state, so any route the graph took can be replayed and
explained from the state alone.
"""

from .config import MAX_CLARIFFICATIONS, MAX_TOOL_ATTEMPST, MAX_REVISION_ATTEMPTS, MAX_APPROVAL_LIMIT
from .schemas import Domain, Verdict
from .state import ChatState


def route_after_triage(state: ChatState) -> str:
    """One LLM call, then this function picks exactly one destination."""

    triage = state.get("triage")
    if triage is None:
        return "escalated"

    if triage.domain is Domain.OUT_OF_SCOPE:
        # One edge for everything nothing here can answer, whether or not it is
        # bank business. Whether a person is needed is decided at out_of_scope,
        # so triage has a single destination for "no agent and no policy covers
        # this" rather than two that mean nearly the same thing.
        return "out_of_scope"

    if triage.missing_info:
        # Cap the clarification loop. A customer who cannot supply what is
        # needed after two tries gets a person, not a third question.
        if state.get("clarifications", 0) >= MAX_CLARIFFICATIONS:
            return "escalated"
        return "ask_user"

    return f"{triage.domain.value}_agent"


def route_after_out_of_scope(state: ChatState) -> str:
    """Is there anybody to hand this to?

    Both kinds of unanswerable request arrive here. A banking matter outside the
    four domains goes to a person who can help. A question with nothing to do
    with the bank has nobody to go to, so raising a ticket for it would be
    theatre; the decline is the whole outcome.
    """

    triage = state.get("triage")
    return "escalated" if triage and triage.bank_related else "end"


def needs_human_approval(state: ChatState) -> bool:
    """Application-owned threshold check, independent of the critic's verdict.

    An approved draft still cannot move money on its own. These are the rules
    from POL-AUTH-008 encoded as code rather than as prompt text.
    """

    if state.get("domain") == Domain.LOAN.value:
        return True

    draft = state.get("draft")
    if draft is None or draft.action is None:
        return False

    action = draft.action
    if action.action_type == "refund":
        return (action.amount or 0.0) > MAX_APPROVAL_LIMIT

    # Anything that is not a refund and not purely informational gets a person.
    return action.action_type not in {"information_only", "reset_access", "open_dispute"}


def route_after_critic(state: ChatState) -> str:
    """Verdict plus application-owned thresholds. The model never decides alone."""

    critique = state.get("critique")
    if critique is None:
        return "escalated"

    if critique.verdict is Verdict.ESCALATE:
        return "escalated"

    if critique.verdict is Verdict.REVISE:
        # The controlled loop. `retry_count` was incremented by the critic node,
        # so passing the budget terminates in escalation instead of cycling.
        if state.get("retry_count", 0) > MAX_REVISION_ATTEMPTS:
            return "escalated"
        # Back to whichever specialist wrote it, with the critic's fixes.
        return f"{state.get('domain', 'account')}_agent"

    return "human_approval" if needs_human_approval(state) else "resolved"
