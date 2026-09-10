"""Case follow-up role.

Answers a customer's question about a case that already reached this turn's
outcome (waiting on a human, or already escalated). It never proposes a new
action and never opens a new ticket — a person already owns the case, so this
role's whole job is to explain the existing decision, not make a new one.
"""

from langchain_core.messages import BaseMessage

from src.agents.triage import format_history
from src.model import LLMUnavailable, invoke_text
from src.prompts import CASE_FOLLOWUP_PROMPT
from src.schemas import Draft, Outcome, PendingApproval


def _case_summary(
    domain: str,
    outcome: Outcome | None,
    draft: Draft | None,
    pending: PendingApproval | None,
    escalation_id: str | None,
) -> str:
    lines = [f"Domain: {domain or 'unknown'}."]

    if pending:
        amount = f"of ${pending.amount:,.2f} " if pending.amount is not None else ""
        lines.append(
            f"A '{pending.action_type}' {amount}is waiting on a human decision. "
            f"Reason: {pending.reason or 'n/a'}. Reference: {pending.approval_id}. "
            f"Status: {pending.status}."
        )
    elif escalation_id:
        lines.append(
            f"This case was handed to a human specialist queue. Reference: {escalation_id}."
        )

    if draft:
        lines.append(f'What was already told to the customer: "{draft.reply}"')

    if outcome:
        lines.append(f"Current outcome: {outcome.value}.")

    return "\n".join(lines)


def _fallback_reply(pending: PendingApproval | None, escalation_id: str | None) -> str:
    if pending:
        return (
            "Your case is still waiting on a human reviewer and nothing has changed "
            f"yet. Your reference is {pending.approval_id}. I don't want to guess at "
            "anything beyond what's already been recorded — a specialist will follow "
            "up with you directly."
        )
    if escalation_id:
        return (
            f"This case is already with a human specialist under reference "
            f"{escalation_id}. I don't have anything new to add beyond what they "
            "already have, and I don't want to guess. They'll follow up with you "
            "directly."
        )
    return (
        "I don't have anything further recorded on this case, and I don't want to "
        "guess. A specialist will follow up with you directly."
    )


def run_case_followup(
    message: str,
    history: list[BaseMessage],
    domain: str,
    outcome: Outcome | None,
    draft: Draft | None,
    pending: PendingApproval | None,
    escalation_id: str | None,
) -> tuple[str, str | None]:
    """Answer a question about a case already in a human's hands.

    Returns `(reply, error)`. Deliberately returns nothing else: outcome,
    pending_approval, and escalation_id belong to the person who already has
    the case, and this role must not change them.
    """

    summary = _case_summary(domain, outcome, draft, pending, escalation_id)

    try:
        reply = invoke_text(
            CASE_FOLLOWUP_PROMPT,
            {
                "case_summary": summary,
                "history": format_history(history),
                "message": message,
            },
        )
    except LLMUnavailable as exc:
        return _fallback_reply(pending, escalation_id), str(exc)

    return reply, None
