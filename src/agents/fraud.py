"""Fraud role.

Structurally different from the three drafting specialists: it produces no
resolution to validate, so it never reaches the critic. A fraud report is a
case to hand to an investigator, not an action to approve. Routing it straight
to escalation is the point, not a shortcut.
"""

from langchain_core.messages import BaseMessage

from src.agents.specialists import format_evidence
from src.agents.triage import format_history
from src.model import LLMUnavailable, invoke_structured
from src.prompts import FRAUD_PROMPT
from src.retriever import format_sources, retrieved_ids
from src.schemas import Draft, ProposedAction, Source, TriageResult


def _fallback(sources: list[Source], reason: str) -> Draft:
    return Draft(
        reply=(
            "Thank you for reporting this. I've flagged it for our fraud team and an "
            "investigator will contact you. I'm not able to tell you the outcome yet, "
            "and I don't want to guess at it. Please don't use the affected card or "
            "account details in the meantime."
        ),
        action=ProposedAction(
            action_type="information_only",
            description="Route the fraud report to a human investigator.",
            citation="POL-AUTH-008",
        ),
        citations=[source.doc_id for source in sources[:2]],
        notes=f"Automated drafting unavailable ({reason}). Escalated with no commitment made.",
        confidence=0.2,
    )


def run_fraud_agent(
    triage: TriageResult,
    message: str,
    history: list[BaseMessage],
    evidence: dict,
    sources: list[Source],
) -> tuple[Draft, str | None]:
    """Write the customer's holding message and the investigator's case summary."""

    inputs = {
        "domain": triage.domain.value,
        "intent": triage.intent,
        "history": format_history(history),
        "message": message,
        "evidence": format_evidence(evidence),
        "policy": format_sources(sources),
        "revision": "",
    }

    try:
        draft = invoke_structured(FRAUD_PROMPT, Draft, inputs)
    except LLMUnavailable as exc:
        return _fallback(sources, str(exc)), str(exc)

    allowed = retrieved_ids(sources)
    draft.citations = [c for c in draft.citations if c in allowed or c.split("#")[0] in allowed]

    # A fraud reply must never carry a money-moving action, whatever the model
    # proposed. Escalation is the only outcome available on this path.
    if draft.action and draft.action.action_type != "information_only":
        draft.action = ProposedAction(
            action_type="information_only",
            description="Route the fraud report to a human investigator.",
            citation=draft.action.citation or "POL-AUTH-008",
        )

    return draft, None
