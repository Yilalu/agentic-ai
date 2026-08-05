"""The three drafting specialists: card, loan, and account.

They share an input shape and a fallback, and nothing else. Each one gets a
different prompt, a different slice of evidence, and a different set of actions
it is permitted to propose, which is what separates them.
"""


import json

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agents.triage import format_history
import src.config as config
from src.config import MAX_APPROVAL_LIMIT, MAX_REVISION_ATTEMPTS
from src.model import LLMUnavailable, invoke_structured
from src.prompts import ACCOUNT_PROMPT, CARD_PROMPT, LOAN_PROMPT, REVISION_BLOCK
from src.retriever import format_sources, retrieved_ids
from src.schemas import Critique, Domain, Draft, ProposedAction, Source, TriageResult


def format_evidence(evidence: dict) -> str:
    """Render whatever bank records this specialist was given."""

    parts: list[str] = []

    customer = evidence.get("customer")
    if customer and customer.get("found"):
        parts.append(
            "Customer: {full_name} ({customer_id}), tier {tier}, age {age}, "
            "customer since {since}, fraud_hold={fraud_hold}".format(**customer)
        )
    else:
        parts.append("Customer: not found in bank systems.")

    for account in evidence.get("accounts", []):
        parts.append(
            f"Account {account['account_id']}: {account['product']}, "
            f"balance ${account['balance']:,.2f}, status {account['status']}"
        )

    transactions = evidence.get("transactions", [])
    if transactions:
        rows = "\n".join(
            f"  {t['posted_on']}  {t['descriptor']:<40s} ${t['amount']:>10,.2f}  "
            f"[{t['channel']}/{t['status']}]"
            for t in transactions
        )
        parts.append(f"Recent transactions:\n{rows}")

    cards = evidence.get("cards", [])
    if cards:
        rows = "\n".join(
            f"  {c['card_id']} ending {c['last_four']} ({c['card_type']}), "
            f"status {c['status']}, expires {c['expires_on']}"
            for c in cards
        )
        parts.append(f"Cards:\n{rows}")

    loans = evidence.get("loans", [])
    if loans:
        rows = "\n".join(
            f"  {loan['loan_id']} {loan['loan_type']}, balance ${loan['balance']:,.2f} "
            f"at {loan['rate']}%, payment ${loan['monthly_payment']:,.2f}, "
            f"{loan['status']} ({loan['days_past_due']} days past due)"
            for loan in loans
        )
        parts.append(f"Existing loans:\n{rows}")

    assessment = evidence.get("loan_assessment")
    if assessment:
        parts.append(
            "Loan eligibility assessment (computed deterministically):\n"
            + json.dumps(assessment, indent=2)
        )

    fee_history = evidence.get("fee_history")
    if fee_history:
        parts.append(
            f"Fee waivers already used: {fee_history['waiver_count']} totalling "
            f"${fee_history['waived_total']:,.2f}."
        )

    parts.append(f"Automated refund limit: ${MAX_APPROVAL_LIMIT:,.2f}.")

    return "\n\n".join(parts)


def _fallback_draft(sources: list[Source], reason: str) -> Draft:
    """Safe holding reply when the model cannot draft.

    Commits the bank to nothing and carries low confidence, so the critic sends
    it to a human rather than out to the customer.
    """

    return Draft(
        reply=(
            "Thanks for getting in touch, and sorry for the trouble. I've recorded the "
            "details and a support specialist will look at your account and follow up "
            "within one business day. I don't want to give you an answer on this until "
            "a person has reviewed it directly."
        ),
        action=ProposedAction(
            action_type="information_only",
            description="Hand to a human specialist for manual handling.",
            citation="POL-AUTH-008",
        ),
        citations=[source.doc_id for source in sources[:2]],
        notes=f"Automated drafting unavailable ({reason}). Nothing was promised.",
        confidence=0.2,
    )


def _draft(
    prompt: ChatPromptTemplate,
    triage: TriageResult,
    message: str,
    history: list[BaseMessage],
    evidence: dict,
    sources: list[Source],
    previous: Draft | None,
    critique: Critique | None,
    attempt: int,
) -> tuple[Draft, str | None]:
    revision = ""
    if attempt > 0 and previous and critique:
        revision = REVISION_BLOCK.format(
            attempt=attempt,
            maximum=MAX_REVISION_ATTEMPTS,
            previous=previous.reply,
            problems="\n".join(f"- {p}" for p in critique.problems) or "- (none listed)",
            fixes="\n".join(f"- {f}" for f in critique.fixes) or "- (none listed)",
        )

    inputs = {
        "domain": triage.domain.value,
        "intent": triage.intent,
        "history": format_history(history),
        "message": message,
        "evidence": format_evidence(evidence),
        "policy": format_sources(sources),
        "revision": revision,
    }

    try:
        draft = invoke_structured(prompt, Draft, inputs)
    except LLMUnavailable as exc:
        return settle_citations(_fallback_draft(sources, str(exc)), sources, attempt), str(exc)

    return settle_citations(draft, sources, attempt), None


def settle_citations(draft: Draft, sources: list[Source], attempt: int) -> Draft:
    """Prune citations the retriever never returned, then apply the demo switch.

    Pruning matters on its own: the critic's guard would catch a fabricated
    document id anyway, but removing it here keeps it out of the evidence panel
    entirely.
    """

    allowed = retrieved_ids(sources)
    draft.citations = [c for c in draft.citations if c in allowed or c.split("#")[0] in allowed]

    if attempt < config.FORCE_BAD_DRAFTS:
        draft.citations = []
        if draft.action:
            draft.action.citation = ""
        draft.notes = (
            f"{draft.notes} [demo: citations stripped on attempt {attempt + 1} "
            "to trigger the guard]"
        ).strip()

    return draft


def run_card_agent(**kwargs) -> tuple[Draft, str | None]:
    """Disputes, duplicate charges, card fees, and card servicing."""
    return _draft(CARD_PROMPT, **kwargs)


def run_loan_agent(**kwargs) -> tuple[Draft, str | None]:
    """New applications and existing loan servicing. Recommends, never decides."""
    return _draft(LOAN_PROMPT, **kwargs)


def run_account_agent(**kwargs) -> tuple[Draft, str | None]:
    """Deposit account access, fees, statements, and general questions."""
    return _draft(ACCOUNT_PROMPT, **kwargs)


SPECIALISTS = {
    Domain.CARD.value: run_card_agent,
    Domain.LOAN.value: run_loan_agent,
    Domain.ACCOUNT.value: run_account_agent,
}
