"""Critic role.

Two layers. A hard-coded citation guard runs first and can reject a draft
without any model call at all. Only a draft that survives the guard is worth
spending a model call on, and only then does judgement enter the picture.

The guard exists because grounding is checkable. Whether a cited document was
actually retrieved is a set-membership question, not an opinion, and a model
should not be asked opinions it cannot be wrong about.
"""

import re

from src.config import MAX_APPROVAL_LIMIT, MAX_REVISION_ATTEMPTS
from src.model import LLMUnavailable, invoke_structured
from src.prompts import CRITIC_PROMPT
from src.retriever import format_sources, retrieved_ids
from src.schemas import Critique, Draft, Source, Verdict

# Language asserting that a consequential action has already happened.
DONE_CLAIMS = re.compile(
    r"\b("
    r"i(?:'ve| have) (?:refunded|credited|reversed|processed|approved|issued)"
    r"|(?:has|have) been (?:refunded|credited|reversed|processed|approved|applied)"
    r"|the (?:money|funds|refund) (?:is|are|has been|have been) back"
    r"|we(?:'ve| have) (?:refunded|credited|reversed|approved)"
    r"|you(?:'re| are) approved"
    r"|your (?:loan|application) (?:is|has been) approved"
    r")\b",
    re.IGNORECASE,
)


def citation_guard(draft: Draft, sources: list[Source]) -> tuple[bool, list[str], list[str]]:
    """Deterministic grounding check. Runs before the critic model is invoked.

    Returns `(passed, problems, fixes)`. A failure here is an automatic revise
    and costs nothing, which is the cheap safety net underneath the model-based
    validation.
    """

    problems: list[str] = []
    fixes: list[str] = []
    allowed = retrieved_ids(sources)

    if not sources:
        problems.append("No policy was retrieved, so nothing in this draft can be grounded.")
        fixes.append("Do not answer from general knowledge; the case needs a human.")
        return False, problems, fixes

    if not draft.citations:
        problems.append("The draft cites no policy source at all.")
        fixes.append(
            "Ground the reply in the retrieved excerpts and list the document ids you used."
        )

    invented = [c for c in draft.citations if c not in allowed and c.split("#")[0] not in allowed]
    if invented:
        problems.append(
            f"These citations do not appear in what was retrieved: {', '.join(invented)}."
        )
        fixes.append("Cite only document ids from the retrieved excerpts.")

    action = draft.action
    if action and action.action_type != "information_only":
        if not action.citation:
            problems.append(f"The proposed '{action.action_type}' names no authorizing policy.")
            fixes.append(f"Add the document id that authorizes '{action.action_type}'.")
        elif action.citation not in allowed and action.citation.split("#")[0] not in allowed:
            problems.append(
                f"The proposed '{action.action_type}' cites '{action.citation}', "
                "which was not retrieved."
            )
            fixes.append("Cite an authorizing document that appears in the excerpts.")

    if DONE_CLAIMS.search(draft.reply):
        problems.append(
            "The reply tells the customer an action is already complete. Nothing is "
            "complete until the workflow's approval check has run."
        )
        fixes.append(
            "Rewrite it as a request that has been submitted for review, naming who "
            "reviews it."
        )

    return not problems, problems, fixes


def _fallback_critique(problems: list[str], fixes: list[str], reason: str) -> Critique:
    """Guard-only verdict when the model is unavailable.

    Defaults to escalation. With the second line of defense degraded, the safe
    answer is a person, not an approval.
    """

    if problems:
        return Critique(
            verdict=Verdict.REVISE,
            grounded=False,
            problems=problems,
            fixes=fixes,
            rationale=f"Citation guard only; critic model unavailable ({reason}).",
        )

    return Critique(
        verdict=Verdict.ESCALATE,
        grounded=True,
        escalation_reason=(
            "Automated review is unavailable, so no draft is auto-approved. Routed to a human."
        ),
        rationale=f"Citation guard passed but the critic model failed ({reason}).",
    )


def run_critic(
    draft: Draft,
    domain: str,
    intent: str,
    message: str,
    sources: list[Source],
    attempt: int,
) -> tuple[Critique, str | None, bool]:
    """Review a draft.

    Returns `(critique, error, used_model)`. `used_model` is False when the
    citation guard rejected the draft outright, which is worth showing in the
    trace: it is a rejection that cost nothing.
    """

    passed, problems, fixes = citation_guard(draft, sources)

    if not passed:
        return (
            Critique(
                verdict=Verdict.REVISE,
                grounded=False,
                problems=problems,
                fixes=fixes,
                rationale=(
                    "Rejected by the deterministic citation guard before the critic model "
                    "was invoked."
                ),
            ),
            None,
            False,
        )

    action = draft.action
    budget = (
        "The revision budget is spent. If this draft is not acceptable, choose escalate "
        "rather than revise."
        if attempt >= MAX_REVISION_ATTEMPTS
        else "Revisions remain available."
    )

    inputs = {
        "domain": domain,
        "intent": intent,
        "message": message,
        "limit": MAX_APPROVAL_LIMIT,
        "policy": format_sources(sources),
        "reply": draft.reply,
        "action": (
            f"{action.action_type} — {action.description} "
            f"(amount={action.amount}, cites {action.citation})"
            if action
            else "(none proposed)"
        ),
        "citations": ", ".join(draft.citations) or "(none)",
        "notes": draft.notes or "(none)",
        "confidence": draft.confidence,
        "attempt": attempt + 1,
        "maximum": MAX_REVISION_ATTEMPTS + 1,
        "budget": budget,
    }

    try:
        critique = invoke_structured(CRITIC_PROMPT, Critique, inputs)
    except LLMUnavailable as exc:
        return _fallback_critique(problems, fixes, str(exc)), str(exc), False

    return critique, None, True
