"""Node implementations.

Every node takes state and returns a partial update. Nodes own application
logic, the roles in `src.agents` own reasoning, and the tools own system
access. Keeping those separate is what makes the trace readable.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from src.agents.critic import run_critic
from src.agents.fraud import run_fraud_agent
from src.agents.specialists import run_account_agent, run_card_agent, run_loan_agent
from src.agents.triage import run_triage
from src.config import MAX_APPROVAL_LIMIT, MAX_CLARIFFICATIONS, MAX_REVISION_ATTEMPTS
from src.retriever import retrieve
from src.schemas import Domain, Outcome, PendingApproval, ToolCall, TraceEvent, Verdict
from src.state import ChatState
from src.tools import (
    assess_loan_eligibility,
    call_tool,
    create_escalation,
    lookup_accounts,
    lookup_cards,
    lookup_customer,
    lookup_fee_waivers,
    lookup_loans,
    lookup_transactions,
    process_refund,
    request_human_approval,
    score_fraud_risk,
)

logger = logging.getLogger(__name__)

QUEUES = {
    Domain.FRAUD.value: "fraud_investigations",
    Domain.LOAN.value: "consumer_lending",
    Domain.CARD.value: "card_operations",
    Domain.ACCOUNT.value: "deposit_operations",
    Domain.OUT_OF_SCOPE.value: "tier2_support",
}


def trace(node: str, role: str, detail: str) -> list[TraceEvent]:
    return [TraceEvent(node=node, role=role, detail=detail)]


# --------------------------------------------------------------------------
# Intake
# --------------------------------------------------------------------------
def intake_node(state: ChatState) -> dict:
    """Record the incoming message on the conversation."""

    return {
        "messages": [HumanMessage(content=state["customer_message"])],
        "trace": trace("intake", "system", "Customer message received."),
    }


# --------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------
def triage_node(state: ChatState) -> dict:
    """Classify the request and check whether anything is missing."""

    # The current turn is already appended, so it is trimmed off and passed
    # separately. What remains is the earlier conversation, which is what lets
    # triage combine an answer with the question that prompted it.
    history = state.get("messages", [])[:-1]
    result, error = run_triage(state["customer_message"], history)

    detail = (
        f"Domain: {result.domain.value}. Customer: {result.customer_id or 'unknown'}. "
        f"Amount: {result.amount if result.amount is not None else 'n/a'}. "
        f"Missing: {len(result.missing_info)} item(s)."
    )

    update: dict = {
        "triage": result,
        "domain": result.domain.value,
        "questions": result.missing_info,
        "trace": trace("triage", "Triage", detail),
    }

    if error:
        update["degraded"] = True
        update["errors"] = [f"triage: {error}"]
        update["trace"] += trace(
            "triage", "Triage", "Chat model unavailable; keyword classifier used."
        )

    return update


# --------------------------------------------------------------------------
# Ask user
# --------------------------------------------------------------------------
def ask_user_node(state: ChatState) -> dict:
    """Pause the graph and ask the customer for what is missing.

    `interrupt` suspends execution here. The interface shows the question, the
    customer answers, and the graph resumes at this same node with the answer,
    then flows straight back into triage with the fuller history.
    """

    questions = state.get("questions", []) or ["a bit more detail about your request"]
    asked = state.get("clarifications", 0) + 1

    prompt = (
        "Happy to help with that. Before I can go further I need "
        + ("a couple of details:" if len(questions) > 1 else "one detail:")
        + "\n\n"
        + "\n".join(f"- {question}" for question in questions)
        + "\n\nNothing has been changed on your account."
    )

    reply = interrupt({"type": "question", "prompt": prompt, "questions": questions})

    return {
        "messages": [AIMessage(content=prompt), HumanMessage(content=str(reply))],
        "customer_message": str(reply),
        "clarifications": asked,
        "questions": [],
        "trace": trace(
            "ask_user",
            "Triage",
            f"Asked for {len(questions)} missing detail(s); customer replied. "
            f"Clarification {asked} of {MAX_CLARIFFICATIONS}. Returning to triage.",
        ),
    }


# --------------------------------------------------------------------------
# Out of scope
# --------------------------------------------------------------------------
OUT_OF_SCOPE_REPLY = (
    "I'm sorry, I don't have enough information about that request to answer it. "
    "I'm the support assistant for  Bank accounts, so I can help with "
    "card charges and disputes, loans, deposit account questions like fees and "
    "login problems, and reports of unauthorised activity.\n\n"
    "If your question is about one of those, tell me a bit more and I'll pick it "
    "up. For anything else I'd rather say so than guess."
)


def out_of_scope_node(state: ChatState) -> dict:
    """Everything no agent and no policy covers arrives here.

    No retrieval, no draft, no model call. Nothing was retrieved that could
    ground an answer, so producing one would mean inventing it. This node exists
    so that "I don't know" is a real outcome the graph can reach rather than
    something the system has to be talked into.

    From here a banking matter goes on to a person and anything else stops. That
    choice is made once, in `route_after_out_of_scope`, so the diagram shows a
    single edge leaving triage for "nothing here answers this".
    """

    triage = state.get("triage")
    reason = (triage.reasoning if triage else "") or "Outside what this assistant covers."
    bank_related = bool(triage and triage.bank_related)

    if bank_related:
        # Say nothing to the customer yet. The escalation node owns that reply,
        # because what it can promise depends on the ticket being raised.
        return {
            "trace": trace(
                "out_of_scope",
                "Triage",
                "Bank business, but outside the card, loan, account, and fraud "
                f"domains, so no policy was retrieved. {reason} "
                "Handing it to a person rather than guessing.",
            )
        }

    return {
        "outcome": Outcome.CANNOT_ASSIST,
        "outcome_summary": (
            "Declined. The request is outside banking support, so no policy was "
            "retrieved and no answer was drafted."
        ),
        "messages": [AIMessage(content=OUT_OF_SCOPE_REPLY)],
        "trace": trace(
            "out_of_scope",
            "Triage",
            f"Not a banking request, so nothing was retrieved and nothing was "
            f"drafted. {reason} There is nobody at the bank to hand it to either.",
        ),
    }


# --------------------------------------------------------------------------
# Evidence gathering, shared by the specialist nodes
# --------------------------------------------------------------------------
def _gather(state: ChatState, want: set[str]) -> tuple[dict, list[ToolCall], list[str]]:
    """Pull only the records this domain actually needs.

    Each specialist requests a different slice, which is part of what makes
    them distinct rather than one agent with four prompts.
    """

    triage = state["triage"]
    calls: list[ToolCall] = []
    errors: list[str] = []
    evidence: dict = {}

    if not triage or not triage.customer_id:
        return evidence, calls, errors

    customer, record = call_tool(lookup_customer, {"customer_id": triage.customer_id})
    calls.append(record)
    evidence["customer"] = customer
    if not record.ok:
        errors.append(f"lookup_customer failed: {record.error}")

    accounts: list[dict] = []
    if want & {"accounts", "transactions", "cards"}:
        accounts, record = call_tool(lookup_accounts, {"customer_id": triage.customer_id})
        calls.append(record)
        accounts = accounts or []
        evidence["accounts"] = accounts
        if not record.ok:
            errors.append(f"lookup_accounts failed: {record.error}")

    account_ids = [a["account_id"] for a in accounts]
    if triage.account_id and triage.account_id not in account_ids:
        account_ids.append(triage.account_id)

    if "transactions" in want:
        transactions: list[dict] = []
        for account_id in account_ids:
            rows, record = call_tool(lookup_transactions, {"account_id": account_id})
            calls.append(record)
            transactions.extend(rows or [])
            if not record.ok:
                errors.append(f"lookup_transactions failed for {account_id}: {record.error}")
        evidence["transactions"] = transactions

    if "cards" in want:
        cards: list[dict] = []
        for account_id in account_ids:
            rows, record = call_tool(lookup_cards, {"account_id": account_id})
            calls.append(record)
            cards.extend(rows or [])
        evidence["cards"] = cards

    if "loans" in want:
        loans, record = call_tool(lookup_loans, {"customer_id": triage.customer_id})
        calls.append(record)
        evidence["loans"] = loans or []

    if "fees" in want:
        fees, record = call_tool(lookup_fee_waivers, {"customer_id": triage.customer_id})
        calls.append(record)
        evidence["fee_history"] = fees

    return evidence, calls, errors


def _retrieve(state: ChatState, domain: str) -> tuple[list, list[TraceEvent], list[str]]:
    triage = state["triage"]
    query = f"{triage.intent} {state['customer_message']}"[:600] if triage else state["customer_message"]

    try:
        sources = retrieve(query=query, domain=domain)
    except Exception as exc:  # noqa: BLE001 - a vector store outage must not crash the turn
        logger.warning("retrieval failed: %s", exc)
        return [], trace("retrieve", "RAG", f"Vector store unavailable: {exc}"), [f"retrieval: {exc}"]

    best = max((s.score or 0.0) for s in sources) if sources else 0.0
    return (
        sources,
        trace(
            "retrieve",
            "RAG",
            f"Retrieved {len(sources)} chunk(s) filtered to domain '{domain}' plus shared "
            f"governance. Best relevance {best:.3f}.",
        ),
        [],
    )


def _specialist_node(state: ChatState, domain: str, want: set[str], runner, label: str) -> dict:
    """Shared mechanics for the three drafting specialists."""

    triage = state["triage"]
    attempt = state.get("retry_count", 0)

    # On a revision the evidence and sources are already in state. Re-running
    # the lookups would spend tool calls to fetch identical rows.
    if attempt > 0 and state.get("sources"):
        evidence = {
            "customer": state.get("customer_record"),
            "accounts": state.get("account_records", []),
            "transactions": state.get("transactions", []),
            "cards": state.get("cards", []),
            "loans": state.get("loans", []),
            "loan_assessment": state.get("loan_assessment"),
            "fee_history": state.get("fee_history"),
        }
        sources = state["sources"]
        calls, errors, retrieval_trace = [], [], []
    else:
        evidence, calls, errors = _gather(state, want)
        sources, retrieval_trace, retrieval_errors = _retrieve(state, domain)
        errors += retrieval_errors

    # The loan specialist runs a deterministic eligibility assessment. This is
    # the substance of its difference from the other two: a credit
    # recommendation has to be reproducible, so it is computed, not reasoned.
    if domain == Domain.LOAN.value and attempt == 0 and triage and triage.customer_id:
        assessment, record = call_tool(
            assess_loan_eligibility,
            {
                "customer_id": triage.customer_id,
                "requested_amount": triage.amount or 0.0,
                "term_months": 60,
            },
        )
        calls.append(record)
        evidence["loan_assessment"] = assessment

    draft, error = runner(
        triage=triage,
        message=state["customer_message"],
        history=state.get("messages", [])[:-1],
        evidence=evidence,
        sources=sources,
        previous=state.get("draft"),
        critique=state.get("critique"),
        attempt=attempt,
    )

    action = draft.action
    detail = (
        f"{'Revision ' + str(attempt) if attempt else 'Initial draft'}: "
        f"proposes {action.action_type if action else 'nothing'}"
        f"{f' of ${action.amount:,.2f}' if action and action.amount else ''}, "
        f"citing {len(draft.citations)} source(s), confidence {draft.confidence:.2f}."
    )

    update: dict = {
        "customer_record": evidence.get("customer"),
        "account_records": evidence.get("accounts", []),
        "transactions": evidence.get("transactions", []),
        "cards": evidence.get("cards", []),
        "loans": evidence.get("loans", []),
        "loan_assessment": evidence.get("loan_assessment"),
        "fee_history": evidence.get("fee_history"),
        "sources": sources,
        "draft": draft,
        "tool_calls": calls,
        "trace": retrieval_trace + trace(f"{domain}_agent", label, detail),
    }

    if errors:
        update["errors"] = errors
    if error:
        update["degraded"] = True
        update["errors"] = [*errors, f"{domain}_agent: {error}"]
        update["trace"] += trace(
            f"{domain}_agent", label, "Chat model unavailable; issued a holding reply."
        )

    return update


def card_agent_node(state: ChatState) -> dict:
    return _specialist_node(
        state, Domain.CARD.value, {"accounts", "transactions", "cards"},
        run_card_agent, "Card specialist",
    )


def loan_agent_node(state: ChatState) -> dict:
    return _specialist_node(
        state, Domain.LOAN.value, {"loans"}, run_loan_agent, "Loan specialist"
    )


def account_agent_node(state: ChatState) -> dict:
    return _specialist_node(
        state, Domain.ACCOUNT.value, {"accounts", "transactions", "fees"},
        run_account_agent, "Account specialist",
    )


# --------------------------------------------------------------------------
# Fraud
# --------------------------------------------------------------------------
def fraud_agent_node(state: ChatState) -> dict:
    """Write the holding message and the investigator's case summary.

    Never reaches the critic. There is no automated action to validate.
    """

    triage = state["triage"]
    evidence, calls, errors = _gather(state, {"accounts", "transactions", "cards"})
    sources, retrieval_trace, retrieval_errors = _retrieve(state, Domain.FRAUD.value)
    errors += retrieval_errors

    channel = ""
    for transaction in evidence.get("transactions", []):
        if triage and triage.amount and abs(transaction["amount"]) == triage.amount:
            channel = transaction["channel"]
            break

    risk, record = call_tool(
        score_fraud_risk,
        {
            "amount": (triage.amount if triage else 0.0) or 0.0,
            "customer_age": (evidence.get("customer") or {}).get("age", 0) or 0,
            "fraud_hold": bool((evidence.get("customer") or {}).get("fraud_hold", False)),
            "channel": channel,
        },
    )
    calls.append(record)

    draft, error = run_fraud_agent(
        triage=triage,
        message=state["customer_message"],
        history=state.get("messages", [])[:-1],
        evidence=evidence,
        sources=sources,
    )

    detail = (
        f"Case prepared for investigation. Risk {(risk or {}).get('risk_score', 0)} "
        f"({(risk or {}).get('risk_band', 'unknown')}), "
        f"{len((risk or {}).get('triggers', []))} trigger(s). Bypassing the critic: "
        "fraud has no automated resolution to validate."
    )

    update: dict = {
        "customer_record": evidence.get("customer"),
        "account_records": evidence.get("accounts", []),
        "transactions": evidence.get("transactions", []),
        "cards": evidence.get("cards", []),
        "fraud_risk": risk,
        "sources": sources,
        "draft": draft,
        "tool_calls": calls,
        "trace": retrieval_trace + trace("fraud_agent", "Fraud specialist", detail),
    }

    if errors:
        update["errors"] = errors
    if error:
        update["degraded"] = True
        update["errors"] = [*errors, f"fraud_agent: {error}"]

    return update


# --------------------------------------------------------------------------
# Critic
# --------------------------------------------------------------------------
def critic_node(state: ChatState) -> dict:
    """Validate the draft. The citation guard runs before any model call."""

    draft = state["draft"]
    triage = state["triage"]
    attempt = state.get("retry_count", 0)

    critique, error, used_model = run_critic(
        draft=draft,
        domain=state.get("domain", ""),
        intent=triage.intent if triage else "",
        message=state["customer_message"],
        sources=state.get("sources", []),
        attempt=attempt,
    )

    if used_model:
        how = "citation guard passed, then model review"
    elif error:
        how = "citation guard only, model unavailable"
    else:
        how = "rejected by the citation guard, no model call made"

    parts = [
        f"Review {attempt + 1}: {critique.verdict.value} ({how}).",
        f"Grounded: {critique.grounded}.",
    ]
    # Spell the reason out. A log that records a rejection without saying what
    # was wrong with the draft cannot be audited, and after a retry succeeds the
    # rejected draft is no longer in `draft` to inspect.
    if critique.verdict is not Verdict.APPROVE:
        why = critique.problems or [critique.escalation_reason]
        why = [reason for reason in why if reason]
        if why:
            parts.append("Rejected because: " + "; ".join(why) + ".")
        if critique.fixes:
            parts.append("Required: " + "; ".join(critique.fixes) + ".")

    update: dict = {
        "critique": critique,
        "critique_history": [*state.get("critique_history", []), critique],
        "draft_history": [*state.get("draft_history", []), draft],
    }

    if critique.verdict is Verdict.REVISE:
        update["retry_count"] = attempt + 1
        update["critic_feedback"] = critique.fixes
        if attempt + 1 > MAX_REVISION_ATTEMPTS:
            parts.append(
                f"Revision budget of {MAX_REVISION_ATTEMPTS} is spent, so this "
                "escalates rather than being sent back again."
            )
        else:
            parts.append(
                f"Sending back to the {state.get('domain', '?')} agent, "
                f"revision {attempt + 1} of {MAX_REVISION_ATTEMPTS}."
            )

    update["trace"] = trace("critic", "Critic", " ".join(parts))

    if error:
        update["degraded"] = True
        update["errors"] = [f"critic: {error}"]

    return update


# --------------------------------------------------------------------------
# Terminals
# --------------------------------------------------------------------------
def resolved_node(state: ChatState) -> dict:
    """The one automatic path: an approved refund within the limit."""

    draft = state["draft"]
    triage = state["triage"]
    action = draft.action
    calls: list[ToolCall] = []

    if action and action.action_type == "refund" and action.amount:
        result, record = call_tool(
            process_refund,
            {
                "session_id": state["session_id"],
                "customer_id": triage.customer_id if triage else "UNKNOWN",
                "amount": action.amount,
                "reason": action.description,
                "citation": action.citation,
            },
        )
        calls.append(record)
        summary = (
            f"Refund of ${action.amount:,.2f} processed automatically. It is within the "
            f"${MAX_APPROVAL_LIMIT:,.2f} limit and the critic approved the draft. "
            "Simulated in the local demo environment."
        )
        if result is None:
            summary = (
                "The refund could not be recorded because the payment tool failed. "
                "Nothing was moved."
            )
    else:
        summary = "Answered within pre-authorized limits. No money moved."

    return {
        "outcome": Outcome.RESOLVED,
        "outcome_summary": summary,
        "messages": [AIMessage(content=draft.reply)],
        "tool_calls": calls,
        "trace": trace("resolved", "system", summary),
    }


def approval_reason(domain: str, action) -> tuple[str, str]:
    """Human-readable reason and approver title for a parked action.

    Matches the same rules as `needs_human_approval`: loans always need an
    officer; over-limit refunds cite the $50 cap; other consequential actions
    (e.g. reissue_card) need a supervisor because they are not pre-authorized
    for automation — not because of a refund amount.
    """

    if domain == Domain.LOAN.value:
        return (
            "Every credit decision is made by a loan officer, at any amount (POL-AUTH-008).",
            "loan officer",
        )

    action_type = action.action_type if action else "unspecified"
    amount = action.amount if action and action.amount is not None else None

    if action_type == "refund":
        shown = amount if amount is not None else 0.0
        return (
            f"${shown:,.2f} is above the ${MAX_APPROVAL_LIMIT:,.2f} automated "
            "refund limit (POL-AUTH-008).",
            "support supervisor",
        )

    if action_type == "reissue_card":
        return (
            "Blocking and reissuing a card is consequential and is not "
            "pre-authorized for automation (POL-AUTH-008 / POL-CARD-002).",
            "support supervisor",
        )

    if action_type == "loan_decision":
        return (
            "Every credit decision is made by a loan officer, at any amount (POL-AUTH-008).",
            "loan officer",
        )

    return (
        f"Action '{action_type}' is not pre-authorized for automation and needs "
        "a support supervisor (POL-AUTH-008).",
        "support supervisor",
    )


def human_approval_node(state: ChatState) -> dict:
    """Park a consequential action for a person. The graph stops here."""

    draft = state["draft"]
    action = draft.action
    domain = state.get("domain", "")
    reason, approver = approval_reason(domain, action)

    result, record = call_tool(
        request_human_approval,
        {
            "session_id": state["session_id"],
            "action_type": action.action_type if action else "unspecified",
            "description": action.description if action else "Action requiring sign-off",
            "reason": reason,
            "citation": action.citation if action else "POL-AUTH-008",
            "amount": action.amount if action else None,
        },
    )

    if result is None:
        # If the approval queue is unreachable the case must not fall through to
        # resolved. It becomes an escalation instead.
        escalation, escalation_record = call_tool(
            create_escalation,
            {
                "session_id": state["session_id"],
                "customer_id": (state["triage"].customer_id if state["triage"] else "UNKNOWN"),
                "domain": domain,
                "reason": "Approval queue unreachable; routed directly to a human.",
                "queue": QUEUES.get(domain, "tier2_support"),
            },
        )
        return {
            "outcome": Outcome.ESCALATED,
            "outcome_summary": (
                "The approval queue could not be reached, so the case went straight to a "
                "human specialist. Nothing was done to the account."
            ),
            "escalation_id": (escalation or {}).get("escalation_id"),
            "messages": [AIMessage(content=draft.reply)],
            "tool_calls": [record, escalation_record],
            "errors": [f"request_human_approval failed: {record.error}"],
            "trace": trace(
                "human_approval", "system", "Approval submission failed; escalated instead."
            ),
        }

    pending = PendingApproval(
        approval_id=result["approval_id"],
        action_type=result["action_type"],
        description=result["description"],
        amount=result.get("amount"),
        reason=reason,
        citation=result.get("citation", ""),
    )

    return {
        "outcome": Outcome.PENDING_APPROVAL,
        "outcome_summary": (
            f"Recommended '{pending.action_type}' is waiting on a {approver}. "
            "It has NOT been performed."
        ),
        "pending_approval": pending,
        "messages": [AIMessage(content=draft.reply)],
        "tool_calls": [record],
        "trace": trace(
            "human_approval",
            "system",
            f"Queued {pending.approval_id} for a {approver}. {reason} "
            "The workflow stopped short of acting.",
        ),
    }


def escalated_node(state: ChatState) -> dict:
    """Hand the whole case to a human specialist queue."""

    triage = state["triage"]
    draft = state.get("draft")
    critique = state.get("critique")
    domain = state.get("domain", "")

    unsupported = domain == Domain.OUT_OF_SCOPE.value

    if domain == Domain.FRAUD.value:
        reason = "Fraud reports are always investigated by a person; nothing is automated."
    elif unsupported:
        reason = (
            "A banking request outside the card, loan, account, and fraud domains this "
            "assistant covers. No policy was retrieved, so no answer was drafted."
        )
    elif critique and critique.escalation_reason:
        reason = critique.escalation_reason
    elif state.get("retry_count", 0) > MAX_REVISION_ATTEMPTS:
        reason = (
            f"The draft was revised {MAX_REVISION_ATTEMPTS} times and still did not pass "
            "review, so the revision budget is spent."
        )
    elif state.get("clarifications", 0) >= MAX_CLARIFFICATIONS:
        reason = (
            "The request is still unclear after "
            f"{MAX_CLARIFFICATIONS} attempts to clarify it."
        )
    else:
        reason = "Beyond what automated support should handle."

    result, record = call_tool(
        create_escalation,
        {
            "session_id": state["session_id"],
            "customer_id": triage.customer_id if triage else "UNKNOWN",
            "domain": domain or "unknown",
            "reason": reason,
            "queue": QUEUES.get(domain, "tier2_support"),
        },
    )

    if draft:
        reply = draft.reply
    elif unsupported:
        reply = (
            "That's a  Bank matter, but it isn't something I can answer "
            "myself, and I don't want to guess at it. I've passed it to a colleague "
            "who handles this kind of request and they'll follow up with you. "
            "Nothing on your account has changed."
        )
    else:
        reply = (
            "Thanks for your patience. I've passed this to a specialist who handles this "
            "kind of request, and they'll follow up with you. I haven't made any changes "
            "to your account."
        )

    queue = QUEUES.get(domain, "tier2_support")
    escalation_id = (result or {}).get("escalation_id")
    if escalation_id:
        # Give the customer the reference. It is also what makes the handoff
        # checkable: the same id can be read back out of the ticket queue.
        reply = f"{reply}\n\nYour reference for this is **{escalation_id}**."

    return {
        "outcome": Outcome.ESCALATED,
        "outcome_summary": f"Escalated to {queue} as {escalation_id or 'unrecorded'}. {reason}",
        "escalation_id": escalation_id,
        "messages": [AIMessage(content=reply)],
        "tool_calls": [record],
        "trace": trace(
            "escalated",
            "system",
            f"Escalated to {queue} as {(result or {}).get('escalation_id', 'unrecorded')}. {reason}",
        ),
    }
