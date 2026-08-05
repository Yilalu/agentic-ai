"""Read-only tools. None of these change any record.

They are LangChain tools so they can be bound to a model, but the graph calls
them through `src.tools.runner` so every invocation is recorded in state
whether or not a model chose it.
"""

import sqlite3

from langchain_core.tools import tool

from src.config import BANK_DB
from src.retriever import retrieve


class ToolError(RuntimeError):
    """A tool could not complete. Distinct from 'found nothing'."""


def connect() -> sqlite3.Connection:
    if not BANK_DB.exists():
        raise ToolError("bank database is unavailable; run scripts.create_data.setup_bank_db()")
    conn = sqlite3.connect(BANK_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(query: str, params: tuple = ()) -> list[dict]:
    conn = connect()
    try:
        return [dict(row) for row in conn.execute(query, params).fetchall()]
    except sqlite3.Error as exc:
        raise ToolError(f"bank query failed: {exc}") from exc
    finally:
        conn.close()


@tool
def lookup_customer(customer_id: str) -> dict:
    """Look up a customer profile by id, for example CUST-001. Read-only."""
    rows = _rows("SELECT * FROM customers WHERE customer_id = ?", (customer_id.upper(),))
    if not rows:
        return {"found": False, "customer_id": customer_id.upper()}

    record = rows[0]
    record["found"] = True
    record["fraud_hold"] = bool(record["fraud_hold"])
    return record


@tool
def lookup_accounts(customer_id: str) -> list[dict]:
    """List the deposit accounts belonging to a customer. Read-only."""
    return _rows("SELECT * FROM accounts WHERE customer_id = ?", (customer_id.upper(),))


@tool
def lookup_transactions(account_id: str, limit: int = 12) -> list[dict]:
    """List recent transactions for an account, newest first. Read-only."""
    return _rows(
        "SELECT * FROM transactions WHERE account_id = ? ORDER BY posted_on DESC LIMIT ?",
        (account_id.upper(), limit),
    )


@tool
def lookup_cards(account_id: str) -> list[dict]:
    """List cards issued against an account with their status. Read-only."""
    return _rows("SELECT * FROM cards WHERE account_id = ?", (account_id.upper(),))


@tool
def lookup_loans(customer_id: str) -> list[dict]:
    """List existing loans held by a customer. Read-only."""
    return _rows("SELECT * FROM loans WHERE customer_id = ?", (customer_id.upper(),))


@tool
def lookup_fee_waivers(customer_id: str) -> dict:
    """Summarize courtesy fee waivers a customer has already used. Read-only."""
    rows = _rows(
        "SELECT * FROM fee_waivers WHERE customer_id = ? ORDER BY waived_on DESC",
        (customer_id.upper(),),
    )
    return {
        "customer_id": customer_id.upper(),
        "waiver_count": len(rows),
        "waived_total": round(sum(row["amount"] for row in rows), 2),
        "waivers": rows,
    }


@tool
def lookup_ticket(ticket_id: str) -> dict:
    """Read back one escalation ticket by id, for example ESC-1A2B3C4D. Read-only.

    The counterpart to `create_escalation`. Without a way to read the queue the
    handoff is unverifiable: the interface would be asserting that a specialist
    picked the case up with nothing to show for it.
    """
    rows = _rows("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id.upper(),))
    if not rows:
        return {"found": False, "ticket_id": ticket_id.upper()}

    record = rows[0]
    record["found"] = True
    record["summary"] = record.get("reason", "")
    return record


@tool
def lookup_specialist_queue(queue: str = "", limit: int = 25) -> list[dict]:
    """List escalation tickets waiting on a human, newest first. Read-only.

    Pass a queue name such as `fraud_investigations` to see one desk, or leave it
    empty for every open case.
    """
    if queue:
        rows = _rows(
            "SELECT * FROM tickets WHERE queue = ? ORDER BY created_at DESC LIMIT ?",
            (queue, limit),
        )
    else:
        rows = _rows("SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?", (limit,))

    for row in rows:
        row["summary"] = row.get("reason", "")
    return rows


@tool
def search_policy(query: str, domain: str = "", doc_types: str = "") -> list[dict]:
    """Semantic search over bank policy and knowledge base documents.

    `domain` filters to the owning specialist: card, loan, account, or fraud.
    `doc_types` is an optional comma-separated filter such as `policy,regulation`.
    Read-only.
    """
    types = [t.strip() for t in doc_types.split(",") if t.strip()] or None
    return [
        {
            "doc_id": source.doc_id,
            "title": source.title,
            "doc_type": source.doc_type,
            "score": source.score,
            "content": source.content,
        }
        for source in retrieve(query=query, domain=domain or None, doc_types=types)
    ]


@tool
def assess_loan_eligibility(
    customer_id: str,
    requested_amount: float,
    term_months: int = 60,
) -> dict:
    """Score a loan application against the POL-LOANORIG-011 criteria.

    Pulls the customer's credit profile, computes debt-to-income including the
    proposed new payment, and returns a recommendation tier. Deterministic on
    purpose: a credit recommendation must be reproducible and explainable, not
    a model's impression. The recommendation is still never a decision.
    Read-only.
    """
    rows = _rows("SELECT * FROM credit_profiles WHERE customer_id = ?", (customer_id.upper(),))
    if not rows:
        return {"assessable": False, "reason": "No credit profile on file for this customer."}

    profile = rows[0]
    score = profile["credit_score"]
    annual_income = profile["annual_income"]
    monthly_debt = profile["monthly_debt"]
    employment_months = profile["employment_months"]
    delinquencies = profile["delinquencies_24m"]

    if annual_income <= 0:
        return {"assessable": False, "reason": "No verifiable income on file."}

    # Indicative rate by credit tier, per the product parameters.
    rate = 8.99 if score >= 760 else 12.99 if score >= 700 else 17.99 if score >= 660 else 24.99
    monthly_rate = rate / 100 / 12
    if monthly_rate > 0:
        factor = (1 + monthly_rate) ** term_months
        new_payment = requested_amount * monthly_rate * factor / (factor - 1)
    else:
        new_payment = requested_amount / term_months

    monthly_income = annual_income / 12
    dti = (monthly_debt + new_payment) / monthly_income

    failures: list[str] = []
    if score < 660:
        failures.append(f"Credit score {score} is below the 660 minimum")
    if dti > 0.43:
        failures.append(f"Debt-to-income {dti:.1%} exceeds the 43% maximum")
    if annual_income < 25000:
        failures.append(f"Income ${annual_income:,.0f} is below the $25,000 minimum")
    if employment_months < 12:
        failures.append(f"Employment of {employment_months} months is under the 12-month minimum")
    if delinquencies > 0:
        failures.append(f"{delinquencies} delinquency(ies) of 60+ days in the last 24 months")

    if not failures:
        tier = "recommend_approval"
    elif score < 620 or dti > 0.50 or annual_income < 25000 or delinquencies > 0:
        tier = "recommend_decline"
    else:
        tier = "refer"

    return {
        "assessable": True,
        "recommendation": tier,
        "credit_score": score,
        "indicative_rate": rate,
        "requested_amount": round(requested_amount, 2),
        "term_months": term_months,
        "estimated_monthly_payment": round(new_payment, 2),
        "debt_to_income": round(dti, 4),
        "debt_to_income_pct": f"{dti:.1%}",
        "annual_income": annual_income,
        "employment_months": employment_months,
        "criteria_failed": failures,
        "policy": "POL-LOANORIG-011",
        "note": "A recommendation only. A loan officer makes every credit decision.",
    }


@tool
def score_fraud_risk(
    amount: float = 0.0,
    customer_age: int = 0,
    fraud_hold: bool = False,
    channel: str = "",
) -> dict:
    """Score a suspected fraud case against the escalation triggers.

    Deterministic so that escalation never depends on a model's mood.
    Read-only.
    """
    triggers: list[str] = []
    score = 0

    if amount >= 5000:
        triggers.append(f"Exposure ${amount:,.2f} meets the $5,000 escalation threshold")
        score += 40
    elif amount >= 1000:
        score += 20

    if customer_age >= 65:
        triggers.append("Customer is 65 or older; elder financial exploitation rules apply")
        score += 25

    if fraud_hold:
        triggers.append("Account already carries a fraud hold")
        score += 30

    if channel in {"zelle", "wire"}:
        triggers.append(f"Funds left over an irreversible rail ({channel})")
        score += 15

    score = min(score, 100)
    return {
        "risk_score": score,
        "risk_band": "high" if score >= 50 else "medium" if score >= 25 else "low",
        "triggers": triggers,
    }


READ_ONLY_TOOLS = [
    lookup_customer,
    lookup_accounts,
    lookup_transactions,
    lookup_cards,
    lookup_loans,
    lookup_fee_waivers,
    lookup_ticket,
    lookup_specialist_queue,
    search_policy,
    assess_loan_eligibility,
    score_fraud_risk,
]
