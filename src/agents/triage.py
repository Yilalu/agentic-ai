"""Triage role: classify the request and decide whether the bank can proceed.

It never reads policy and never proposes an action. Its whole output is the
routing decision the graph acts on.
"""


import re

from langchain_core.messages import BaseMessage

from src.model import LLMUnavailable, invoke_structured
from src.prompts import TRIAGE_PROMPT
from src.schemas import Domain, TriageResult

CUSTOMER_ID = re.compile(r"\bCUST[-\s]?(\d{3,4})\b", re.IGNORECASE)
ACCOUNT_ID = re.compile(r"\bACCT[-\s]?(\d{3,4})\b", re.IGNORECASE)
LAST_FOUR = re.compile(r"(?:ending(?:\s+in)?|last\s*(?:four|4))\D{0,6}(\d{4})", re.IGNORECASE)
AMOUNT = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")

_KEYWORDS: list[tuple[Domain, tuple[str, ...]]] = [
    (Domain.FRAUD,
     ("fraud", "unauthorized", "didn't make", "did not make", "never made",
      "don't recognize", "do not recognize", "never authorized",
      "didn't authorize", "never shopped", "scam", "stolen", "someone took",
      "shouldn't be there", "hacked", "someone else")),
    (Domain.LOAN,
     ("loan", "borrow", "mortgage", "financing", "apply for", "deferral",
      "forbearance", "hardship", "credit application")),
    (Domain.CARD,
     ("duplicate", "twice", "double", "two times", "billed me", "my card",
      "new card", "replacement card", "card fee", "expedited", "chargeback",
      "merchant", "subscription")),
    (Domain.ACCOUNT,
     ("overdraft", "maintenance fee", "log in", "login", "locked out", "password",
      "statement", "wire", "balance", "deposit")),
]


# Vocabulary that marks a message as banking business at all. Without at least
# one of these the fallback has no basis for picking a domain, and picking one
# anyway is how an off-topic question gets answered out of deposit policy.
_BANKING_VOCABULARY = (
    "account", "bank", "card", "charge", "charged", "credit", "debit", "deposit",
    "withdraw", "balance", "transaction", "payment", "pay", "paid", "refund",
    "fee", "overdraft", "statement", "transfer", "wire", "zelle", "loan",
    "mortgage", "borrow", "interest", "atm", "branch", "teller", "cheque",
    "check", "routing", "pin", "login", "log in", "locked out", "password",
    "fraud", "scam", "unauthorized", "dispute", "merchant", "billed", "money",
    "funds", "cust-", "acct-", "$",
)

# Banking business, but not one of the four domains. A person can help.
_UNSUPPORTED_TOPICS = (
    "invest", "brokerage", "portfolio", "stock", "mutual fund", "retirement",
    "401k", "ira ", "insurance", "annuity", "business account", "commercial",
    "close my account", "closing my account", "tax document", "1099", "notary",
    "notarise", "notarize", "safe deposit", "currency exchange", "foreign currency",
    "appointment", "branch hours", "complaint about", "sue ", "lawyer", "attorney",
)


def _heuristic(message: str, history: list[BaseMessage]) -> TriageResult:
    """Deterministic classifier used when the chat model is unavailable.

    Conservative in both directions. It will not guess at fraud or credit when
    unsure, and it will not classify a message at all unless something in it
    reads as banking, because a forced classification is what turns an
    off-topic question into a confidently wrong answer.
    """

    combined = " ".join([*(str(m.content) for m in history), message])
    text = combined.lower()

    # out of scope.
    def out_of_scope(bank_related: bool, why: str) -> TriageResult:
        return TriageResult(
            domain=Domain.OUT_OF_SCOPE,
            bank_related=bank_related,
            intent=message.strip()[:180],
            missing_info=[],
            reasoning=f"Keyword classifier (chat model unavailable): {why}",
        )

    if any(topic in text for topic in _UNSUPPORTED_TOPICS):
        return out_of_scope(True, "a banking topic outside the four supported domains")

    if not any(term in text for term in _BANKING_VOCABULARY):
        return out_of_scope(False, "nothing in the message reads as a banking request")

    domain = Domain.ACCOUNT
    for candidate, words in _KEYWORDS:
        if any(word in text for word in words):
            domain = candidate
            break

    amounts = [float(m.replace(",", "")) for m in AMOUNT.findall(combined)]
    customer = CUSTOMER_ID.search(combined)
    account = ACCOUNT_ID.search(combined)
    card = LAST_FOUR.search(combined)

    missing: list[str] = []
    if not customer:
        missing.append("your customer ID, for example CUST-001")
    if domain in {Domain.CARD, Domain.FRAUD} and not amounts:
        missing.append("the amount of the charge in question")
    if domain is Domain.LOAN and not amounts:
        missing.append("the loan amount you're requesting and what it's for")

    return TriageResult(
        domain=domain,
        intent=message.strip()[:180],
        customer_id=f"CUST-{customer.group(1)}" if customer else None,
        account_id=f"ACCT-{account.group(1)}" if account else None,
        card_last_four=card.group(1) if card else None,
        amount=max(amounts) if amounts else None,
        missing_info=missing,
        reasoning="Keyword classifier used because the chat model was unavailable.",
    )


def format_history(messages: list[BaseMessage], limit: int = 8) -> str:
    if not messages:
        return "(this is the first message)"

    lines = []
    for message in messages[-limit:]:
        who = "Customer" if message.type == "human" else "Support"
        lines.append(f"{who}: {message.content}")
    return "\n".join(lines)


def run_triage(
    message: str, history: list[BaseMessage] | None = None
) -> tuple[TriageResult, str | None]:
    """Classify the request.

    Returns the result plus an error string when the model failed and the
    deterministic fallback produced the answer instead.
    """

    history = history or []

    try:
        result = invoke_structured(
            TRIAGE_PROMPT, TriageResult, {"message": message, "history": format_history(history)}
        )
    except LLMUnavailable as exc:
        return _heuristic(message, history), str(exc)

    # Identifiers are also pulled with a regex across the whole conversation. A
    # model that drops an id the customer gave two turns ago would send the case
    # back to the ask-user loop for no reason.
    combined = " ".join([*(str(m.content) for m in history), message])
    if not result.customer_id:
        match = CUSTOMER_ID.search(combined)
        if match:
            result.customer_id = f"CUST-{match.group(1)}"
    if not result.account_id:
        match = ACCOUNT_ID.search(combined)
        if match:
            result.account_id = f"ACCT-{match.group(1)}"

    result.missing_info = _prune_questions(result.missing_info, result.customer_id)
    return result, None


# Anything the bank can look up from a customer id. Asking the customer for one
# of these sends the conversation into a needless round trip, so they are
# dropped even if the model asks for them.
_LOOKUP_ABLE = (
    "account id", "account_id", "account number", "account no",
    "card number", "card_number", "card id", "last four", "last 4",
    "loan id", "loan number", "loan_id",
    "balance", "transaction history", "statement", "fee history", "waiver",
    "credit score", "credit_score", "credit report",
    "your name", "full name", "address", "phone number", "email address",
    "date of birth",
)


def _prune_questions(questions: list[str], customer_id: str | None) -> list[str]:
    """Drop questions the bank can answer for itself."""

    kept: list[str] = []
    for question in questions:
        lowered = question.lower()
        if any(term in lowered for term in _LOOKUP_ABLE):
            continue
        if customer_id and ("customer id" in lowered or "customer_id" in lowered):
            continue
        kept.append(question)
    return kept
