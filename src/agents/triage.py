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

# Prefer currency forms; also accept bare figures next to loan/charge language.
# "15000" without "$" used to leave amount empty and trap loans in ask_user.
AMOUNT = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")
_AMOUNT_DOLLARS = re.compile(
    r"\b([\d,]+(?:\.\d{1,2})?)\s*(?:dollars|usd)\b", re.IGNORECASE
)
_AMOUNT_K = re.compile(r"\b(\d+(?:\.\d+)?)\s*k\b", re.IGNORECASE)
_AMOUNT_CONTEXT = re.compile(
    r"(?:loan|borrow|refund|charge|charged|billed|amount|for|of|a)\s+"
    r"(?:of\s+|a\s+|for\s+|about\s+|around\s+)?"
    r"([\d,]+(?:\.\d{1,2})?)\b",
    re.IGNORECASE,
)
_AMOUNT_BEFORE_LOAN = re.compile(
    r"\b([\d,]+(?:\.\d{1,2})?)\s+(?:personal\s+|auto\s+|home\s+)?(?:loan|refund)\b",
    re.IGNORECASE,
)

_KEYWORDS: list[tuple[Domain, tuple[str, ...]]] = [
    (Domain.FRAUD,
     ("fraud", "unauthorized", "didn't make", "did not make", "never made",
      "don't recognize", "do not recognize", "never authorized",
      "didn't authorize", "never shopped", "scam", "someone took",
      "shouldn't be there", "hacked", "someone else", "didn't buy",
      "did not buy", "never bought", "not my purchase")),
    (Domain.LOAN,
     ("loan", "borrow", "mortgage", "financing", "apply for", "deferral",
      "forbearance", "hardship", "credit application")),
    (Domain.CARD,
     ("duplicate", "twice", "double", "two times", "billed me", "my card",
      "new card", "replacement card", "lost my card", "lost card",
      "stolen card", "card was stolen", "card fee", "expedited", "chargeback",
      "merchant", "subscription", "reissue")),
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


def _combined_text(message: str, history: list[BaseMessage]) -> str:
    return " ".join([*(str(m.content) for m in history), message])


def _keyword_domain(text: str) -> Domain | None:
    """First matching domain keyword, or None if nothing matches."""

    lowered = text.lower()
    for candidate, words in _KEYWORDS:
        if any(word in lowered for word in words):
            return candidate
    return None


def _parse_amounts(text: str) -> list[float]:
    """Extract money-like amounts from free text."""

    found: list[float] = []

    def add(raw: str, *, thousands: bool = False) -> None:
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            return
        if thousands:
            value *= 1000
        if value > 0:
            found.append(value)

    for match in AMOUNT.findall(text):
        add(match)
    for match in _AMOUNT_DOLLARS.findall(text):
        add(match)
    for match in _AMOUNT_K.findall(text):
        add(match, thousands=True)
    for match in _AMOUNT_CONTEXT.findall(text):
        add(match)
    for match in _AMOUNT_BEFORE_LOAN.findall(text):
        add(match)

    if not found:
        return []

    # Prefer a real money figure over a small term like "5 years".
    significant = [value for value in found if value >= 50]
    return significant or found


def _fill_identifiers(result: TriageResult, combined: str) -> TriageResult:
    """Pull ids and amounts the model dropped from the conversation text."""

    if not result.customer_id:
        match = CUSTOMER_ID.search(combined)
        if match:
            result.customer_id = f"CUST-{match.group(1)}"
    if not result.account_id:
        match = ACCOUNT_ID.search(combined)
        if match:
            result.account_id = f"ACCT-{match.group(1)}"
    if not result.card_last_four:
        match = LAST_FOUR.search(combined)
        if match:
            result.card_last_four = match.group(1)
    if result.amount is None:
        amounts = _parse_amounts(combined)
        if amounts:
            result.amount = max(amounts)
    return result


def _required_missing(result: TriageResult) -> list[str]:
    """Application-owned questions. The model may omit these; the graph must not."""

    if result.domain is Domain.OUT_OF_SCOPE:
        return []

    missing: list[str] = []
    if not result.customer_id:
        missing.append("your customer ID, for example CUST-001")
    if result.domain in {Domain.CARD, Domain.FRAUD} and result.amount is None:
        missing.append("the amount of the charge in question")
    if result.domain is Domain.LOAN and result.amount is None:
        missing.append("the loan amount you are requesting")
    return missing


def _settle_missing_info(result: TriageResult) -> TriageResult:
    """Only application-required questions may pause the graph.

    The model often keeps asking for nice-to-haves (loan purpose, term, etc.)
    even after the customer already answered. Keeping those in missing_info
    trapped the workflow in ask_user forever. Routing uses the hard
    requirements only; specialists can still read purpose from the message.
    """

    if result.domain is Domain.OUT_OF_SCOPE:
        result.missing_info = []
        return result

    result.missing_info = _required_missing(result)
    return result


def _settle_domain(result: TriageResult, combined: str) -> TriageResult:
    """Prefer clear keyword signals when the model picks a conflicting domain."""

    if result.domain is Domain.OUT_OF_SCOPE:
        return result

    hinted = _keyword_domain(combined)
    if hinted is None or hinted is result.domain:
        return result

    # Only override when the keyword family is a strong, higher-priority signal.
    priority = {
        Domain.FRAUD: 4,
        Domain.LOAN: 3,
        Domain.CARD: 2,
        Domain.ACCOUNT: 1,
        Domain.OUT_OF_SCOPE: 0,
    }
    if priority[hinted] > priority.get(result.domain, 0):
        result.reasoning = (
            f"{result.reasoning} Domain corrected from {result.domain.value} to "
            f"{hinted.value} by keyword signal."
        ).strip()
        result.domain = hinted
    return result


def _heuristic(message: str, history: list[BaseMessage]) -> TriageResult:
    """Deterministic classifier used when the chat model is unavailable.

    Conservative in both directions. It will not guess at fraud or credit when
    unsure, and it will not classify a message at all unless something in it
    reads as banking, because a forced classification is what turns an
    off-topic question into a confidently wrong answer.
    """

    combined = _combined_text(message, history)
    text = combined.lower()

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

    domain = _keyword_domain(combined) or Domain.ACCOUNT

    result = TriageResult(
        domain=domain,
        intent=message.strip()[:180],
        missing_info=[],
        reasoning="Keyword classifier used because the chat model was unavailable.",
    )
    result = _fill_identifiers(result, combined)
    return _settle_missing_info(result)


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
    combined = _combined_text(message, history)

    try:
        result = invoke_structured(
            TRIAGE_PROMPT, TriageResult, {"message": message, "history": format_history(history)}
        )
    except LLMUnavailable as exc:
        return _heuristic(message, history), str(exc)

    # Identifiers, domain, and missing_info are settled in application code so
    # a model omission cannot skip ask_user or send the case to the wrong agent.
    result = _fill_identifiers(result, combined)
    result = _settle_domain(result, combined)
    result = _settle_missing_info(result)
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
