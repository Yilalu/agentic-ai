"""
    Demonstration scenarios with the route each one is expected to take.

"""
from dataclasses import dataclass, field

from src.schemas import Outcome


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    kind: str
    message: str
    expected_outcome: Outcome
    expected_route: list[str]
    notes: str
    follow_up: str = ""
    injections: dict[str, object] = field(default_factory=dict)
    # True where the critic's judgement legitimately varies run to run, so an
    # extra agent/critic cycle is acceptable. False where the loop itself is the
    # thing being demonstrated and the count must match exactly.
    route_may_loop: bool = False
    # True for the scenario that deliberately spends the revision budget. Every
    # other scenario must finish at or under the cap; this one must land exactly
    # one past it, which is the signal the router terminates on.
    expect_budget_exhausted: bool = False


SCENARIOS: list[Scenario] = [
    Scenario(
        key="happy",
        label="1. Happy path — duplicate $42 charge, refunded automatically",
        kind="Happy path",
        message=(
            "Hi, I'm CUST-001. Summit Outdoors charged me $42.00 twice on July 29 and I "
            "only bought one thing. Can you sort that out?"
        ),
        expected_outcome=Outcome.RESOLVED,
        expected_route=["intake", "triage", "card_agent", "critic", "resolved"],
        notes=(
            "The transaction list shows the same merchant and amount posting twice, so "
            "the card agent proposes a $42 refund. That is at or below the $50 automated "
            "limit and the domain is not loan, so this is the one fully automatic path "
            "in the system."
        ),
    ),
    Scenario(
        key="branch_approval",
        label="2. Branching — overdraft fees over the cap need a supervisor",
        kind="Branching path + human approval",
        message=(
            "CUST-004 here. On July 30 you hit me with six separate $35 overdraft fees, "
            "$210 in one day. I thought there was a limit on how many you can charge."
        ),
        expected_outcome=Outcome.PENDING_APPROVAL,
        expected_route=["intake", "triage", "account_agent", "critic", "human_approval"],
        notes=(
            "A different domain than scenario 1, so a different metadata filter, a "
            "different policy set, a different agent, and a different terminal. The cap "
            "is three fees per business day, so three of the six were charged in error "
            "and the bank owes $105. That is above the $50 automated limit, so the graph "
            "stops and waits for a person. Note that the agent has to apply the cap "
            "rather than refund what the customer asked for."
        ),
        route_may_loop=True,
    ),
    Scenario(
        key="missing_info",
        label="3. Missing information — vague refund request, then the customer answers",
        kind="Missing information (pause and resume)",
        message="I want a refund on something, this is ridiculous.",
        follow_up=(
            "Sorry. It's CUST-001. Summit Outdoors charged me $42.00 twice on July 29."
        ),
        expected_outcome=Outcome.RESOLVED,
        expected_route=[
            "intake", "triage", "ask_user", "wait_for_user", "triage",
            "card_agent", "critic", "resolved",
        ],
        notes=(
            "Triage finds nothing to work with, so the graph hits ask_user, pauses at "
            "wait_for_user, and the turn genuinely stops. The customer's reply resumes "
            "the same thread and flows straight back into triage, which now has enough "
            "to route normally."
        ),
    ),
    Scenario(
        key="fraud",
        label="4. Escalation — unauthorized charge bypasses the critic",
        kind="Escalation",
        message=(
            "This is CUST-001. There's an $812.44 charge from Northgate Electronics on "
            "my account from July 30 and I have never shopped there in my life."
        ),
        expected_outcome=Outcome.ESCALATED,
        expected_route=["intake", "triage", "fraud_agent", "escalated"],
        notes=(
            "The fraud agent never reaches the critic. There is no draft resolution to "
            "validate, only a case to hand to an investigator. This is the route that "
            "guarantees nothing about a fraud claim is ever auto-approved."
        ),
    ),
    Scenario(
        key="loan",
        label="5. Loan — recommendation always goes to a loan officer",
        kind="Human approval, unconditional",
        message=(
            "This is CUST-001. I'd like to apply for a $15,000 personal loan over five "
            "years to remodel my kitchen."
        ),
        expected_outcome=Outcome.PENDING_APPROVAL,
        expected_route=["intake", "triage", "loan_agent", "critic", "human_approval"],
        notes=(
            "The loan agent runs a deterministic eligibility assessment: credit score, "
            "debt-to-income including the proposed payment, income, and tenure. Even when "
            "every criterion passes, the route is human approval. Amount is irrelevant; "
            "credit decisions are never automated."
        ),
    ),
    Scenario(
        key="revision",
        label="6. Revision loop — first draft rejected by the citation guard",
        kind="Controlled loop within budget",
        message=(
            "I'm CUST-001. Summit Outdoors billed me $42.00 twice on July 29. Please "
            "refund the duplicate."
        ),
        expected_outcome=Outcome.RESOLVED,
        expected_route=[
            "intake", "triage", "card_agent", "critic", "card_agent", "critic", "resolved",
        ],
        notes=(
            "The first draft has its citations stripped, so the deterministic guard "
            "rejects it before the critic model is even called. The agent redrafts with "
            "the guard's specific feedback and passes on attempt two, one revision inside "
            "the budget of two."
        ),
        injections={"force_bad_drafts": 1},
    ),
    Scenario(
        key="tool_failure",
        label="7. Failure path — customer lookup unavailable",
        kind="Tool failure",
        message=(
            "CUST-006 here, I've been locked out of online banking since yesterday. It "
            "says my profile is temporarily locked."
        ),
        expected_outcome=Outcome.RESOLVED,
        expected_route=["intake", "triage", "account_agent", "critic", "resolved"],
        notes=(
            "lookup_customer fails on every retry. The runner records the failure, the "
            "node continues on partial evidence, and the answer still comes from "
            "retrieved policy rather than crashing the turn. Working from thinner "
            "evidence, the critic sometimes asks for one revision first; both routes "
            "are correct."
        ),
        injections={"force_tool_failure": "lookup_customer"},
        route_may_loop=True,
    ),
    Scenario(
        key="model_failure",
        label="8. Failure path — chat model outage",
        kind="Model failure",
        message=(
            "I'm CUST-001 and Summit Outdoors charged me $42.00 twice on July 29."
        ),
        expected_outcome=Outcome.ESCALATED,
        expected_route=["intake", "triage", "card_agent", "critic", "escalated"],
        notes=(
            "With the model gone, triage falls back to the keyword classifier, the agent "
            "issues a holding reply that promises nothing, and the critic refuses to "
            "auto-approve. The case lands with a human, which is the correct failure mode "
            "for a bank."
        ),
        injections={"force_llm_failure": True},
    ),
    Scenario(
        key="off_topic",
        label="9. Out of scope — not a banking question at all",
        kind="Refusal, no answer invented",
        message=(
            "Hey, what's the weather going to be like in Boston this weekend? And can "
            "you write me a short poem about it?"
        ),
        expected_outcome=Outcome.CANNOT_ASSIST,
        expected_route=["intake", "triage", "out_of_scope"],
        notes=(
            "Triage classifies this as out of scope and the graph stops immediately. "
            "Nothing is retrieved, no specialist runs, no ticket is raised, and no "
            "answer is drafted. Without this route the request would fall through to "
            "the account agent, which would retrieve deposit policy and answer "
            "confidently out of documents that have nothing to do with the question."
        ),
    ),
    Scenario(
        key="unsupported",
        label="10. Out of scope — a banking question no specialist covers",
        kind="Refusal with a handoff",
        message=(
            "I'm CUST-002. I'd like to move $40,000 from my checking into a brokerage "
            "account and get some advice on which index funds to pick."
        ),
        expected_outcome=Outcome.ESCALATED,
        expected_route=["intake", "triage", "out_of_scope", "escalated"],
        notes=(
            "The same edge out of triage as scenario 9, because the judgement is "
            "identical: no agent owns this and no policy covers it. What differs is "
            "what happens next. The `bank_related` flag sends this one to a person, "
            "while the weather question stops. Raising a ticket for the weather is "
            "theatre, and refusing an investment question outright abandons a customer "
            "somebody at the bank can actually help. Note the temptation the router "
            "resists — this involves a checking account and a dollar amount, so it "
            "looks like `account` work."
        ),
    ),
    Scenario(
        key="revision_twice",
        label="11. Revision loop — two send-backs, then accepted on the last try",
        kind="Controlled loop at the budget",
        message=(
            "I'm CUST-001. Summit Outdoors billed me $42.00 twice on July 29. Please "
            "refund the duplicate."
        ),
        expected_outcome=Outcome.RESOLVED,
        expected_route=[
            "intake", "triage",
            "card_agent", "critic",
            "card_agent", "critic",
            "card_agent", "critic",
            "resolved",
        ],
        notes=(
            "The first two drafts have their citations stripped, so the guard rejects "
            "both. The third is left intact and is accepted. This is the scenario the "
            "brief asks for: the controlled loop runs to its limit of two revisions "
            "without exceeding it, and the case still reaches a normal outcome. Compare "
            "the retry counter, which ends at 2 of 2, against scenario 12 where it "
            "reaches 3 and the route changes."
        ),
        injections={"force_bad_drafts": 2},
    ),
    Scenario(
        key="revision_exhausted",
        label="12. Revision loop — budget spent, so the case escalates",
        kind="Loop termination",
        message=(
            "I'm CUST-001. Summit Outdoors billed me $42.00 twice on July 29. Please "
            "refund the duplicate."
        ),
        expected_outcome=Outcome.ESCALATED,
        expected_route=[
            "intake", "triage",
            "card_agent", "critic",
            "card_agent", "critic",
            "card_agent", "critic",
            "escalated",
        ],
        notes=(
            "Every draft is broken, so no retry can ever satisfy the guard. The third "
            "rejection pushes retry_count past the budget and the router sends the case "
            "to a human instead of asking for a fourth draft. This is the termination "
            "condition that makes the loop safe: identical route to scenario 11 up to "
            "the last review, then a different terminal. Note what does not happen — "
            "the agent never gives up and invents an answer to escape the loop, and no "
            "refund is issued on a draft the critic never accepted."
        ),
        injections={"force_bad_drafts": 99},
        expect_budget_exhausted=True,
    ),
]

SCENARIOS_BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}
