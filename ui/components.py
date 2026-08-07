"""Rendering helpers for the chat interface."""

import json

import pandas as pd
import streamlit as st

from src.config import MAX_REVISION_ATTEMPTS
from src.schemas import Outcome
from src.tools import lookup_specialist_queue, lookup_ticket

OUTCOME_STYLE: dict[str, tuple[str, str]] = {
    Outcome.RESOLVED.value: ("success", "Resolved automatically"),
    Outcome.PENDING_APPROVAL.value: ("warning", "Recommended, waiting on a human"),
    Outcome.ESCALATED.value: ("error", "Escalated to a human specialist"),
    Outcome.AWAITING_INFO.value: ("info", "Paused, waiting on the customer"),
    Outcome.CANNOT_ASSIST.value: ("info", "Declined — outside banking support"),
}

DOMAIN_LABEL = {
    "card": "Card specialist",
    "loan": "Loan specialist",
    "account": "Account specialist",
    "fraud": "Fraud specialist",
    "out_of_scope": "None — declined",
}

NODES = [
    ("intake", "Intake"),
    ("triage", "Triage\nclassify + extract"),
    ("ask_user", "Ask user"),
    ("wait_for_user", "Wait for reply"),
    ("card_agent", "Card agent\ndisputes + card fees"),
    ("loan_agent", "Loan agent\nrecommend only"),
    ("account_agent", "Account agent\naccess + deposit fees"),
    ("fraud_agent", "Fraud agent\nbypasses critic"),
    ("out_of_scope", "Out of scope\nnothing to ground an answer"),
    ("critic", "Critic"),
    ("resolved", "Resolved\nrefund ≤ limit"),
    ("human_approval", "Human approval\nover limit / any loan"),
    ("escalated", "Escalated\nhuman takes over"),
]

# Columns, left to right. Graphviz is told these explicitly rather than being
# left to infer them, because the revise edges point backwards and inferring
# ranks from a cyclic graph moves the critic in front of the agents.
COLUMNS = [
    ["intake"],
    ["ask_user", "wait_for_user", "triage"],
    ["card_agent", "loan_agent", "account_agent", "fraud_agent", "out_of_scope"],
    ["critic"],
    ["resolved", "human_approval", "escalated"],
]

EDGES = [
    ("intake", "triage", ""), # first step 
    ("triage", "ask_user", "missing info"),
    ("ask_user", "wait_for_user", "pause"),
    ("wait_for_user", "triage", "customer replies"),
    ("triage", "card_agent", ""),
    ("triage", "loan_agent", ""),
    ("triage", "account_agent", ""),
    ("triage", "fraud_agent", ""),
    ("triage", "out_of_scope", ""),
    ("out_of_scope", "escalated", "bank business"),
    ("card_agent", "critic", ""),
    ("loan_agent", "critic", ""),
    ("account_agent", "critic", ""),
    ("fraud_agent", "escalated", "always"),
    ("critic", "resolved", "approve"),
    ("critic", "human_approval", "over limit / loan"),
    ("critic", "escalated", "escalate"),
    # One revise edge per specialist. The critic returns a draft to whichever
    # agent wrote it.
    ("critic", "card_agent", "revise"),
    ("critic", "loan_agent", "revise"),
    ("critic", "account_agent", "revise"),
]

# Edges drawn without influencing layout. Two kinds: arrows that point backwards
# (the revise loop, and ask_user returning to triage), and arrows that skip a
# column (triage straight to escalated). Letting either drive ranking drags the
# terminal column left and reorders the pipeline.
DECORATIVE_EDGES = {
    # Clarification is a detour off triage rather than a step past it.
    ("triage", "ask_user"),
    ("ask_user", "wait_for_user"),
    ("wait_for_user", "triage"),
    ("critic", "card_agent"),
    ("critic", "loan_agent"),
    ("critic", "account_agent"),
}

PALETTE = {
    "resolved": "#2e7d32",
    "human_approval": "#b26a00",
    "escalated": "#b3261e",
    "out_of_scope": "#5f6368",
    "ask_user": "#7a4a13",
    "wait_for_user": "#7a4a13",
    "critic": "#4a3fbf",
    "triage": "#4a3fbf",
}

KNOWN_NODES = {node_id for node_id, _ in NODES}


def traversed_path(visited: list[str]) -> list[str]:
    """The ordered nodes this turn actually ran.

    Drops trace entries that are not graph nodes, such as the retrieval step,
    and collapses a node that logged twice into one visit. A genuine revision
    still appears twice, which is the point.
    """
    path: list[str] = []
    for node in visited:
        if node in KNOWN_NODES and (not path or path[-1] != node):
            path.append(node)
    return path


def workflow_dot(visited: list[str]) -> str:

    path = traversed_path(visited)
    seen = set(path)
    # Edges are lit by the transitions that actually happened, not by whether
    # both endpoints were visited. Otherwise every card case would light up the
    # revise arrow purely for having passed through the card agent and the
    # critic at some point.
    taken = set(zip(path, path[1:]))

    lines = ["digraph G {", "rankdir=LR;", "bgcolor=transparent;", "nodesep=0.28;", "ranksep=0.75;"]

    
    # The nodes are rendered as boxes with the label and the color of the node. The color is determined by the node_id.
    # The label is the text of the node.
    for node_id, label in NODES:
        if node_id in seen:
            fill = PALETTE.get(node_id, "#1f6feb")
            attrs = f'fillcolor="{fill}", color="{fill}", fontcolor="white", penwidth=2'
        else:
            attrs = 'fillcolor="#00000000", color="#9aa0a6", fontcolor="#9aa0a6", penwidth=1'
        lines.append(
            f'"{node_id}" [label="{label.replace(chr(10), chr(92) + "n")}", '
            f'shape=box, style="rounded,filled", '
            f"{attrs}, fontname=Helvetica, fontsize=9];"
        )
    # The columns are rendered as a group of nodes that are all on the same level.
    # The members are the nodes in the column.
    for column in COLUMNS:
        members = " ".join(f'"{node}";' for node in column)
        lines.append(f"{{ rank=same; {members} }}")

    # The edges are rendered as arrows between the nodes.
    # The source is the node the arrow starts from.
    # The target is the node the arrow ends at.
    # The label is the text of the arrow.
    # The active is a boolean that is true if the edge is active.
    # The color is the color of the arrow.
    # The decorative is a boolean that is true if the edge is decorative.
    for source, target, label in EDGES:
        active = (source, target) in taken
        color = "#1f6feb" if active else "#5f6368"
        decorative = (source, target) in DECORATIVE_EDGES
        lines.append(
            f'"{source}" -> "{target}" [label="{label}", color="{color}", '
            f"penwidth={2.2 if active else 0.8}, fontsize=8, "
            f'fontcolor="{color}", fontname=Helvetica'
            + (", constraint=false" if decorative else "")
            + "];"
        )

    lines.append("}")
    return "\n".join(lines)


def review_rounds(state: dict) -> list[dict]:
    """One entry per critic review, pairing the draft judged with the verdict.

    The state's `draft` and `critique` fields only ever hold the latest of each,
    so on a turn that was revised they describe the draft that passed. These
    rounds are what keep the rejected draft and the reason it was rejected
    visible in the log.
    """

    critiques = state.get("critique_history", [])
    drafts = state.get("draft_history", [])
    rounds = []

    # The index is the index of the critique in the list of critiques.
    # The critique is the critique object.
    # The judged is the draft object that was judged by the critique
    for index, critique in enumerate(critiques):
        judged = drafts[index] if index < len(drafts) else None
        rounds.append(
            {
                "round": index + 1,
                "verdict": critique.verdict.value,
                "grounded": critique.grounded,
                "problems": critique.problems,
                "required_fixes": critique.fixes,
                "escalation_reason": critique.escalation_reason or None,
                "rationale": critique.rationale or None,
                "draft_judged": (
                    {
                        "reply": judged.reply,
                        "citations": judged.citations,
                        "action": (
                            judged.action.action_type if judged.action else None
                        ),
                        "amount": judged.action.amount if judged.action else None,
                        "confidence": judged.confidence,
                    }
                    if judged
                    else None
                ),
            }
        )
    return rounds

def outcome_banner(state: dict) -> None:
    outcome = state.get("outcome")
    if outcome is None:
        return

    style, headline = OUTCOME_STYLE.get(outcome.value, ("info", outcome.value))
    getattr(st, style)(f"**{headline}** — {state.get('outcome_summary', '')}")

def approval_card(state: dict, on_decision) -> None:
    pending = state.get("pending_approval")
    if not pending or pending.status != "pending":
        return

    with st.container(border=True):
        st.markdown("#### Waiting on your decision")
        st.caption(
            "The workflow stopped here on purpose. Nothing has been done to the account."
        )

        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**Action requested:** `{pending.action_type}`")
            st.markdown(pending.description)
            if pending.amount is not None:
                st.markdown(f"**Amount:** ${pending.amount:,.2f}")
        with right:
            st.markdown(f"**Why a human:** {pending.reason}")
            if pending.citation:
                st.markdown(f"**Authority:** `{pending.citation}`")

        approve, reject = st.columns(2)
        if approve.button("Approve", key=f"ok-{pending.approval_id}", width="stretch"):
            on_decision(pending, "approved")
        if reject.button("Reject", key=f"no-{pending.approval_id}", width="stretch"):
            on_decision(pending, "rejected")

def _read_ticket(ticket_id: str) -> dict:
    try:
        return lookup_ticket.invoke({"ticket_id": ticket_id})
    except Exception as exc:  # ToolError, or sqlite trouble
        return {"found": False, "ticket_id": ticket_id, "error": str(exc)}

def escalation_card(state: dict) -> None:
    """
        Escalating writes a real row to the tickets table, so this reads it back
        rather than restating what the graph believes it did. A handoff nobody can
        look up is indistinguishable from no handoff.
    """

    ticket_id = state.get("escalation_id")
    if not ticket_id:
        return

    ticket = _read_ticket(ticket_id)

    with st.container(border=True):
        st.markdown("#### Handed to a specialist")

        if not ticket.get("found"):
            st.warning(
                f"The workflow recorded `{ticket_id}`, but the ticket could not be "
                "read back from the queue"
                + (f": {ticket['error']}" if ticket.get("error") else ".")
            )
            return

        st.caption(
            "Read back from the tickets table, not restated from memory. "
            "Simulated: the queue is a local SQLite table."
        )

        left, right = st.columns([2, 3])
        with left:
            st.markdown(f"**Case reference:** `{ticket['ticket_id']}`")
            st.markdown(f"**Queue:** `{ticket['queue']}`")
            st.markdown(f"**Status:** `{ticket['status']}`")
        with right:
            st.markdown(f"**Customer:** `{ticket['customer_id']}`")
            st.markdown(f"**Domain:** {DOMAIN_LABEL.get(ticket['domain'], ticket['domain'])}")
            st.markdown(f"**Opened:** {ticket['created_at']}")

        st.markdown(f"**Why it reached a person:** {ticket['summary']}")

        rejected = [r for r in review_rounds(state) if r["verdict"] != "approve"]
        inherits = [
            f"the full conversation for session `{ticket['session_id']}`",
            f"{len(state.get('sources', []))} retrieved policy chunk(s)",
            f"{len(state.get('tool_calls', []))} recorded tool call(s)",
        ]
        if rejected:
            inherits.append(
                f"{len(rejected)} draft(s) the reviewer rejected, with the reasons"
            )
        st.markdown("**The specialist picks up:** " + ", ".join(inherits) + ".")

        st.caption(
            "No money moved and no account changed. The customer was told a person "
            "would follow up, which is a promise this ticket is the record of."
        )

def specialist_queue_panel() -> None:

    st.caption(
        "Cases the workflow handed to a human, newest first. Read back through the "
        "`lookup_specialist_queue` read-only tool. The queue is a local SQLite "
        "table;"
    )

    try:
        tickets = lookup_specialist_queue.invoke({})
    except Exception as exc:
        st.warning(f"The queue could not be read: {exc}")
        return

    if not tickets:
        st.caption("The queue is empty. Escalate a case and it will appear here.")
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "case": t["ticket_id"],
                    "queue": t["queue"],
                    "customer": t["customer_id"],
                    "domain": t["domain"],
                    "status": t["status"],
                    "opened": t["created_at"],
                    "why it escalated": t.get("summary") or t.get("reason", ""),
                }
                for t in tickets
            ]
        ),
        width="stretch",
        hide_index=False,
    )


def decision_panel(state: dict) -> None:
    """Everything behind one assistant reply: route, evidence, trace, state."""

    route_tab, evidence_tab, trace_tab, state_tab = st.tabs(
        ["Route", "Evidence", "Trace", "Shared state"]
    )

    visited = [event.node for event in state.get("trace", [])]

    with route_tab:
        triage = state.get("triage")
        if triage:
            cols = st.columns(4)
            cols[0].metric("Domain", DOMAIN_LABEL.get(triage.domain.value, triage.domain.value))
            cols[1].metric("Customer", triage.customer_id or "—")
            cols[2].metric(
                "Amount", f"${triage.amount:,.2f}" if triage.amount is not None else "—"
            )
            cols[3].metric(
                "Revisions", f"{state.get('retry_count', 0)} / {MAX_REVISION_ATTEMPTS}"
            )
            if triage.reasoning:
                st.caption(f"Triage reasoning: {triage.reasoning}")

        st.graphviz_chart(workflow_dot(visited), width="stretch")

        rounds = review_rounds(state)
        if rounds:
            st.markdown(f"**Critic reviews** — {len(rounds)} this turn")
        for entry in rounds:
            verdict = entry["verdict"]
            icon = {"approve": "✓", "revise": "↻", "escalate": "↑"}.get(verdict, "•")
            headline = {
                "approve": "accepted",
                "revise": "rejected, sent back",
                "escalate": "rejected, escalated",
            }.get(verdict, verdict)
            with st.expander(
                f"{icon} Review {entry['round']}: {headline}"
                f"{' (grounded)' if entry['grounded'] else ' (not grounded)'}",
                expanded=verdict != "approve",
            ):
                if entry["rationale"]:
                    st.markdown(entry["rationale"])
                if entry["problems"]:
                    st.markdown("**Why it was rejected**")
                    for problem in entry["problems"]:
                        st.markdown(f"- {problem}")
                if entry["required_fixes"]:
                    st.markdown("**Required fixes**")
                    for fix in entry["required_fixes"]:
                        st.markdown(f"- {fix}")
                if entry["escalation_reason"]:
                    st.markdown(f"**Escalation reason:** {entry['escalation_reason']}")

                judged = entry["draft_judged"]
                if judged:
                    cited = (
                        ", ".join(f"`{c}`" for c in judged["citations"])
                        or "_none — this is what the citation guard rejects_"
                    )
                    st.markdown(
                        f"**The draft it judged** (cited: {cited})"
                        if verdict == "approve"
                        else f"**The draft it rejected** (cited: {cited})"
                    )
                    st.caption(judged["reply"])

    with evidence_tab:
        sources = state.get("sources", [])
        st.markdown(f"**Retrieved policy** — {len(sources)} chunk(s), "
                    f"filtered to `{state.get('domain', '?')}` plus shared governance")
        for source in sources:
            with st.expander(
                f"{source.doc_id} — {source.title} "
                f"({source.doc_type}, relevance {source.score})"
            ):
                st.markdown(source.content)

        draft = state.get("draft")
        if draft and draft.citations:
            st.markdown(f"**Cited by the specialist:** {', '.join(f'`{c}`' for c in draft.citations)}")

        assessment = state.get("loan_assessment")
        if assessment:
            st.markdown("**Loan eligibility assessment** (computed, not generated)")
            st.json(assessment)

        risk = state.get("fraud_risk")
        if risk:
            st.markdown("**Fraud risk score** (computed, not generated)")
            st.json(risk)

        transactions = state.get("transactions", [])
        if transactions:
            st.markdown("**Account transactions**")
            st.dataframe(pd.DataFrame(transactions), width="stretch", hide_index=True)

        calls = state.get("tool_calls", [])
        if calls:
            st.markdown("**Tool calls**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "tool": c.tool,
                            "ok": "yes" if c.ok else "NO",
                            "attempts": c.attempts,
                            "result": c.result or (c.error or ""),
                        }
                        for c in calls
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    with trace_tab:
        events = state.get("trace", [])
        if not events:
            st.caption("No trace recorded.")
        for index, event in enumerate(events, start=1):
            st.markdown(
                f"**{index}. `{event.node}`** · {event.role} · "
                f"{event.at.strftime('%H:%M:%S')}  \n{event.detail}"
            )
        errors = state.get("errors", [])
        if errors:
            st.markdown("**Handled failures**")
            for error in errors:
                st.warning(error)

    with state_tab:
        st.caption(
            "The object every role reads from and writes to. Roles never call each other."
        )
        triage = state.get("triage")
        draft = state.get("draft")
        critique = state.get("critique")
        st.json(
            {
                "session_id": state.get("session_id"),
                "domain": state.get("domain"),
                "clarifications": state.get("clarifications", 0),
                "retry_count": state.get("retry_count", 0),
                "revision_budget": MAX_REVISION_ATTEMPTS,
                "degraded": state.get("degraded", False),
                "triage": json.loads(triage.model_dump_json()) if triage else None,
                "draft": json.loads(draft.model_dump_json()) if draft else None,
                "critique": json.loads(critique.model_dump_json()) if critique else None,
                # Every round, so a rejection is still on the record after a
                # later draft is accepted.
                "review_rounds": review_rounds(state),
                "outcome": getattr(state.get("outcome"), "value", None),
                "escalation_id": state.get("escalation_id"),
            }
        )
