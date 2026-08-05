"""Side-effect tools. Every one of these is simulated.

Writes land in the local SQLite database and in `storage/action_log.json`.
Nothing here touches a real banking system, and the interface labels these
results as simulated so no one can mistake them for a completed real action.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from langchain_core.tools import tool

from src.config import ACTION_LOG
from src.tools.readonly import ToolError, connect

SIMULATED = "SIMULATED: recorded locally only, no real banking system was contacted"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(entry: dict) -> None:
    """Append-only audit log of every simulated side effect."""

    path = ACTION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []

    history.append(entry)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def read_action_log() -> list[dict]:
    if not ACTION_LOG.exists():
        return []
    try:
        return json.loads(ACTION_LOG.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _write_ticket(record: tuple) -> None:
    conn = connect()
    try:
        conn.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)", record)
        conn.commit()
    except sqlite3.Error as exc:
        raise ToolError(f"ticket write failed: {exc}") from exc
    finally:
        conn.close()


@tool
def process_refund(
    session_id: str, customer_id: str, amount: float, reason: str, citation: str
) -> dict:
    """Process a refund to the customer's account.

    Only ever called for amounts within the automated limit, after the critic
    approved and the application-owned threshold check passed. Simulated.
    """
    refund_id = f"RFD-{uuid.uuid4().hex[:8].upper()}"
    at = _now()

    _log(
        {
            "action": "process_refund",
            "refund_id": refund_id,
            "session_id": session_id,
            "customer_id": customer_id,
            "amount": amount,
            "reason": reason,
            "citation": citation,
            "at": at,
            "note": SIMULATED,
        }
    )

    return {
        "refund_id": refund_id,
        "amount": amount,
        "status": "processed",
        "at": at,
        "simulated": True,
        "note": SIMULATED,
    }


@tool
def request_human_approval(
    session_id: str,
    action_type: str,
    description: str,
    reason: str,
    citation: str,
    amount: float | None = None,
) -> dict:
    """Queue a consequential action for a human decision.

    This is the safety boundary. The action is recorded as *requested*, never as
    performed. Simulated.
    """
    approval_id = f"APR-{uuid.uuid4().hex[:8].upper()}"
    at = _now()

    _log(
        {
            "action": "request_human_approval",
            "approval_id": approval_id,
            "session_id": session_id,
            "action_type": action_type,
            "amount": amount,
            "reason": reason,
            "citation": citation,
            "status": "pending",
            "at": at,
            "note": SIMULATED,
        }
    )

    return {
        "approval_id": approval_id,
        "status": "pending",
        "action_type": action_type,
        "description": description,
        "amount": amount,
        "reason": reason,
        "citation": citation,
        "at": at,
        "simulated": True,
        "note": f"{SIMULATED}. The action has NOT been performed.",
    }


@tool
def create_escalation(
    session_id: str, customer_id: str, domain: str, reason: str, queue: str = "tier2_support"
) -> dict:
    """Hand the case to a human specialist queue. Simulated."""
    escalation_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    at = _now()

    _write_ticket(
        (escalation_id, session_id, customer_id, domain, "escalated", reason, queue, at)
    )
    _log(
        {
            "action": "create_escalation",
            "escalation_id": escalation_id,
            "session_id": session_id,
            "customer_id": customer_id,
            "domain": domain,
            "reason": reason,
            "queue": queue,
            "at": at,
            "note": SIMULATED,
        }
    )

    return {
        "escalation_id": escalation_id,
        "status": "escalated",
        "queue": queue,
        "at": at,
        "simulated": True,
        "note": SIMULATED,
    }


@tool
def record_approval_decision(
    approval_id: str, decision: str, approver: str, note: str = ""
) -> dict:
    """Record a human's decision on a queued approval.

    Called only from the interface after a person clicks approve or reject.
    Simulated.
    """
    if decision not in {"approved", "rejected"}:
        raise ToolError(f"decision must be 'approved' or 'rejected', got {decision!r}")

    at = _now()
    _log(
        {
            "action": "record_approval_decision",
            "approval_id": approval_id,
            "decision": decision,
            "approver": approver,
            "note": note,
            "source": "human",
            "at": at,
            "disclaimer": SIMULATED,
        }
    )

    return {
        "approval_id": approval_id,
        "decision": decision,
        "approver": approver,
        "at": at,
        "simulated": True,
        "note": SIMULATED,
    }


SIDE_EFFECT_TOOLS = [
    process_refund,
    request_human_approval,
    create_escalation,
    record_approval_decision,
]
