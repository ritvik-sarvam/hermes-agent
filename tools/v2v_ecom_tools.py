"""V2V e-commerce domain tools (Milestone 8).

A tiny fixture-backed customer-support backend the v2v voice agent can
call during a live call. All five tools are pure-Python file readers
over ``tools/fixtures/orders.json`` (with a couple of mutating writers
for return / address changes); no network, no external services.

Why this exists
---------------
The hackathon harness wants the LLM to be able to *do* something
useful in a customer-support call — look up an order, check refund
status, kick off a return — without depending on a real backend. The
fixture file simulates that state so demos are reproducible.

Functions
---------
- :func:`lookup_order`          — exact + fuzzy trailing-digit search
- :func:`refund_status`         — refund state + ETA
- :func:`initiate_return`       — generates an RMA, persists to disk
- :func:`update_address`        — pre-shipping address change
- :func:`escalation_handoff`    — writes a ticket for the human queue

Each function returns a plain Python dict the LLM can summarise out
loud — never JSON-encoded strings or huge nested blobs.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from filelock import FileLock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixture paths — overridable by tests via ``monkeypatch.setattr``.
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ORDERS_PATH: Path = _FIXTURES_DIR / "orders.json"
ESCALATIONS_PATH: Path = _FIXTURES_DIR / "escalations.json"


# Refund SLA windows (business days) keyed by payment method. These
# numbers are illustrative — they exist so the agent can quote a realistic
# ETA out loud rather than saying "it depends".
_REFUND_SLA_DAYS = {
    "upi": 3,
    "card": 7,
    "netbanking": 5,
    "cod": 10,
}

_REFUND_REMAINING_DAYS = {
    "initiated": None,    # uses full SLA
    "in_transit": 4,
    "received": 2,
    "completed": 0,
}

# Statuses that indicate the parcel has left the warehouse — past the
# free-address-change cutoff for the demo.
_SHIPPED_STATUSES = {
    "in_transit",
    "delivered",
    "returned",
    "lost",
    "refund_initiated",
    "refund_completed",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lockfile(path: Path) -> FileLock:
    """Return a FileLock guarding writes to *path*. Lock file lives next
    to the data file with a ``.lock`` suffix.
    """
    return FileLock(str(path) + ".lock")


def _read_orders() -> Dict[str, Any]:
    """Load the orders fixture. Returns ``{"orders": [...]}``.

    Missing file is treated as empty rather than a hard failure so
    misconfigured demos don't crash the call.
    """
    try:
        return json.loads(ORDERS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"orders": []}


def _write_orders(data: Dict[str, Any]) -> None:
    """Write the orders fixture atomically under a file lock."""
    with _lockfile(ORDERS_PATH):
        ORDERS_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _normalise(s: str) -> str:
    """Strip non-alphanumerics and lowercase. Used for fuzzy matching."""
    return re.sub(r"[^A-Za-z0-9]", "", s or "").lower()


def _find_order(orders: List[Dict[str, Any]], query: str) -> Dict[str, Any] | None:
    """Return the order whose ``order_id`` matches *query* exactly
    (case-insensitive, ignoring punctuation), or None.
    """
    nq = _normalise(query)
    for o in orders:
        if _normalise(o.get("order_id", "")) == nq:
            return o
    return None


def _summary(order: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small subset of an order suitable for a candidate list —
    enough for the agent to disambiguate verbally without reading the
    whole record aloud.
    """
    return {
        "order_id": order.get("order_id"),
        "status": order.get("status"),
        "placed_at": order.get("placed_at"),
        "total_inr": order.get("total_inr"),
    }


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID.

    Supports partial / fuzzy match on the trailing digits — e.g.
    ``order_id="3456"`` matches ``"ACME-3456"``. Returns the full order
    record on an exact (or single-fuzzy) hit. If multiple orders match
    the trailing-digits pattern, returns up to 3 candidates as
    ``{"candidates": [...]}`` so the agent can ask the user to clarify.
    Returns ``{"error": "not_found"}`` on no match.
    """
    data = _read_orders()
    orders = data.get("orders", [])

    # 1. Exact match (after stripping punctuation/case).
    hit = _find_order(orders, order_id)
    if hit:
        return dict(hit)

    # 2. Fuzzy: orders whose normalised order_id ends with the
    #    normalised query string.
    nq = _normalise(order_id)
    if not nq:
        return {"error": "not_found"}

    candidates = [
        o for o in orders if _normalise(o.get("order_id", "")).endswith(nq)
    ]

    if len(candidates) == 1:
        return dict(candidates[0])
    if len(candidates) > 1:
        return {"candidates": [_summary(o) for o in candidates[:3]]}
    return {"error": "not_found"}


def refund_status(order_id: str) -> dict:
    """Get refund status and ETA for an order.

    Returns ``{"status", "eta_days", "initiated_at", "payment_method_sla_days"}``
    for orders with an active or completed refund record. Returns
    ``{"error": "no_refund"}`` if the order has no refund attached, and
    ``{"error": "not_found"}`` if the order doesn't exist.
    """
    data = _read_orders()
    order = _find_order(data.get("orders", []), order_id)
    if not order:
        return {"error": "not_found"}

    refund = order.get("refund")
    if not refund:
        return {"error": "no_refund"}

    pm = (order.get("payment_method") or "").lower()
    sla = _REFUND_SLA_DAYS.get(pm, 7)
    status = refund.get("status", "initiated")
    remaining = _REFUND_REMAINING_DAYS.get(status, sla)
    if remaining is None:
        remaining = sla

    return {
        "status": status,
        "eta_days": int(remaining),
        "initiated_at": refund.get("initiated_at"),
        "completed_at": refund.get("completed_at"),
        "amount_inr": refund.get("amount_inr"),
        "payment_method_sla_days": sla,
    }


def initiate_return(order_id: str, reason: str) -> dict:
    """Mark an order as return-initiated and generate an RMA number.

    If the order already has a ``return`` record, the call is idempotent
    and returns the existing RMA — no duplicate is created. Returns
    ``{"rma", "next_step"}`` on success, or ``{"error": "not_found"}``
    if the order doesn't exist.
    """
    with _lockfile(ORDERS_PATH):
        data = _read_orders()
        order = _find_order(data.get("orders", []), order_id)
        if not order:
            return {"error": "not_found"}

        existing = order.get("return")
        if existing and existing.get("rma"):
            return {
                "rma": existing["rma"],
                "next_step": (
                    "A pickup is already scheduled. The courier will collect "
                    "the package within 2 business days."
                ),
                "already_initiated": True,
            }

        # Generate an RMA tied to the order_id for human-readability.
        suffix = uuid.uuid4().hex[:6].upper()
        rma = f"RMA-{order.get('order_id', 'UNKNOWN')}-{suffix}"
        order["return"] = {
            "rma": rma,
            "reason": reason,
            "initiated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Don't overwrite a more-specific terminal status (returned, lost).
        if order.get("status") in {"delivered", "in_transit"}:
            order["status"] = "return_initiated"

        ORDERS_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "rma": rma,
        "next_step": (
            "Please keep the item in its original packaging. A courier will "
            "pick it up within 2 business days."
        ),
    }


def update_address(order_id: str, new_address: str) -> dict:
    """Update the delivery address on an in-transit order.

    Refuses if the order has already shipped past the change cutoff —
    i.e. its status is one of {in_transit, delivered, returned, lost,
    refund_initiated, refund_completed} *and* the ``shipped`` flag is
    set on the record. Returns ``{"ok": True, "order_id", "new_address"}``
    on success, ``{"error": "past_cutoff"}`` if the cutoff has passed,
    or ``{"error": "not_found"}`` if no such order.
    """
    with _lockfile(ORDERS_PATH):
        data = _read_orders()
        order = _find_order(data.get("orders", []), order_id)
        if not order:
            return {"error": "not_found"}

        # Past-cutoff: actually shipped. ``in_transit`` orders that have
        # not yet been handed to the courier (``shipped`` is False) are
        # still mutable — this lets the demo include a realistic
        # in-flight order that can still be redirected.
        if order.get("shipped") and order.get("status") in _SHIPPED_STATUSES:
            return {
                "error": "past_cutoff",
                "order_id": order.get("order_id"),
                "status": order.get("status"),
            }

        order["delivery_address"] = {
            "line1": new_address,
            "raw": new_address,
        }
        order["address_updated_at"] = datetime.now(timezone.utc).isoformat()

        ORDERS_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "ok": True,
        "order_id": order.get("order_id"),
        "new_address": new_address,
    }


def escalation_handoff(reason: str, summary: str) -> dict:
    """Capture the escalation context for the human agent.

    Writes a ticket to ``tools/fixtures/escalations.json`` (creating the
    file on first use) and returns ``{"ticket_id", "queue"}`` so the
    agent can quote them to the caller.
    """
    queue = _route_queue(reason)
    ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    ticket = {
        "ticket_id": ticket_id,
        "reason": reason,
        "summary": summary,
        "queue": queue,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with _lockfile(ESCALATIONS_PATH):
        if ESCALATIONS_PATH.exists():
            try:
                data = json.loads(ESCALATIONS_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {"tickets": []}
        else:
            data = {"tickets": []}

        data.setdefault("tickets", []).append(ticket)
        ESCALATIONS_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {"ticket_id": ticket_id, "queue": queue}


def _route_queue(reason: str) -> str:
    """Pick a human queue based on the escalation reason. Cheap keyword
    routing is enough for the demo.
    """
    r = (reason or "").lower()
    if any(k in r for k in ("refund", "money", "billing", "payment")):
        return "billing"
    if any(k in r for k in ("lost", "stolen", "fraud")):
        return "trust_and_safety"
    if any(k in r for k in ("angry", "manager", "complaint")):
        return "tier2_support"
    return "general_support"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

LOOKUP_ORDER_SCHEMA = {
    "name": "lookup_order",
    "description": (
        "Look up a customer order by its ID. Supports fuzzy matching on the "
        "trailing digits — e.g. order_id='3456' matches 'ACME-3456'. Returns "
        "the full order details on a unique hit, a candidate list if multiple "
        "orders match, or {'error': 'not_found'} if nothing matches."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "Order identifier or trailing-digit suffix (e.g. 'ACME-3456' or '3456').",
            },
        },
        "required": ["order_id"],
    },
}

REFUND_STATUS_SCHEMA = {
    "name": "refund_status",
    "description": (
        "Get the refund status and estimated days remaining for an order. "
        "Returns status (initiated|in_transit|received|completed), eta_days, "
        "initiated_at, and payment_method_sla_days. Returns {'error': 'no_refund'} "
        "if the order has no refund record."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID whose refund status to fetch.",
            },
        },
        "required": ["order_id"],
    },
}

INITIATE_RETURN_SCHEMA = {
    "name": "initiate_return",
    "description": (
        "Mark an order as return-initiated and generate an RMA number. "
        "Idempotent: a second call on the same order returns the existing "
        "RMA rather than creating a duplicate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID to return.",
            },
            "reason": {
                "type": "string",
                "description": "Short free-text reason from the customer (e.g. 'wrong size', 'defective').",
            },
        },
        "required": ["order_id", "reason"],
    },
}

UPDATE_ADDRESS_SCHEMA = {
    "name": "update_address",
    "description": (
        "Update the delivery address on an order. Refuses with "
        "{'error': 'past_cutoff'} if the order has already shipped."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID to update.",
            },
            "new_address": {
                "type": "string",
                "description": "The new delivery address as a single free-text string.",
            },
        },
        "required": ["order_id", "new_address"],
    },
}

ESCALATION_HANDOFF_SCHEMA = {
    "name": "escalation_handoff",
    "description": (
        "Capture the escalation context for a human agent and return the "
        "ticket id + queue assignment. Use when the caller asks for a manager, "
        "the issue is out of scope, or repeated tool calls fail to resolve it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Short tag describing why we're escalating (e.g. 'angry_customer', 'refund_dispute').",
            },
            "summary": {
                "type": "string",
                "description": "1-3 sentence summary of the call so far so the human agent has context.",
            },
        },
        "required": ["reason", "summary"],
    },
}


def _check_v2v_ecom_requirements() -> bool:
    """The fixtures ship with the repo, so the toolset is always
    available. Kept as a hook for future gating (e.g. demo-only flag).
    """
    return True


# Register at module top level — the discover_builtin_tools AST scanner only
# detects ``registry.register(...)`` statements at the module body, not when
# they're nested inside a ``try:`` (or any other compound statement). Wrapping
# the calls in a try/except previously meant the module was never auto-imported
# at startup, so the v2v toolset showed as "available" with zero tools.
from tools.registry import registry

registry.register(
    name="lookup_order",
    toolset="v2v",
    schema=LOOKUP_ORDER_SCHEMA,
    handler=lambda args, **kw: lookup_order(args.get("order_id", "")),
    check_fn=_check_v2v_ecom_requirements,
    emoji="📦",
)
registry.register(
    name="refund_status",
    toolset="v2v",
    schema=REFUND_STATUS_SCHEMA,
    handler=lambda args, **kw: refund_status(args.get("order_id", "")),
    check_fn=_check_v2v_ecom_requirements,
    emoji="💸",
)
registry.register(
    name="initiate_return",
    toolset="v2v",
    schema=INITIATE_RETURN_SCHEMA,
    handler=lambda args, **kw: initiate_return(
        args.get("order_id", ""), args.get("reason", "")
    ),
    check_fn=_check_v2v_ecom_requirements,
    emoji="↩️",
)
registry.register(
    name="update_address",
    toolset="v2v",
    schema=UPDATE_ADDRESS_SCHEMA,
    handler=lambda args, **kw: update_address(
        args.get("order_id", ""), args.get("new_address", "")
    ),
    check_fn=_check_v2v_ecom_requirements,
    emoji="🏠",
)
registry.register(
    name="escalation_handoff",
    toolset="v2v",
    schema=ESCALATION_HANDOFF_SCHEMA,
    handler=lambda args, **kw: escalation_handoff(
        args.get("reason", ""), args.get("summary", "")
    ),
    check_fn=_check_v2v_ecom_requirements,
    emoji="🚨",
)
