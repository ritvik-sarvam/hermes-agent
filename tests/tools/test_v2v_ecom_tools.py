"""Tests for the v2v fixture-backed e-commerce domain tools.

These tools simulate a tiny customer-support backend by reading (and
occasionally mutating) ``tools/fixtures/orders.json``. Tests run against
a private copy of the fixture in ``tmp_path`` so they never pollute the
canonical file on disk.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


CANONICAL_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "fixtures"
    / "orders.json"
)


@pytest.fixture
def ecom(tmp_path, monkeypatch):
    """Copy the canonical orders.json into ``tmp_path`` and patch the
    module-level fixture paths to point at the temp copy. Returns the
    tools module so tests can call functions directly.
    """
    # Late import — the tests fail with ImportError until the module
    # exists, which is the failing-test-first state for TDD.
    from tools import v2v_ecom_tools

    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    orders_copy = fixtures_dir / "orders.json"
    shutil.copy(CANONICAL_FIXTURE, orders_copy)

    monkeypatch.setattr(v2v_ecom_tools, "ORDERS_PATH", orders_copy)
    monkeypatch.setattr(
        v2v_ecom_tools,
        "ESCALATIONS_PATH",
        fixtures_dir / "escalations.json",
    )
    return v2v_ecom_tools


# ---------------------------------------------------------------------------
# lookup_order
# ---------------------------------------------------------------------------


def test_lookup_order_exact_match(ecom):
    result = ecom.lookup_order("ACME-3456")
    assert result.get("order_id") == "ACME-3456"
    assert result.get("status") == "in_transit"
    assert "items" in result
    # Must not return an error or candidates wrapper for an exact match
    assert "error" not in result
    assert "candidates" not in result


def test_lookup_order_fuzzy_trailing_digits(ecom):
    """Trailing-digit search: '3456' must resolve to ACME-3456."""
    result = ecom.lookup_order("3456")
    assert result.get("order_id") == "ACME-3456"
    assert "error" not in result


def test_lookup_order_multiple_candidates(ecom):
    """'99' matches both ACME-4499 and ACME-2299 → return candidates list."""
    result = ecom.lookup_order("99")
    assert "candidates" in result
    assert isinstance(result["candidates"], list)
    cand_ids = {c.get("order_id") if isinstance(c, dict) else c for c in result["candidates"]}
    assert "ACME-4499" in cand_ids
    assert "ACME-2299" in cand_ids
    # Spec: up to 3 candidates
    assert len(result["candidates"]) <= 3


def test_lookup_order_not_found(ecom):
    result = ecom.lookup_order("ZZZZZZ")
    assert result.get("error") == "not_found"


# ---------------------------------------------------------------------------
# refund_status
# ---------------------------------------------------------------------------


def test_refund_status_for_in_progress_refund(ecom):
    """ACME-9012 has refund.status = 'initiated' in the fixture."""
    result = ecom.refund_status("ACME-9012")
    assert "error" not in result
    assert result.get("status") == "initiated"
    assert "eta_days" in result
    assert isinstance(result["eta_days"], int)
    assert result.get("initiated_at")
    assert "payment_method_sla_days" in result


def test_refund_status_for_order_with_no_refund(ecom):
    """ACME-7821 is a plain delivered order with no refund record."""
    result = ecom.refund_status("ACME-7821")
    assert result.get("error") == "no_refund"


# ---------------------------------------------------------------------------
# initiate_return
# ---------------------------------------------------------------------------


def test_initiate_return_creates_rma_and_persists(ecom):
    """First call generates an RMA and the JSON fixture reflects it."""
    result = ecom.initiate_return("ACME-7821", reason="Changed mind")
    assert "error" not in result
    rma = result.get("rma")
    assert rma and rma.startswith("RMA-")
    assert "next_step" in result

    # Read back from disk — the mutation must persist.
    data = json.loads(ecom.ORDERS_PATH.read_text())
    order = next(o for o in data["orders"] if o["order_id"] == "ACME-7821")
    assert order.get("return", {}).get("rma") == rma
    assert order["return"].get("reason") == "Changed mind"


def test_initiate_return_idempotent(ecom):
    """Calling twice on the same order returns the same RMA, not a duplicate."""
    first = ecom.initiate_return("ACME-7821", reason="Changed mind")
    second = ecom.initiate_return("ACME-7821", reason="Changed mind again")
    assert first.get("rma") == second.get("rma")

    data = json.loads(ecom.ORDERS_PATH.read_text())
    order = next(o for o in data["orders"] if o["order_id"] == "ACME-7821")
    # Still a single return record — not a list / duplicate.
    assert isinstance(order["return"], dict)


# ---------------------------------------------------------------------------
# update_address
# ---------------------------------------------------------------------------


def test_update_address_before_cutoff(ecom):
    """ACME-3456 is in_transit but not yet shipped → address can be updated."""
    new_addr = "99 New Street, Bengaluru 560002"
    result = ecom.update_address("ACME-3456", new_addr)
    assert result.get("ok") is True

    data = json.loads(ecom.ORDERS_PATH.read_text())
    order = next(o for o in data["orders"] if o["order_id"] == "ACME-3456")
    # Address was updated — the new value must be visible somewhere.
    addr_blob = json.dumps(order["delivery_address"])
    assert "99 New Street" in addr_blob


def test_update_address_past_cutoff_refuses(ecom):
    """ACME-7821 was already delivered → past the change cutoff."""
    result = ecom.update_address("ACME-7821", "Anywhere else")
    assert result.get("error") == "past_cutoff"


# ---------------------------------------------------------------------------
# escalation_handoff
# ---------------------------------------------------------------------------


def test_escalation_handoff_writes_ticket(ecom):
    result = ecom.escalation_handoff(
        reason="angry_customer",
        summary="User wants to speak to a manager about lost ACME-4499.",
    )
    assert "ticket_id" in result
    assert "queue" in result

    data = json.loads(ecom.ESCALATIONS_PATH.read_text())
    tickets = data.get("tickets", [])
    assert len(tickets) == 1
    t = tickets[0]
    assert t.get("ticket_id") == result["ticket_id"]
    assert t.get("reason") == "angry_customer"
    assert "lost ACME-4499" in t.get("summary", "")
