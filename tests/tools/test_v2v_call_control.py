"""Unit tests for the v2v call-control tool handlers."""

from __future__ import annotations

import importlib

from tools import v2v_call_control
from tools.registry import registry


def test_end_call_returns_structured_result():
    out = v2v_call_control.end_call("issue resolved")
    assert out["status"] == "ok"
    assert out["action"] == "end_call"
    assert out["reason"] == "issue resolved"


def test_agent_handover_returns_structured_result():
    out = v2v_call_control.agent_handover("angry_customer", "User wants a refund.")
    assert out["status"] == "ok"
    assert out["action"] == "handover"
    assert out["reason"] == "angry_customer"
    assert out["summary"] == "User wants a refund."


def test_end_call_and_handover_registered_in_v2v_toolset():
    importlib.reload(v2v_call_control)
    names = registry.get_tool_names_for_toolset("v2v")
    assert "end_call" in names
    assert "agent_handover" in names
