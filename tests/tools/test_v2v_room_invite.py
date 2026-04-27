"""Tests for the v2v_room_invite tool (Milestone 9).

The tool mints a LiveKit access token for an outbound call and either
emails the join URL to the user or — when SMTP creds are missing —
falls back to writing it into a per-user log file. We never touch a
real LiveKit server or SMTP relay in these tests; everything is mocked
or pure-Python.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_jwt_segment(segment: str) -> dict:
    """Decode a base64url JWT segment without verifying the signature."""
    pad = "=" * (-len(segment) % 4)
    raw = base64.urlsafe_b64decode(segment + pad)
    return json.loads(raw.decode("utf-8"))


def _peek_jwt(token: str) -> tuple[dict, dict]:
    parts = token.split(".")
    assert len(parts) == 3, f"not a JWT: {token!r}"
    return _decode_jwt_segment(parts[0]), _decode_jwt_segment(parts[1])


@pytest.fixture
def env_livekit(monkeypatch):
    """Stub the LiveKit env vars so the tool can mint a JWT."""
    monkeypatch.setenv("LIVEKIT_URL", "wss://test.livekit.example")
    monkeypatch.setenv("LIVEKIT_API_KEY", "API_test_key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test_secret_with_enough_length_for_hmac")
    return None


@pytest.fixture
def env_no_smtp(monkeypatch):
    """Make sure SMTP_* are unset so the tool falls back to log writing."""
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "SMTP_FROM"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("V2V_DATA_ROOT", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_creates_room_with_correct_name_pattern(env_livekit, env_no_smtp, data_root):
    from tools import v2v_room_invite as m

    out = asyncio.run(m.v2v_room_invite(user_id="abc123", reason="say hi"))
    room = out["room"]
    assert re.match(r"^v2v-abc123-[0-9a-f]{8}$", room), room


def test_returns_token_and_url(env_livekit, env_no_smtp, data_root):
    from tools import v2v_room_invite as m

    out = asyncio.run(m.v2v_room_invite(user_id="alice", reason="ping"))
    assert isinstance(out["token"], str) and out["token"]
    assert isinstance(out["url"], str) and out["url"]
    header, _ = _peek_jwt(out["token"])
    # Must be a JWT — typ "JWT" and a recognised alg.
    assert header.get("alg")
    assert header.get("typ") in (None, "JWT")  # PyJWT often omits typ


def test_token_grants_room_join(env_livekit, env_no_smtp, data_root):
    from tools import v2v_room_invite as m

    out = asyncio.run(m.v2v_room_invite(user_id="bob", reason="check refund"))
    _, payload = _peek_jwt(out["token"])
    video = payload.get("video", {})
    assert video.get("room") == out["room"]
    assert video.get("roomJoin") is True
    assert video.get("canPublish") is True
    assert video.get("canSubscribe") is True


def test_falls_back_to_log_when_smtp_missing(env_livekit, env_no_smtp, data_root):
    from tools import v2v_room_invite as m

    out = asyncio.run(
        m.v2v_room_invite(user_id="carol", reason="post-purchase", notify_email="x@y.test")
    )
    assert out["notified"] is True
    log = data_root / "users" / "carol" / "outbound_log.md"
    assert log.exists(), "outbound_log.md should be written"
    text = log.read_text(encoding="utf-8")
    # URL line should mention the join URL
    assert out["url"] in text or out["token"] in text
    # Should include an ISO-ish timestamp
    assert re.search(r"\d{4}-\d{2}-\d{2}T", text)


def test_smtp_path_called_when_creds_present(env_livekit, data_root, monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.test.example")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "bot@test.example")
    monkeypatch.setenv("SMTP_PASS", "pw")
    monkeypatch.setenv("SMTP_FROM", "bot@test.example")

    captured: dict = {}

    async def fake_send(*, to, subject, body, **kw):
        captured["to"] = to
        captured["subject"] = subject
        captured["body"] = body
        return True

    from tools import v2v_room_invite as m

    monkeypatch.setattr(m, "_send_email", fake_send)

    out = asyncio.run(
        m.v2v_room_invite(user_id="dave", reason="follow-up", notify_email="dave@example.com")
    )
    assert out["notified"] is True
    assert captured["to"] == "dave@example.com"
    assert out["url"] in captured["body"]
    # Log fallback should not have been used
    log = data_root / "users" / "dave" / "outbound_log.md"
    assert not log.exists()


def test_no_email_no_smtp_returns_notified_false(env_livekit, env_no_smtp, data_root):
    from tools import v2v_room_invite as m

    out = asyncio.run(m.v2v_room_invite(user_id="erin", reason="silent"))
    assert out["notified"] is False
    log = data_root / "users" / "erin" / "outbound_log.md"
    assert not log.exists()


def test_register_under_v2v_toolset(env_livekit):
    # Importing the module registers the tool.
    from tools import v2v_room_invite  # noqa: F401
    from tools.registry import registry

    names = registry.get_tool_names_for_toolset("v2v")
    assert "v2v_room_invite" in names
