"""Tests for SessionRegistry wired into the voice_rtc adapter (Task 6.3).

The adapter constructs a per-user session on first use, feeds ASR
finals into ``submit_user_turn``, and routes the resulting token stream
into ``on_assistant_token_stream`` (which the M5.3 audio-out pipeline
already consumes). Disconnects release the session — but reconnects
within the idle window re-attach to the same session object.

We mock the ``V2VAgentSession`` factory at the adapter boundary so no
network call is made.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.voice_rtc import VoiceRTCAdapter


def _adapter_no_env(monkeypatch) -> VoiceRTCAdapter:
    for var in (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "SARVAM_API_KEY",
        "V2V_DATA_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    return VoiceRTCAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "wss://lk.example", "api_key": "ak", "api_secret": "sk"},
        )
    )


class _FakeSession:
    """Stand-in for V2VAgentSession with a deterministic streaming output."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.turns: List[str] = []
        self.closed = False
        # Each ``submit_user_turn`` call produces this token sequence by
        # default; tests may swap it.
        self.tokens: List[str] = ["Hi.", " There."]

    async def submit_user_turn(self, text: str) -> AsyncIterator[str]:
        self.turns.append(text)
        tokens = list(self.tokens)

        async def _gen():
            for t in tokens:
                yield t

        return _gen()

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_factory_state(monkeypatch):
    """Replace the adapter's session factory with one returning _FakeSession."""

    created: Dict[str, _FakeSession] = {}
    factory_calls: List[str] = []

    async def factory(user_id: str) -> _FakeSession:
        factory_calls.append(user_id)
        sess = _FakeSession(user_id)
        created[user_id] = sess
        return sess

    return created, factory_calls, factory


def test_deliver_final_calls_session_for_user(monkeypatch, fake_factory_state):
    created, factory_calls, factory = fake_factory_state
    adapter = _adapter_no_env(monkeypatch)
    # Inject our fake factory.
    adapter._sessions._factory = factory

    captured_streams: List[Tuple[str, List[str]]] = []

    async def _capture(chat_id, token_iterator):
        tokens: List[str] = []
        async for t in token_iterator:
            tokens.append(t)
        captured_streams.append((chat_id, tokens))

    adapter.on_assistant_token_stream = _capture  # type: ignore[method-assign]

    # Pre-register a fake active call (skip _on_room since we already test
    # that elsewhere).
    adapter._active_calls["v2v-userA-call1"] = {
        "user_id": "userA",
        "call_id": "call1",
    }

    async def _go():
        # Acquire the session via the same path _on_room would use.
        sess = await adapter._sessions.get_or_create("userA")
        adapter._active_calls["v2v-userA-call1"]["session"] = sess
        await adapter._deliver_final("hello agent", "userA", "v2v-userA-call1")

    asyncio.run(_go())

    assert factory_calls == ["userA"]
    sess = created["userA"]
    assert sess.turns == ["hello agent"]
    assert captured_streams == [("v2v-userA-call1", ["Hi.", " There."])]


def test_two_finals_same_user_reuse_session(monkeypatch, fake_factory_state):
    created, factory_calls, factory = fake_factory_state
    adapter = _adapter_no_env(monkeypatch)
    adapter._sessions._factory = factory

    async def _swallow(chat_id, token_iterator):
        async for _ in token_iterator:
            pass

    adapter.on_assistant_token_stream = _swallow  # type: ignore[method-assign]

    adapter._active_calls["v2v-userA-call1"] = {
        "user_id": "userA",
        "call_id": "call1",
    }

    async def _go():
        sess = await adapter._sessions.get_or_create("userA")
        adapter._active_calls["v2v-userA-call1"]["session"] = sess
        await adapter._deliver_final("first", "userA", "v2v-userA-call1")
        await adapter._deliver_final("second", "userA", "v2v-userA-call1")

    asyncio.run(_go())

    # One factory call, two turns recorded on the same session.
    assert factory_calls == ["userA"]
    assert created["userA"].turns == ["first", "second"]


def test_two_users_get_distinct_sessions(monkeypatch, fake_factory_state):
    created, factory_calls, factory = fake_factory_state
    adapter = _adapter_no_env(monkeypatch)
    adapter._sessions._factory = factory

    async def _swallow(chat_id, token_iterator):
        async for _ in token_iterator:
            pass

    adapter.on_assistant_token_stream = _swallow  # type: ignore[method-assign]

    adapter._active_calls["v2v-userA-call1"] = {"user_id": "userA", "call_id": "call1"}
    adapter._active_calls["v2v-userB-call1"] = {"user_id": "userB", "call_id": "call1"}

    async def _go():
        sa = await adapter._sessions.get_or_create("userA")
        sb = await adapter._sessions.get_or_create("userB")
        adapter._active_calls["v2v-userA-call1"]["session"] = sa
        adapter._active_calls["v2v-userB-call1"]["session"] = sb
        await adapter._deliver_final("hi from A", "userA", "v2v-userA-call1")
        await adapter._deliver_final("hi from B", "userB", "v2v-userB-call1")

    asyncio.run(_go())

    assert sorted(factory_calls) == ["userA", "userB"]
    assert created["userA"].turns == ["hi from A"]
    assert created["userB"].turns == ["hi from B"]
    assert created["userA"] is not created["userB"]


def test_on_room_acquires_session_and_disconnect_releases(monkeypatch, fake_factory_state):
    """End-to-end: ``_on_room`` registers the session on the call state
    and ``disconnect`` releases it (so a future ``get_or_create`` for the
    same user_id, after eviction, would mint a fresh one)."""
    created, factory_calls, factory = fake_factory_state
    adapter = _adapter_no_env(monkeypatch)
    adapter._sessions._factory = factory

    # Stub SarvamASRStream and the publish-audio seam.
    class _FakeASR:
        def __init__(self, *a, **k):
            self.closed = False

        async def push_pcm(self, frame):
            pass

        async def events(self):
            # Block until close() puts a sentinel.
            self._q = asyncio.Queue()
            while True:
                ev = await self._q.get()
                if ev is None:
                    return
                yield ev

        async def close(self):
            self.closed = True
            try:
                await self._q.put(None)  # type: ignore[attr-defined]
            except Exception:
                pass

    monkeypatch.setattr("tools.sarvam_asr.SarvamASRStream", _FakeASR)

    async def _no_publish(self_, ctx, state):
        pass

    monkeypatch.setattr(
        "gateway.platforms.voice_rtc.VoiceRTCAdapter._open_publish_audio",
        _no_publish,
    )

    async def _no_frames(self_, ctx):
        if False:  # pragma: no cover
            yield b""
        return

    monkeypatch.setattr(
        "gateway.platforms.voice_rtc.VoiceRTCAdapter._iter_audio_frames",
        _no_frames,
    )

    ctx = MagicMock()
    ctx.room = MagicMock()
    ctx.room.name = "v2v-userA-callX"

    async def _go():
        await adapter._on_room(ctx, "userA", "callX")
        state = adapter._active_calls["v2v-userA-callX"]
        assert "session" in state
        assert state["session"] is created["userA"]
        # The registry should hold an in-use ref while the call is open.
        assert "userA" in adapter._sessions
        await adapter.disconnect()
        # After disconnect the session is released — still cached during
        # the idle window but no longer in use.
        # (We don't assert it was evicted because the TTL hasn't elapsed.)

    asyncio.run(_go())

    assert factory_calls == ["userA"]


def test_session_factory_default_uses_v2v_agent_session(monkeypatch):
    """The adapter's default session factory wraps ``V2VAgentSession``
    using the configured Sarvam API key and the v2v memory loader."""
    monkeypatch.delenv("V2V_DATA_ROOT", raising=False)
    monkeypatch.setenv("SARVAM_API_KEY", "sk-test")

    adapter = _adapter_no_env(monkeypatch)
    # Override after construction to keep monkeypatch order simple.
    adapter._sarvam_api_key = "sk-test"

    seen: Dict[str, Any] = {}

    class _FakeV2VAgentSession:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.user_id = kwargs.get("user_id")

        async def submit_user_turn(self, text):
            async def _g():
                yield ""

            return _g()

        async def close(self):
            pass

    monkeypatch.setattr(
        "gateway.platforms.voice_rtc.V2VAgentSession",
        _FakeV2VAgentSession,
    )

    async def _go():
        sess = await adapter._sessions.get_or_create("userZ")
        return sess

    sess = asyncio.run(_go())
    assert sess.user_id == "userZ"
    assert seen.get("user_id") == "userZ"
    assert seen.get("api_key") == "sk-test"
    assert seen.get("model") == "sarvam-m"
    # System prompt should have been built (string, possibly empty when
    # the data root has no files).
    assert "system_prompt" in seen
    assert isinstance(seen["system_prompt"], str)


# Note: the old OpenAI-direct V2VAgentSession tests were removed when the
# session was rewired to wrap Hermes' AIAgent. Equivalent coverage now
# lives in tests/gateway/test_voice_rtc_aiagent_integration.py.
