"""Integration tests for V2VAgentSession wired to Hermes' ``AIAgent``.

The session is responsible for bridging AIAgent's synchronous run loop
(executed on a worker thread) into an async iterator of content tokens
the audio-out pipeline consumes. We monkeypatch ``run_agent.AIAgent``
so no real model call goes out.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.voice_rtc import (
    V2VAgentSession,
    VoiceRTCAdapter,
    _DEFAULT_V2V_TOOLSETS,
    _VOICE_MODE_DIRECTIVE,
)


class _FakeAIAgent:
    """Records constructor kwargs; emits deterministic deltas + tool
    callbacks when ``run_conversation`` is called."""

    instances: List["_FakeAIAgent"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.stream_delta_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self.run_calls: List[Dict[str, Any]] = []
        _FakeAIAgent.instances.append(self)

    def run_conversation(
        self,
        *,
        user_message: str,
        conversation_history: List[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        self.run_calls.append(
            {
                "user_message": user_message,
                "conversation_history": list(conversation_history or []),
            }
        )
        # Stream two deltas synchronously.
        if self.stream_delta_callback is not None:
            self.stream_delta_callback("Hello ")
            time.sleep(0.001)
            self.stream_delta_callback("world.")
        # Fire one tool start + complete (using AIAgent's real positional
        # signature: tc_id, name, args[, result]).
        if self.tool_start_callback is not None:
            self.tool_start_callback("call-1", "lookup_order", {"id": "ACME-3456"})
        if self.tool_complete_callback is not None:
            self.tool_complete_callback(
                "call-1", "lookup_order", {"id": "ACME-3456"}, {"status": "in transit"}
            )
        return {"final_response": "Hello world.", "messages": []}


@pytest.fixture(autouse=True)
def _reset_fake_aiagent_instances():
    _FakeAIAgent.instances.clear()
    yield
    _FakeAIAgent.instances.clear()


def _patch_aiagent(monkeypatch) -> None:
    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", _FakeAIAgent)


def test_max_tokens_passed_to_aiagent(monkeypatch):
    _patch_aiagent(monkeypatch)

    V2VAgentSession(
        user_id="u1",
        api_key="sk",
        model="sarvam-m",
        system_prompt="SYS",
        max_tokens=2048,
    )

    assert len(_FakeAIAgent.instances) == 1
    assert _FakeAIAgent.instances[0].kwargs.get("max_tokens") == 2048


def test_session_constructor_passes_expected_kwargs_to_aiagent(monkeypatch):
    _patch_aiagent(monkeypatch)

    sess = V2VAgentSession(
        user_id="u1",
        api_key="sk-test",
        model="sarvam-m",
        system_prompt="SYSTEM_BODY",
        base_url="https://api.sarvam.ai/v1",
    )

    assert len(_FakeAIAgent.instances) == 1
    kwargs = _FakeAIAgent.instances[0].kwargs
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["base_url"] == "https://api.sarvam.ai/v1"
    assert kwargs["model"] == "sarvam-m"
    assert kwargs["enabled_toolsets"] == list(_DEFAULT_V2V_TOOLSETS)
    assert "SYSTEM_BODY" in kwargs["ephemeral_system_prompt"]
    assert kwargs["session_id"] == "v2v-u1"
    assert kwargs["user_id"] == "u1"
    assert kwargs["chat_id"] == "u1"
    assert kwargs["platform"] == "voice_rtc"
    assert kwargs.get("quiet_mode") is True
    assert sess.user_id == "u1"


def test_submit_user_turn_streams_tokens_and_persists_history(monkeypatch):
    _patch_aiagent(monkeypatch)

    sess = V2VAgentSession(
        user_id="u1",
        api_key="sk",
        model="sarvam-m",
        system_prompt="SYS",
    )

    tool_starts: List[Tuple[str, Dict[str, Any]]] = []
    tool_completes: List[Tuple[str, Dict[str, Any], Any]] = []

    sess._on_tool_start = lambda name, targs: tool_starts.append((name, targs))
    sess._on_tool_complete = lambda name, targs, result: tool_completes.append((name, targs, result))

    async def _go() -> List[str]:
        gen = await sess.submit_user_turn("hi")
        out: List[str] = []
        async for tok in gen:
            out.append(tok)
        return out

    tokens = asyncio.run(_go())
    assert tokens == ["Hello ", "world."]
    # History accrued user + assistant turns.
    roles = [m["role"] for m in sess._history]
    assert roles == ["user", "assistant"]
    assert sess._history[0]["content"] == "hi"
    assert sess._history[1]["content"] == "Hello world."
    # Run was invoked with a *snapshot* of history that did NOT yet
    # include this user message — AIAgent appends it internally.
    call = _FakeAIAgent.instances[0].run_calls[0]
    assert call["user_message"] == "hi"
    assert call["conversation_history"] == []
    # Tool callbacks were forwarded.
    assert tool_starts == [("lookup_order", {"id": "ACME-3456"})]
    assert tool_completes == [("lookup_order", {"id": "ACME-3456"}, {"status": "in transit"})]


def test_two_turns_share_history_via_snapshot(monkeypatch):
    _patch_aiagent(monkeypatch)

    sess = V2VAgentSession(
        user_id="u1",
        api_key="sk",
        model="sarvam-m",
        system_prompt="SYS",
    )

    async def _go() -> None:
        gen1 = await sess.submit_user_turn("first")
        async for _ in gen1:
            pass
        gen2 = await sess.submit_user_turn("second")
        async for _ in gen2:
            pass

    asyncio.run(_go())

    runs = _FakeAIAgent.instances[0].run_calls
    assert len(runs) == 2
    # Second call's history snapshot must contain user/assistant from turn 1.
    hist = runs[1]["conversation_history"]
    assert [m["role"] for m in hist] == ["user", "assistant"]
    assert hist[0]["content"] == "first"
    assert hist[1]["content"] == "Hello world."


def test_falls_back_to_final_response_when_no_deltas(monkeypatch):
    """If AIAgent produces a result but no deltas (rare), the assistant
    history entry is sourced from ``final_response``."""

    class _NoDeltaAIAgent(_FakeAIAgent):
        def run_conversation(self, *, user_message, conversation_history=None, **_):
            self.run_calls.append({"user_message": user_message, "conversation_history": list(conversation_history or [])})
            return {"final_response": "From dict."}

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", _NoDeltaAIAgent)

    sess = V2VAgentSession(
        user_id="u1",
        api_key="sk",
        model="sarvam-m",
        system_prompt="SYS",
    )

    async def _go() -> List[str]:
        gen = await sess.submit_user_turn("hi")
        out: List[str] = []
        async for tok in gen:
            out.append(tok)
        return out

    out = asyncio.run(_go())
    assert out == []
    assert sess._history[-1]["content"] == "From dict."


def _adapter_no_env(monkeypatch) -> VoiceRTCAdapter:
    for var in (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "SARVAM_API_KEY",
        "PRAVAH_API_KEY",
        "V2V_DATA_ROOT",
        "V2V_TOOLSETS",
    ):
        monkeypatch.delenv(var, raising=False)
    return VoiceRTCAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "wss://lk.example", "api_key": "ak", "api_secret": "sk"},
        )
    )


def test_build_v2v_session_passes_toolsets_and_sandwiches_directive(monkeypatch):
    _patch_aiagent(monkeypatch)
    monkeypatch.setenv("SARVAM_API_KEY", "sk-test")
    adapter = _adapter_no_env(monkeypatch)
    adapter._sarvam_api_key = "sk-test"

    captured: Dict[str, Any] = {}

    class _CapturingSession:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.user_id = kwargs.get("user_id")
            self.model = kwargs.get("model")

        async def submit_user_turn(self, text):  # pragma: no cover
            async def _g():
                yield ""
            return _g()

        async def close(self) -> None:  # pragma: no cover
            pass

    monkeypatch.setattr(
        "gateway.platforms.voice_rtc.V2VAgentSession",
        _CapturingSession,
    )

    async def _go() -> Any:
        return await adapter._sessions.get_or_create("userZ")

    asyncio.run(_go())

    # Default toolsets propagate.
    assert captured["enabled_toolsets"] == list(_DEFAULT_V2V_TOOLSETS)
    assert captured["chat_id"] == "userZ"
    # Voice-mode directive is sandwiched at the factory layer (not by the
    # session), so the system_prompt the factory hands the session must
    # already start with the directive header.
    sp = captured["system_prompt"]
    assert sp.startswith(_VOICE_MODE_DIRECTIVE)
    assert "REMINDER: spoken-conversation mode" in sp
    # Tool callbacks were wired so the data-channel forwarder can drain.
    assert callable(captured.get("on_tool_start"))
    assert callable(captured.get("on_tool_complete"))


def test_build_v2v_session_honors_v2v_toolsets_env(monkeypatch):
    _patch_aiagent(monkeypatch)
    adapter = _adapter_no_env(monkeypatch)
    adapter._sarvam_api_key = "sk-test"
    monkeypatch.setenv("SARVAM_API_KEY", "sk-test")
    monkeypatch.setenv("V2V_TOOLSETS", "skills,memory,v2v")

    captured: Dict[str, Any] = {}

    class _CapturingSession:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.user_id = kwargs.get("user_id")
            self.model = kwargs.get("model")

        async def submit_user_turn(self, text):  # pragma: no cover
            async def _g():
                yield ""
            return _g()

        async def close(self) -> None:  # pragma: no cover
            pass

    monkeypatch.setattr(
        "gateway.platforms.voice_rtc.V2VAgentSession",
        _CapturingSession,
    )

    async def _go() -> Any:
        return await adapter._sessions.get_or_create("userZ")

    asyncio.run(_go())

    assert captured["enabled_toolsets"] == ["skills", "memory", "v2v"]
