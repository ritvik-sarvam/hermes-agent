"""Tests that end_call / agent_handover tool completion triggers a hangup."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.voice_rtc import VoiceRTCAdapter


def _adapter(monkeypatch) -> VoiceRTCAdapter:
    for var in (
        "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
        "SARVAM_API_KEY", "PRAVAH_API_KEY", "V2V_DATA_ROOT", "V2V_TOOLSETS",
    ):
        monkeypatch.delenv(var, raising=False)
    return VoiceRTCAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "wss://lk.example", "api_key": "ak", "api_secret": "sk"},
        )
    )


def _install_room(adapter: VoiceRTCAdapter, room_name: str) -> Any:
    room = MagicMock()
    room.disconnect = AsyncMock()
    # local_participant.publish_data must exist but be tolerant; the test's
    # focus is room.disconnect so use a no-op AsyncMock.
    room.local_participant = MagicMock()
    room.local_participant.publish_data = AsyncMock()
    adapter._active_calls[room_name] = {
        "room": room,
        "user_id": "u1",
        "call_id": "c1",
    }
    return room


@pytest.mark.parametrize("tool_name", ["end_call", "agent_handover"])
def test_call_control_tool_triggers_hangup(monkeypatch, tool_name):
    monkeypatch.setenv("V2V_HANGUP_GRACE_MS", "10")
    adapter = _adapter(monkeypatch)
    room_name = "v2v-u1-c1"
    room = _install_room(adapter, room_name)

    async def _go() -> None:
        queue: asyncio.Queue = asyncio.Queue()
        drain_task = asyncio.create_task(adapter._drain_tool_events(room_name, queue))
        await queue.put(("tool_complete", tool_name, {"reason": "ok"}, {"status": "ok"}))
        # Wait long enough for the grace + hangup to run.
        await asyncio.sleep(0.1)
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass

    asyncio.run(_go())

    assert room.disconnect.await_count >= 1
    assert room_name not in adapter._active_calls
