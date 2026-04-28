"""Data-channel transcript event tests for the voice_rtc adapter.

Mocks at the LiveKit boundary — ``room.local_participant.publish_data`` is
an ``AsyncMock`` whose call args we inspect to assert the wire format the
browser visualiser consumes.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.voice_rtc import VoiceRTCAdapter, _TRANSCRIPT_TOPIC
from tools.voice_rtc.state import Event as TurnEvent
from tools.voice_rtc.state import TurnState


def _adapter_no_env(monkeypatch) -> VoiceRTCAdapter:
    for var in (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "SARVAM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return VoiceRTCAdapter(PlatformConfig(enabled=True, extra={
        "url": "wss://lk.example",
        "api_key": "ak",
        "api_secret": "sk",
    }))


class _FakeAudioSource:
    def __init__(self) -> None:
        self.frames: List[Any] = []
        self.queue_cleared = 0

    async def capture_frame(self, frame: Any) -> None:
        self.frames.append(frame)

    def clear_queue(self) -> None:
        self.queue_cleared += 1


class _FakeTTSStream:
    def __init__(self, bursts: Dict[str, List[bytes]] | None = None) -> None:
        self.bursts = bursts or {}
        self.calls: List[str] = []

    async def synth(self, text: str):
        self.calls.append(text)
        chunks = self.bursts.get(text, [b"\x01\x00" * 320])
        for ch in chunks:
            yield ch

    async def close(self) -> None:
        pass


def _make_room_with_publish() -> tuple[MagicMock, AsyncMock]:
    room = MagicMock()
    publish = AsyncMock()
    room.local_participant = MagicMock()
    room.local_participant.publish_data = publish
    return room, publish


def _decode_payloads(publish: AsyncMock) -> List[tuple[Dict[str, Any], Dict[str, Any]]]:
    """Return a list of (payload_dict, kwargs) for every publish_data call."""
    out: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for call in publish.await_args_list:
        args, kwargs = call
        payload_bytes = args[0] if args else kwargs.get("payload")
        decoded = json.loads(payload_bytes.decode("utf-8"))
        out.append((decoded, kwargs))
    return out


def test_user_final_publishes_user_message_event(monkeypatch):
    adapter = _adapter_no_env(monkeypatch)
    room_name = "v2v-u-c"
    room, publish = _make_room_with_publish()
    adapter._active_calls[room_name] = {
        "user_id": "u",
        "call_id": "c",
        "turn_state": TurnState(),
        "room": room,
        "assistant_buffer": [],
    }

    async def _go():
        await adapter._deliver_final("hello world", "u", room_name)

    asyncio.run(_go())

    assert publish.await_count >= 1
    payloads = _decode_payloads(publish)
    msg_events = [p for p, _ in payloads if p.get("type") == "user_message"]
    assert len(msg_events) == 1
    assert msg_events[0]["text"] == "hello world"
    assert "ts" in msg_events[0]

    # Topic was supplied as a kwarg.
    user_msg_kwargs = next(kw for p, kw in payloads if p.get("type") == "user_message")
    assert user_msg_kwargs.get("topic") == _TRANSCRIPT_TOPIC
    assert _TRANSCRIPT_TOPIC == "v2v.transcript"


def test_assistant_stream_publishes_chunks_and_done(monkeypatch):
    adapter = _adapter_no_env(monkeypatch)
    room_name = "v2v-u-c"
    room, publish = _make_room_with_publish()

    src = _FakeAudioSource()
    turn_state = TurnState()
    turn_state.handle(TurnEvent.USER_FINAL)

    adapter._active_calls[room_name] = {
        "audio_source": src,
        "turn_state": turn_state,
        "user_id": "u",
        "call_id": "c",
        "room": room,
        "assistant_buffer": [],
    }

    fake_tts = _FakeTTSStream({
        "Hello there.": [b"\xaa\x00" * 320],
        "How are you?": [b"\xbb\x00" * 320],
    })
    monkeypatch.setattr(
        "tools.sarvam_tts.SarvamTTSStream",
        lambda *a, **kw: fake_tts,
    )

    async def _tokens():
        yield "Hello there. "
        yield "How are you?"

    async def _go():
        await adapter._on_assistant_stream(room_name, _tokens())

    asyncio.run(_go())

    payloads = _decode_payloads(publish)
    chunks = [p for p, _ in payloads if p.get("type") == "assistant_chunk"]
    dones = [p for p, _ in payloads if p.get("type") == "assistant_done"]

    assert len(chunks) >= 2
    assert len(dones) == 1
    assert dones[0]["text"] == "".join(c["text"] for c in chunks)


def test_barge_in_publishes_event(monkeypatch):
    adapter = _adapter_no_env(monkeypatch)
    room_name = "v2v-u-c"
    room, publish = _make_room_with_publish()

    turn_state = TurnState()
    turn_state.handle(TurnEvent.USER_FINAL)
    assert turn_state.state == "THINKING"

    adapter._active_calls[room_name] = {
        "turn_state": turn_state,
        "user_id": "u",
        "call_id": "c",
        "room": room,
        "audio_source": _FakeAudioSource(),
        "assistant_buffer": [],
    }

    async def _go():
        await adapter._on_vad_speech_start(room_name)

    asyncio.run(_go())

    payloads = _decode_payloads(publish)
    barge = [p for p, _ in payloads if p.get("type") == "barge_in"]
    assert len(barge) == 1


def test_publish_event_no_op_when_room_missing(monkeypatch):
    adapter = _adapter_no_env(monkeypatch)
    room_name = "v2v-u-c"
    adapter._active_calls[room_name] = {"room": None}

    async def _go():
        await adapter._publish_event(room_name, {"type": "x"})

    asyncio.run(_go())


def test_publish_event_falls_back_when_topic_kw_unsupported(monkeypatch):
    adapter = _adapter_no_env(monkeypatch)
    room_name = "v2v-u-c"

    calls: List[Dict[str, Any]] = []

    async def _success(*args, **kwargs):
        return None

    def _publish(*args, **kwargs):
        calls.append({"args": args, "kwargs": dict(kwargs)})
        if "reliable" in kwargs:
            raise TypeError("unexpected keyword argument 'reliable'")
        return _success()

    room = MagicMock()
    room.local_participant = MagicMock()
    room.local_participant.publish_data = _publish

    adapter._active_calls[room_name] = {"room": room}

    async def _go():
        await adapter._publish_event(room_name, {"type": "x"})

    asyncio.run(_go())

    assert len(calls) == 2
    assert "reliable" in calls[0]["kwargs"]
    assert "reliable" not in calls[1]["kwargs"]
    assert calls[1]["kwargs"].get("topic") == "v2v.transcript"
