"""Audio-in path tests for the voice_rtc adapter (Task 5.2).

The audio-in pipeline is::

    LiveKit room participant
      └─ audio track   ──► 20ms PCM s16le @ 16 kHz
            ├─► Silero VAD  (drives barge-in via TurnState in 5.3)
            └─► SarvamASRStream  ──► {"type":"final","text":...}
                       └─► MessageEvent  ──► self.handle_message(event)

These tests stub at the LiveKit and Sarvam boundaries so nothing here
opens a network connection. Run them with no env vars set::

    uv run pytest tests/gateway/test_voice_rtc_audio_in.py -q

The `_iter_audio_frames` async-iterator is overridden so we can drive
synthetic frames deterministically. Likewise the ASR stream is replaced
with a tiny class whose ``events()`` queue is fed by the test body.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.platforms.voice_rtc import (
    VoiceRTCAdapter,
    _parse_room_name,
)


# ----------------------------------------------------------------------
# Room-name parser
# ----------------------------------------------------------------------


def test_parse_room_name_ok():
    assert _parse_room_name("v2v-user42-call_abc123") == ("user42", "call_abc123")


def test_parse_room_name_keeps_uuid_dashes_in_call_id():
    # call_id may itself contain dashes (e.g. UUID v4). Splitting on the
    # first dash after the v2v- prefix preserves them.
    assert _parse_room_name("v2v-u1-aaaa-bbbb-cccc") == ("u1", "aaaa-bbbb-cccc")


@pytest.mark.parametrize("name", [
    "",
    "lobby",
    "v2v-",
    "v2v-onlyuser",
    "v2v--missinguser",
    "v2v-user-",  # empty call_id
    None,
    123,
])
def test_parse_room_name_rejects_malformed(name):
    with pytest.raises(ValueError):
        _parse_room_name(name)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _adapter_no_env(monkeypatch) -> VoiceRTCAdapter:
    for var in (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "SARVAM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    # Audio-in tests don't exercise the opening greeting; turning it off
    # keeps the call-state machinery simple (no synthetic first turn).
    monkeypatch.setenv("V2V_AGENT_OPENS_CALL", "false")
    return VoiceRTCAdapter(PlatformConfig(enabled=True, extra={
        "url": "wss://lk.example",
        "api_key": "ak",
        "api_secret": "sk",
    }))


class _FakeASR:
    """Test double for SarvamASRStream — events() yields whatever is
    pushed via ``emit``; ``push_pcm`` records frames seen."""

    def __init__(self, *_, **__) -> None:
        self.frames: List[bytes] = []
        self.closed = False
        self._queue: asyncio.Queue = asyncio.Queue()

    async def push_pcm(self, frame: bytes) -> None:
        self.frames.append(frame)

    async def events(self):
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def close(self) -> None:
        self.closed = True
        await self._queue.put(None)

    # Test-only API
    async def emit(self, ev: dict) -> None:
        await self._queue.put(ev)


class _FakeVAD:
    """Records pushed frames so we can assert the fanout."""

    def __init__(self) -> None:
        self.frames: List[bytes] = []

    def push_frame(self, frame: bytes) -> None:
        self.frames.append(frame)


# ----------------------------------------------------------------------
# connect() worker bootstrap
# ----------------------------------------------------------------------


def test_connect_starts_worker_with_credentials(monkeypatch):
    """connect() wraps WorkerOptions with the configured URL/key/secret
    and dispatches a background task running the worker."""
    adapter = _adapter_no_env(monkeypatch)

    seen: Dict[str, Any] = {}

    fake_worker = MagicMock()
    fake_worker.run = AsyncMock()
    fake_worker.aclose = AsyncMock()

    def _fake_build_worker(options):
        seen["options"] = options
        return fake_worker

    adapter._build_worker = _fake_build_worker  # type: ignore[method-assign]

    async def _go():
        ok = await adapter.connect()
        assert ok is True
        assert adapter.is_connected is True
        # The worker task is running; let it tick once so it actually
        # invokes worker.run(). Cancellation happens in disconnect.
        await asyncio.sleep(0)
        await adapter.disconnect()
        assert adapter.is_connected is False

    asyncio.run(_go())

    options = seen["options"]
    assert options.ws_url == "wss://lk.example"
    assert options.api_key == "ak"
    assert options.api_secret == "sk"
    # The entrypoint is the adapter's bound method — used by livekit-agents
    # to dispatch a job per room.
    assert options.entrypoint_fnc == adapter._entrypoint
    # CRITICAL: must use thread executor, not the default process pool.
    # The default tries to pickle WorkerOptions (which closes over the
    # adapter's non-picklable state — asyncio locks, SessionRegistry,
    # logger handles) and crashes with `cannot pickle '_thread.lock'
    # object`. A regression here is a runtime fail, not a test fail —
    # this assertion is what catches it.
    from livekit.agents import JobExecutorType  # type: ignore
    assert options.job_executor_type == JobExecutorType.THREAD
    fake_worker.run.assert_awaited()


# ----------------------------------------------------------------------
# Per-frame fanout
# ----------------------------------------------------------------------


def test_audio_frame_pushes_to_asr_and_vad(monkeypatch):
    """Each PCM frame fed through ``_handle_audio_frame`` goes to *both*
    the ASR stream and the VAD instance."""
    adapter = _adapter_no_env(monkeypatch)
    asr = _FakeASR()
    vad = _FakeVAD()
    state = {"asr": asr, "vad": vad}

    async def _go():
        await adapter._handle_audio_frame(b"\x00\x01" * 320, state)
        await adapter._handle_audio_frame(b"\x02\x03" * 320, state)

    asyncio.run(_go())
    assert asr.frames == [b"\x00\x01" * 320, b"\x02\x03" * 320]
    assert vad.frames == [b"\x00\x01" * 320, b"\x02\x03" * 320]


# ----------------------------------------------------------------------
# Audio reader → ASR final → handle_message
# ----------------------------------------------------------------------


def test_asr_final_emits_message_event(monkeypatch):
    """When the ASR stream emits a final, the adapter routes the parsed
    text to the per-user agent session (M6 contract).

    The pre-M6 path called ``handle_message`` directly; M6 attached a
    ``V2VAgentSession`` per user_id and feeds finals into
    ``submit_user_turn``. This test now asserts the session received the
    text and that on_assistant_token_stream was invoked with the parsed
    chat_id (so the audio-out pipeline downstream gets the right room)."""
    adapter = _adapter_no_env(monkeypatch)

    # Record the (chat_id, tokens) pair the audio-out pipeline would see.
    captured: List[tuple] = []

    async def _capture_stream(chat_id, token_iterator):
        tokens = []
        async for t in token_iterator:
            tokens.append(t)
        captured.append((chat_id, tokens))

    adapter.on_assistant_token_stream = _capture_stream  # type: ignore[method-assign]

    # Inject a fake session factory so we don't construct a real
    # AsyncOpenAI client (no network).
    submitted: List[tuple] = []

    class _FakeSession:
        def __init__(self, user_id):
            self.user_id = user_id

        async def submit_user_turn(self, text):
            submitted.append((self.user_id, text))

            async def _gen():
                yield "ack"

            return _gen()

        async def close(self):
            pass

    async def _factory(user_id):
        return _FakeSession(user_id)

    adapter._sessions._factory = _factory

    fake_asr = _FakeASR()

    # Patch SarvamASRStream so _on_room constructs our fake.
    monkeypatch.setattr(
        "tools.sarvam_asr.SarvamASRStream",
        lambda *a, **kw: fake_asr,
    )

    # Stub the publish-audio seam so the MagicMock ctx.room doesn't
    # blow up on `await local_participant.publish_track(...)`.
    async def _no_publish(self_, ctx, state):
        return

    monkeypatch.setattr(
        "gateway.platforms.voice_rtc.VoiceRTCAdapter._open_publish_audio",
        _no_publish,
    )

    # Stub the audio-frame iterator so _read_room_audio doesn't block.
    async def _no_frames(self_, ctx):
        if False:  # pragma: no cover
            yield b""
        return

    monkeypatch.setattr(
        "gateway.platforms.voice_rtc.VoiceRTCAdapter._iter_audio_frames",
        _no_frames,
    )

    # Build a minimal ctx with a room object.
    ctx = MagicMock()
    ctx.room = MagicMock()
    ctx.room.name = "v2v-userX-call99"

    async def _go():
        await adapter._on_room(ctx, "userX", "call99")
        # Push a final through the fake ASR; the consumer task should
        # pick it up and route into the session.
        await fake_asr.emit({"type": "final", "text": "hello world"})
        for _ in range(5):
            await asyncio.sleep(0)
        await fake_asr.close()
        for _ in range(5):
            await asyncio.sleep(0)

    asyncio.run(_go())

    assert submitted == [("userX", "hello world")]
    assert captured and captured[0][0] == "v2v-userX-call99"


def test_asr_final_drops_empty_text(monkeypatch):
    """Empty/whitespace-only finals from ASR should not produce a
    MessageEvent — they're noise from the streaming protocol."""
    adapter = _adapter_no_env(monkeypatch)
    seen: List[MessageEvent] = []

    async def _capture(event):
        seen.append(event)

    adapter.handle_message = _capture  # type: ignore[method-assign]

    async def _go():
        await adapter._deliver_final("   ", "userX", "v2v-userX-call99")
        await adapter._deliver_final("", "userX", "v2v-userX-call99")

    asyncio.run(_go())
    assert seen == []


# ----------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------


def test_disconnect_closes_asr(monkeypatch):
    """disconnect() must call asr.close() for each active call and
    cancel any audio/consumer tasks so we don't leak fds or coroutines."""
    adapter = _adapter_no_env(monkeypatch)
    fake_asr = _FakeASR()

    async def _runs_until_cancel():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    async def _go():
        # Inject a fake active call directly.
        consumer_task = asyncio.create_task(_runs_until_cancel())
        audio_task = asyncio.create_task(_runs_until_cancel())
        adapter._active_calls["v2v-u-c"] = {
            "asr": fake_asr,
            "asr_consumer_task": consumer_task,
            "audio_task": audio_task,
        }

        # No worker started — just exercise the cleanup branch.
        await adapter.disconnect()

        # Yield once so the cancellation propagates and the tasks reach
        # terminal state — ``cancel()`` is fire-and-forget.
        for t in (consumer_task, audio_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        assert fake_asr.closed is True
        assert consumer_task.cancelled() or consumer_task.done()
        assert audio_task.cancelled() or audio_task.done()
        assert adapter._active_calls == {}
        assert adapter.is_connected is False

    asyncio.run(_go())


def test_entrypoint_silently_skips_non_v2v_rooms(monkeypatch):
    """A room not matching the v2v- prefix must NOT crash the worker —
    the LiveKit deployment may host other agents alongside us."""
    adapter = _adapter_no_env(monkeypatch)

    on_room = AsyncMock()
    adapter._on_room = on_room  # type: ignore[method-assign]

    ctx = MagicMock()
    ctx.room = MagicMock()
    ctx.room.name = "telegram-123"

    async def _go():
        # Should return without raising.
        await adapter._entrypoint(ctx)

    asyncio.run(_go())
    on_room.assert_not_awaited()
