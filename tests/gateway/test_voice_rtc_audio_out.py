"""Audio-out path tests for the voice_rtc adapter (Task 5.3).

Pipeline under test::

    LLM token stream
       → ClauseChunker.feed/flush()
         → SarvamTTSStream.synth(chunk) → PCM s16le @ 16 kHz bytes
            → livekit.rtc.AudioSource.capture_frame(AudioFrame)

    + barge-in:
       VAD speech_start  →  state.handle(VAD_SPEECH_START)
                         →  cancel TTS, clear AudioSource queue
                         →  state.handle(CANCEL_DONE)

Mocked at the LiveKit and Sarvam boundaries — no network in tests.

The adapter exposes a few seams the tests drive directly:

* ``_open_publish_audio(ctx, state)`` — constructs the AudioSource
  + LocalAudioTrack and publishes it on the local participant. Tests
  override it so the existing audio-in tests' MagicMock ``ctx.room`` keeps
  working without trying to ``await`` a real ``publish_track``.
* ``_on_assistant_stream(room_name, token_iter)`` — async coroutine that
  consumes an async iterable of token deltas and drives the chunker → TTS
  → AudioSource pipeline.
* ``send(chat_id, content)`` — one-shot fallback that runs the same
  pipeline over a single string.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.voice_rtc import VoiceRTCAdapter
from tools.voice_rtc.state import Event as TurnEvent
from tools.voice_rtc.state import TurnState


# ----------------------------------------------------------------------
# Helpers / fakes
# ----------------------------------------------------------------------


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
    """Records every captured AudioFrame so tests can assert PCM flow."""

    def __init__(self, sample_rate: int = 16000, num_channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.frames: List[Any] = []
        self.queue_cleared = 0
        self.closed = False

    async def capture_frame(self, frame: Any) -> None:
        self.frames.append(frame)

    def clear_queue(self) -> None:
        self.queue_cleared += 1

    async def aclose(self) -> None:
        self.closed = True


class _FakeTTSStream:
    """Test double for SarvamTTSStream — returns a configurable burst of
    PCM bytes per ``synth(text)`` call."""

    def __init__(
        self,
        bursts: Dict[str, List[bytes]] | None = None,
        *,
        delay: float = 0.0,
        on_synth=None,
    ) -> None:
        # Per-text bursts. If a text isn't in the mapping a default 640-byte
        # (20 ms @ 16 kHz mono s16le) frame is yielded once.
        self.bursts = bursts or {}
        self.delay = delay
        self.on_synth = on_synth
        self.calls: List[str] = []
        self.cancelled_synths: List[str] = []
        self.closed = False

    async def synth(self, text: str):
        self.calls.append(text)
        if self.on_synth is not None:
            await self.on_synth(text)
        chunks = self.bursts.get(text, [b"\x01\x00" * 320])
        try:
            for ch in chunks:
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield ch
        except (asyncio.CancelledError, GeneratorExit):
            self.cancelled_synths.append(text)
            raise

    async def close(self) -> None:
        self.closed = True


async def _aiter_from_list(tokens):
    for t in tokens:
        yield t


# ----------------------------------------------------------------------
# Audio-out: token stream → chunker → TTS → AudioSource
# ----------------------------------------------------------------------


def test_assistant_stream_emits_tts_chunks_to_audio_source(monkeypatch):
    """Each clause boundary becomes a TTS call; PCM bytes are captured
    into the AudioSource frame-by-frame."""
    adapter = _adapter_no_env(monkeypatch)

    src = _FakeAudioSource()
    turn_state = TurnState()
    room_name = "v2v-u-c"
    adapter._active_calls[room_name] = {
        "audio_source": src,
        "turn_state": turn_state,
    }

    # Two known utterances → two distinct PCM bursts.
    fake_tts = _FakeTTSStream({
        "Hello there.": [b"\xaa\x00" * 320, b"\xab\x00" * 320],   # 2 frames
        "How are you today?": [b"\xbb\x00" * 320],                 # 1 frame
    })
    monkeypatch.setattr(
        "tools.sarvam_tts.SarvamTTSStream",
        lambda *a, **kw: fake_tts,
    )

    # Token stream emitting the message split into many small deltas.
    raw = "Hello there. How are you today?"

    async def _go():
        # Drive the FSM into THINKING (USER_FINAL) so the first TTS frame
        # transitions us into SPEAKING per the FSM contract.
        turn_state.handle(TurnEvent.USER_FINAL)
        await adapter._on_assistant_stream(
            room_name,
            _aiter_from_list(list(raw)),
        )

    asyncio.run(_go())

    # Two TTS calls — one per clause.
    assert fake_tts.calls == ["Hello there.", "How are you today?"]

    # Frames captured: 2 + 1 = 3 AudioFrame instances.
    assert len(src.frames) == 3
    # Each frame is a 20 ms s16le @ 16 kHz mono frame. ``frame.data`` is
    # a memoryview into a uint16 array, so its ``len()`` is the sample
    # count (not byte count) — which matches ``samples_per_channel``.
    for frame in src.frames:
        assert frame.sample_rate == 16000
        assert frame.num_channels == 1
        assert frame.samples_per_channel == 320  # 20 ms @ 16 kHz mono


def test_first_audio_transitions_state_to_speaking(monkeypatch):
    """Capturing the first frame must dispatch TTS_FIRST_AUDIO so
    THINKING → SPEAKING."""
    adapter = _adapter_no_env(monkeypatch)
    src = _FakeAudioSource()
    turn_state = TurnState()
    turn_state.handle(TurnEvent.USER_FINAL)  # → THINKING
    assert turn_state.state == "THINKING"

    room_name = "v2v-u-c"
    adapter._active_calls[room_name] = {
        "audio_source": src,
        "turn_state": turn_state,
    }

    observed: List[str] = []

    fake_tts = _FakeTTSStream({"Hi.": [b"\x01\x00" * 320]})

    # Wrap capture_frame so we observe state at the moment the first frame lands.
    orig_capture = src.capture_frame

    async def _capture(frame):
        observed.append(turn_state.state)
        await orig_capture(frame)

    src.capture_frame = _capture  # type: ignore[assignment]

    monkeypatch.setattr(
        "tools.sarvam_tts.SarvamTTSStream",
        lambda *a, **kw: fake_tts,
    )

    async def _go():
        await adapter._on_assistant_stream(room_name, _aiter_from_list(["Hi."]))

    asyncio.run(_go())

    # When the first frame was captured, the FSM had already advanced.
    assert observed and observed[0] == "SPEAKING"


def test_tts_done_returns_to_listening(monkeypatch):
    """An unbarged turn ends with TTS_DONE → LISTENING."""
    adapter = _adapter_no_env(monkeypatch)
    src = _FakeAudioSource()
    turn_state = TurnState()
    turn_state.handle(TurnEvent.USER_FINAL)

    room_name = "v2v-u-c"
    adapter._active_calls[room_name] = {
        "audio_source": src,
        "turn_state": turn_state,
    }

    fake_tts = _FakeTTSStream({"Done.": [b"\x01\x00" * 320]})
    monkeypatch.setattr(
        "tools.sarvam_tts.SarvamTTSStream",
        lambda *a, **kw: fake_tts,
    )

    async def _go():
        await adapter._on_assistant_stream(room_name, _aiter_from_list(["Done."]))

    asyncio.run(_go())
    assert turn_state.state == "LISTENING"


def test_barge_in_cancels_tts(monkeypatch):
    """Mid-stream VAD speech_start cancels TTS, drains the audio source
    queue, and the FSM returns to LISTENING via INTERRUPTED → CANCEL_DONE.
    No frames captured after the cancel signal."""
    adapter = _adapter_no_env(monkeypatch)
    src = _FakeAudioSource()
    turn_state = TurnState()
    turn_state.handle(TurnEvent.USER_FINAL)

    room_name = "v2v-u-c"
    adapter._active_calls[room_name] = {
        "audio_source": src,
        "turn_state": turn_state,
    }

    barge_event = asyncio.Event()
    cancel_marker = {"frames_at_cancel": -1}

    async def _on_synth(text: str) -> None:
        # First synth call: signal the test that streaming has started so
        # it can fire the barge-in.
        if text == "First sentence.":
            barge_event.set()

    # 50 PCM frames per sentence — slow enough that the test can fire a
    # barge-in mid-stream.
    fake_tts = _FakeTTSStream(
        {
            "First sentence.": [b"\xaa\x00" * 320] * 50,
            "Second sentence.": [b"\xbb\x00" * 320] * 50,
        },
        delay=0.005,
        on_synth=_on_synth,
    )
    monkeypatch.setattr(
        "tools.sarvam_tts.SarvamTTSStream",
        lambda *a, **kw: fake_tts,
    )

    async def _go():
        # Spawn the audio-out coroutine.
        text = "First sentence. Second sentence."
        stream_task = asyncio.create_task(
            adapter._on_assistant_stream(
                room_name,
                _aiter_from_list(list(text)),
            )
        )

        # Wait for the first TTS call to begin streaming.
        await barge_event.wait()
        # Let a few frames accumulate.
        await asyncio.sleep(0.02)
        cancel_marker["frames_at_cancel"] = len(src.frames)

        # Fire VAD speech_start through the adapter's barge-in handler.
        await adapter._on_vad_speech_start(room_name)

        # The audio-out task should now finish (cancellation propagated).
        try:
            await asyncio.wait_for(stream_task, timeout=2.0)
        except asyncio.TimeoutError:
            stream_task.cancel()
            raise

    asyncio.run(_go())

    # FSM transitioned through INTERRUPTED back to LISTENING.
    assert turn_state.state == "LISTENING"
    assert turn_state.cancel_requested is False

    # AudioSource queue was cleared on barge-in.
    assert src.queue_cleared >= 1

    # No frames captured after cancellation.
    assert len(src.frames) == cancel_marker["frames_at_cancel"]

    # Second sentence was either never started or was cancelled — under
    # serial execution the safe assertion is that fewer than 100 frames
    # total reached the source (cap on 50 from the first burst).
    assert len(src.frames) < 100


def test_audio_in_real_track_subscribe_resamples_to_16k(monkeypatch):
    """The real ``_iter_audio_frames`` uses LiveKit's AudioStream with
    sample_rate=16000 so resampling happens inside the SDK. We assert the
    constructor was called with the right keyword args and the bytes
    yielded reach ASR."""
    adapter = _adapter_no_env(monkeypatch)

    seen_kwargs: Dict[str, Any] = {}

    class _FakeFrame:
        def __init__(self, data: bytes) -> None:
            self.data = bytearray(data)

    class _FakeAudioFrameEvent:
        def __init__(self, data: bytes) -> None:
            self.frame = _FakeFrame(data)

    class _FakeAudioStream:
        def __init__(self, track, **kwargs) -> None:
            seen_kwargs.update(kwargs)
            self._frames = [
                _FakeAudioFrameEvent(b"\x10\x00" * 320),
                _FakeAudioFrameEvent(b"\x20\x00" * 320),
            ]

        def __aiter__(self):
            self._iter = iter(self._frames)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("livekit.rtc.AudioStream", _FakeAudioStream)

    # Build a fake ctx.room with a participant having one audio track.
    track = MagicMock()
    track.kind = 1  # arbitrary; not inspected because we patched AudioStream
    track.sid = "tr1"

    publication = MagicMock()
    publication.track = track
    publication.kind = 1

    participant = MagicMock()
    participant.track_publications = {"tr1": publication}

    room = MagicMock()
    room.remote_participants = {"p1": participant}
    # Capture .on() handlers so we can simulate track_subscribed.
    handlers: Dict[str, Any] = {}

    def _on(event, handler=None):
        if handler is None:
            def _decorator(h):
                handlers[event] = h
                return h
            return _decorator
        handlers[event] = handler
        return handler
    room.on = _on
    room.off = lambda *a, **kw: None

    ctx = MagicMock()
    ctx.room = room

    collected: List[bytes] = []

    async def _go():
        agen = adapter._iter_audio_frames(ctx)
        # Schedule a track_subscribed event after we begin iterating.
        async def _trigger():
            await asyncio.sleep(0)
            handler = handlers.get("track_subscribed")
            assert handler is not None, "adapter must register track_subscribed handler"
            handler(track, publication, participant)
        trigger_task = asyncio.create_task(_trigger())

        async for frame in agen:
            collected.append(bytes(frame))
            if len(collected) >= 2:
                break
        await trigger_task
        # Close the generator cleanly.
        await agen.aclose()

    asyncio.run(_go())

    # AudioStream was constructed with the 16 kHz mono target.
    assert seen_kwargs.get("sample_rate") == 16000
    assert seen_kwargs.get("num_channels") == 1
    assert collected[0] == b"\x10\x00" * 320
    assert collected[1] == b"\x20\x00" * 320


def test_audio_in_backfill_picks_up_already_subscribed_tracks(monkeypatch):
    """Regression: when ``ctx.connect()`` auto-subscribes the user's mic
    track BEFORE ``_iter_audio_frames`` registers its ``track_subscribed``
    listener, the listener never sees the event and audio is silently
    dropped.

    Fix is a backfill scan over ``room.remote_participants`` immediately
    after registering the listener. This test exercises that path
    explicitly: tracks are present in ``remote_participants`` but the
    ``track_subscribed`` event is NEVER manually fired. Frames must
    still flow.
    """
    adapter = _adapter_no_env(monkeypatch)

    class _FakeFrame:
        def __init__(self, data: bytes) -> None:
            self.data = bytearray(data)

    class _FakeAudioFrameEvent:
        def __init__(self, data: bytes) -> None:
            self.frame = _FakeFrame(data)

    class _FakeAudioStream:
        def __init__(self, track, **kwargs) -> None:
            self._frames = [
                _FakeAudioFrameEvent(b"\xaa" * 640),
                _FakeAudioFrameEvent(b"\xbb" * 640),
            ]

        def __aiter__(self):
            self._iter = iter(self._frames)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("livekit.rtc.AudioStream", _FakeAudioStream)

    # Track is ALREADY subscribed before we get here — exactly what
    # LiveKit auto_subscribe gives us when ctx.connect() returns.
    track = MagicMock()
    track.kind = 1
    track.sid = "track-existing"
    publication = MagicMock()
    publication.track = track
    publication.kind = 1
    participant = MagicMock()
    participant.sid = "p-1"
    participant.track_publications = {"track-existing": publication}

    room = MagicMock()
    room.remote_participants = {"p-1": participant}
    # The listener IS registered, but never fired — only the backfill
    # path can produce frames.
    room.on = lambda *a, **kw: None
    room.off = lambda *a, **kw: None

    ctx = MagicMock()
    ctx.room = room

    collected: List[bytes] = []

    async def _go():
        agen = adapter._iter_audio_frames(ctx)
        async for frame in agen:
            collected.append(bytes(frame))
            if len(collected) >= 2:
                break
        await agen.aclose()

    asyncio.run(_go())

    # Without the backfill scan this list would be empty and the loop
    # would hang waiting for a track_subscribed event that already fired.
    assert collected == [b"\xaa" * 640, b"\xbb" * 640]


def test_audio_in_backfill_does_not_double_subscribe(monkeypatch):
    """If the same track shows up via BOTH the backfill scan AND a
    delayed ``track_subscribed`` event (which can happen if LiveKit
    fires the event after we've already walked ``remote_participants``),
    we must not start two ``AudioStream`` drains for it."""
    adapter = _adapter_no_env(monkeypatch)

    audio_stream_constructions: List[Any] = []

    class _FakeFrame:
        def __init__(self, data: bytes) -> None:
            self.data = bytearray(data)

    class _FakeAudioFrameEvent:
        def __init__(self, data: bytes) -> None:
            self.frame = _FakeFrame(data)

    class _FakeAudioStream:
        def __init__(self, track, **kwargs) -> None:
            audio_stream_constructions.append(track)
            self._frames = [_FakeAudioFrameEvent(b"\x01" * 640)]

        def __aiter__(self):
            self._iter = iter(self._frames)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("livekit.rtc.AudioStream", _FakeAudioStream)

    track = MagicMock()
    track.kind = 1
    track.sid = "track-dup"
    publication = MagicMock()
    publication.track = track
    publication.kind = 1
    participant = MagicMock()
    participant.sid = "p-1"
    participant.track_publications = {"track-dup": publication}

    handlers: Dict[str, Any] = {}

    def _on(event, handler):
        handlers[event] = handler

    room = MagicMock()
    room.remote_participants = {"p-1": participant}
    room.on = _on
    room.off = lambda *a, **kw: None

    ctx = MagicMock()
    ctx.room = room

    async def _go():
        agen = adapter._iter_audio_frames(ctx)

        # Fire a delayed track_subscribed for the SAME track the backfill
        # already saw. This must NOT trigger a second drain.
        async def _trigger():
            await asyncio.sleep(0.01)
            cb = handlers.get("track_subscribed")
            if cb is not None:
                cb(track, publication, participant)

        trigger_task = asyncio.create_task(_trigger())

        collected = []
        try:
            async for frame in agen:
                collected.append(bytes(frame))
                if len(collected) >= 1:
                    break
        finally:
            trigger_task.cancel()
            try:
                await trigger_task
            except asyncio.CancelledError:
                pass
        await agen.aclose()

    asyncio.run(_go())

    # Exactly one AudioStream was constructed — backfill OR future-event,
    # not both.
    assert len(audio_stream_constructions) == 1, (
        f"track was drained {len(audio_stream_constructions)} times — "
        "backfill+future-event de-dup broken"
    )


def test_vad_speech_start_event_drives_barge_in(monkeypatch):
    """The VAD-event consumer drains a Silero-style event stream and
    routes ``start_of_speech`` events into ``_on_vad_speech_start``,
    which advances the FSM through INTERRUPTED → CANCEL_DONE.

    This closes the loop the production audio-in path needs: VAD output
    → FSM cancel signal, without the test having to call the barge-in
    method directly."""
    adapter = _adapter_no_env(monkeypatch)
    src = _FakeAudioSource()
    turn_state = TurnState()
    turn_state.handle(TurnEvent.USER_FINAL)  # → THINKING
    turn_state.handle(TurnEvent.TTS_FIRST_AUDIO)  # → SPEAKING

    room_name = "v2v-u-c"
    adapter._active_calls[room_name] = {
        "audio_source": src,
        "turn_state": turn_state,
        "user_id": "u",
        "call_id": "c",
    }

    class _FakeVADEvent:
        def __init__(self, value: str) -> None:
            self.type = type("T", (), {"value": value})()

    async def _vad_stream():
        yield _FakeVADEvent("start_of_speech")

    async def _go():
        await adapter._consume_vad_events(_vad_stream(), room_name)

    asyncio.run(_go())

    # The FSM transitioned through INTERRUPTED back to LISTENING because
    # _on_vad_speech_start ran end-to-end on the speech_start event.
    assert turn_state.state == "LISTENING"
    assert turn_state.cancel_requested is False
    # The audio source's queue was cleared as part of the barge-in.
    assert src.queue_cleared >= 1


def test_send_uses_chunker_and_tts(monkeypatch):
    """``send(chat_id, content)`` runs the same pipeline as a single-shot
    text utterance — chunker, TTS, audio source."""
    adapter = _adapter_no_env(monkeypatch)
    src = _FakeAudioSource()
    turn_state = TurnState()

    room_name = "v2v-u-c"
    adapter._active_calls[room_name] = {
        "audio_source": src,
        "turn_state": turn_state,
    }

    fake_tts = _FakeTTSStream({"Confirmation": [b"\xcc\x00" * 320]})
    monkeypatch.setattr(
        "tools.sarvam_tts.SarvamTTSStream",
        lambda *a, **kw: fake_tts,
    )

    async def _go():
        result = await adapter.send(room_name, "Confirmation")
        assert result.success is True

    asyncio.run(_go())

    assert fake_tts.calls == ["Confirmation"]
    assert len(src.frames) == 1
    assert src.frames[0].sample_rate == 16000
