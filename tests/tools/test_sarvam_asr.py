"""Tests for the streaming Sarvam Saaras ASR client wrapper.

The wrapper at ``tools.sarvam_asr.SarvamASRStream`` is intentionally thin:
it opens a Sarvam SDK ``speech_to_text_streaming.connect`` session lazily,
forwards PCM frames the caller pushes, and re-emits the SDK's events as a
normalized shape (``{"type": ..., "text": ...}``) so the rest of the
codebase doesn't depend on Sarvam's response model directly.

These tests mock the SDK boundary — no live network calls.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fakes for the Sarvam SDK boundary
# ---------------------------------------------------------------------------


class FakeSocket:
    """Stand-in for ``AsyncSpeechToTextStreamingSocketClient``.

    Records ``transcribe`` calls and yields a scripted sequence of SDK
    response objects when async-iterated.
    """

    def __init__(self, scripted_events: list):
        self.transcribe_calls: list = []
        self.flush_called = False
        self._scripted = list(scripted_events)
        self._iter_started = asyncio.Event()

    async def transcribe(self, audio: str, encoding: str = "audio/wav", sample_rate: int = 16000):
        self.transcribe_calls.append(
            {"audio": audio, "encoding": encoding, "sample_rate": sample_rate}
        )

    async def flush(self) -> None:
        self.flush_called = True

    def __aiter__(self):
        self._iter_started.set()
        return self._iterator()

    async def _iterator(self):
        for ev in self._scripted:
            await asyncio.sleep(0)
            yield ev


class FakeASRStreamingClient:
    """Stand-in for ``client.speech_to_text_streaming``."""

    def __init__(self, socket: FakeSocket):
        self._socket = socket
        self.connect_kwargs: dict | None = None

    def connect(self, **kwargs):
        self.connect_kwargs = kwargs
        socket = self._socket

        @asynccontextmanager
        async def _ctx():
            yield socket

        return _ctx()


class FakeAsyncSarvamAI:
    """Stand-in for ``AsyncSarvamAI``."""

    def __init__(self, *, api_subscription_key: str, socket: FakeSocket):
        self.api_subscription_key = api_subscription_key
        self.speech_to_text_streaming = FakeASRStreamingClient(socket)


def _make_response(rtype: str, **data) -> SimpleNamespace:
    """Build a fake SDK response: ``SimpleNamespace(type=..., data=...)``."""
    return SimpleNamespace(type=rtype, data=SimpleNamespace(**data))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSarvamASRStreamConstruction:
    def test_constructs_with_defaults(self):
        from tools.sarvam_asr import SarvamASRStream

        s = SarvamASRStream(api_key="test-key")
        assert s.language_code == "en-IN"
        assert s.sample_rate == 16000

    def test_accepts_overrides(self):
        from tools.sarvam_asr import SarvamASRStream

        s = SarvamASRStream(api_key="k", language_code="hi-IN", sample_rate=8000)
        assert s.language_code == "hi-IN"
        assert s.sample_rate == 8000


class TestSarvamASRLazyConnect:
    """The SDK connection should not open until the first push."""

    def test_no_connect_at_construction(self, monkeypatch):
        from tools import sarvam_asr

        client_calls = MagicMock()

        def fake_ctor(*args, **kwargs):
            client_calls(*args, **kwargs)
            return FakeAsyncSarvamAI(
                api_subscription_key=kwargs["api_subscription_key"],
                socket=FakeSocket([]),
            )

        monkeypatch.setattr(sarvam_asr, "AsyncSarvamAI", fake_ctor)
        sarvam_asr.SarvamASRStream(api_key="k")
        client_calls.assert_not_called()

    @pytest.mark.asyncio
    async def test_connects_on_first_push(self, monkeypatch):
        from tools import sarvam_asr

        socket = FakeSocket(scripted_events=[])
        fake_client = FakeAsyncSarvamAI(api_subscription_key="k", socket=socket)
        monkeypatch.setattr(
            sarvam_asr, "AsyncSarvamAI", lambda **kw: fake_client
        )

        s = sarvam_asr.SarvamASRStream(api_key="k", language_code="en-IN", sample_rate=16000)
        await s.push_pcm(b"\x00\x00" * 160)

        # Give the background pump a tick to drain the queue.
        for _ in range(20):
            if socket.transcribe_calls:
                break
            await asyncio.sleep(0.01)

        assert fake_client.speech_to_text_streaming.connect_kwargs is not None
        kwargs = fake_client.speech_to_text_streaming.connect_kwargs
        assert kwargs.get("model") == "saaras:v3"
        assert kwargs.get("language_code") == "en-IN"
        assert kwargs.get("input_audio_codec") == "pcm_s16le"
        # sample_rate is passed as a string per the SDK signature.
        assert str(kwargs.get("sample_rate")) == "16000"

        await s.close()


class TestSarvamASRPushAndEvents:
    @pytest.mark.asyncio
    async def test_push_pcm_forwards_base64(self, monkeypatch):
        from tools import sarvam_asr

        socket = FakeSocket(scripted_events=[])
        fake_client = FakeAsyncSarvamAI(api_subscription_key="k", socket=socket)
        monkeypatch.setattr(
            sarvam_asr, "AsyncSarvamAI", lambda **kw: fake_client
        )

        s = sarvam_asr.SarvamASRStream(api_key="k")
        frame = b"\x01\x02\x03\x04"
        await s.push_pcm(frame)

        # Drain
        for _ in range(20):
            if socket.transcribe_calls:
                break
            await asyncio.sleep(0.01)

        assert len(socket.transcribe_calls) == 1
        call = socket.transcribe_calls[0]
        # Audio is base64-encoded on the wire.
        assert call["audio"] == base64.b64encode(frame).decode("ascii")

        await s.close()

    @pytest.mark.asyncio
    async def test_events_normalize_transcripts_and_vad(self, monkeypatch):
        from tools import sarvam_asr

        scripted = [
            _make_response(
                "events",
                event_type="speech_start",
                signal_type="START_SPEECH",
            ),
            _make_response(
                "data",
                transcript="hello world",
                language_code="en-IN",
            ),
            _make_response(
                "events",
                event_type="speech_end",
                signal_type="END_SPEECH",
            ),
        ]
        socket = FakeSocket(scripted_events=scripted)
        fake_client = FakeAsyncSarvamAI(api_subscription_key="k", socket=socket)
        monkeypatch.setattr(
            sarvam_asr, "AsyncSarvamAI", lambda **kw: fake_client
        )

        s = sarvam_asr.SarvamASRStream(api_key="k")
        await s.push_pcm(b"\x00\x00")

        collected: list[dict] = []

        async def collect():
            async for ev in s.events():
                collected.append(ev)
                if ev["type"] == "final":
                    break

        await asyncio.wait_for(collect(), timeout=2.0)

        # speech_start should arrive normalized.
        types_seen = [e["type"] for e in collected]
        assert "speech_start" in types_seen
        finals = [e for e in collected if e["type"] == "final"]
        assert finals and finals[0]["text"] == "hello world"

        await s.close()


class TestSarvamASRClose:
    @pytest.mark.asyncio
    async def test_close_releases_sdk_context(self, monkeypatch):
        from tools import sarvam_asr

        socket = FakeSocket(scripted_events=[])
        fake_client = FakeAsyncSarvamAI(api_subscription_key="k", socket=socket)
        monkeypatch.setattr(
            sarvam_asr, "AsyncSarvamAI", lambda **kw: fake_client
        )

        s = sarvam_asr.SarvamASRStream(api_key="k")
        await s.push_pcm(b"\x00\x00")
        # Ensure the pump has connected before closing
        for _ in range(20):
            if fake_client.speech_to_text_streaming.connect_kwargs is not None:
                break
            await asyncio.sleep(0.01)

        await s.close()
        # After close, further pushes should be rejected (the wrapper is
        # one-shot — caller would build a new stream for a new session).
        with pytest.raises(RuntimeError):
            await s.push_pcm(b"\x00\x00")


def test_close_tolerates_cross_loop_call(monkeypatch):
    """Regression: when the pump task was created on one event loop
    (LiveKit job runner) and ``close()`` is called on a different loop
    (gateway shutdown's main loop), the original implementation raised
    ``RuntimeError: Task attached to a different loop``.

    Reproduces by creating the pump in one ``asyncio.run`` and calling
    ``close()`` in a separate ``asyncio.run``. With the fix, close()
    detects the cross-loop scenario, cancels the pump (cancel is
    loop-safe), and returns without raising.
    """
    from tools import sarvam_asr

    socket = FakeSocket(scripted_events=[
        {"type": "data", "data": {"transcript": "hello"}},
    ])
    fake_client = FakeAsyncSarvamAI(api_subscription_key="k", socket=socket)
    monkeypatch.setattr(sarvam_asr, "AsyncSarvamAI", lambda **kw: fake_client)

    holder: dict = {}

    # Loop A — create the stream and start its pump task.
    async def _create():
        s = sarvam_asr.SarvamASRStream(api_key="k")
        await s.push_pcm(b"\x00" * 320)  # triggers _start() and pump
        # Wait for the pump to enter `async with`.
        for _ in range(20):
            if fake_client.speech_to_text_streaming.connect_kwargs is not None:
                break
            await asyncio.sleep(0.01)
        holder["stream"] = s

    asyncio.run(_create())

    # Loop B — call close() on a fresh event loop. Without the fix this
    # would raise RuntimeError("Task attached to a different loop").
    async def _close():
        await holder["stream"].close()

    # The whole point: this should NOT raise.
    asyncio.run(_close())

    # Pump task should be cancelled or terminal — not orphaned.
    pump = holder["stream"]._pump_task
    assert pump is None or pump.done() or pump.cancelled(), (
        "pump task should be terminal after cross-loop close()"
    )
