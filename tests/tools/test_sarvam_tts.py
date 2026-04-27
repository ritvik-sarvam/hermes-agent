"""Tests for the streaming Sarvam Bulbul TTS client wrapper.

The wrapper at ``tools.sarvam_tts.SarvamTTSStream`` is a thin facade over
``sarvamai.AsyncSarvamAI.text_to_speech.convert_stream``. It owns one
``AsyncSarvamAI`` instance per stream object (so multiple ``synth`` calls
reuse the same client) and exposes a single async iterator yielding raw
``bytes`` chunks.

These tests mock the SDK boundary — no live network calls.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fakes for the Sarvam SDK boundary
# ---------------------------------------------------------------------------


class FakeTextToSpeech:
    """Stand-in for ``client.text_to_speech``."""

    def __init__(self, scripted_chunks: list[bytes]):
        self._scripted = list(scripted_chunks)
        self.calls: list[dict] = []

    def convert_stream(self, **kwargs):
        self.calls.append(kwargs)
        scripted = list(self._scripted)

        async def _aiter():
            for chunk in scripted:
                await asyncio.sleep(0)
                yield chunk

        return _aiter()


class FakeAsyncSarvamAI:
    def __init__(self, *, api_subscription_key: str, scripted_chunks: list[bytes]):
        self.api_subscription_key = api_subscription_key
        self.text_to_speech = FakeTextToSpeech(scripted_chunks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSarvamTTSStreamConstruction:
    def test_constructs_with_defaults(self):
        from tools.sarvam_tts import SarvamTTSStream

        s = SarvamTTSStream(api_key="test-key")
        assert s.voice == "shubh"
        assert s.sample_rate == 16000

    def test_accepts_overrides(self):
        from tools.sarvam_tts import SarvamTTSStream

        s = SarvamTTSStream(
            api_key="k",
            voice="priya",
            sample_rate=22050,
            target_language_code="hi-IN",
        )
        assert s.voice == "priya"
        assert s.sample_rate == 22050
        assert s.target_language_code == "hi-IN"

    def test_requires_api_key(self):
        from tools.sarvam_tts import SarvamTTSStream

        with pytest.raises(ValueError):
            SarvamTTSStream(api_key="")


class TestSarvamTTSSynth:
    @pytest.mark.asyncio
    async def test_synth_yields_chunks(self, monkeypatch):
        from tools import sarvam_tts

        chunks = [b"\x00\x01\x02", b"\x03\x04\x05", b"\x06"]
        fake_client = FakeAsyncSarvamAI(
            api_subscription_key="k", scripted_chunks=chunks
        )
        monkeypatch.setattr(
            sarvam_tts, "AsyncSarvamAI", lambda **kw: fake_client
        )

        s = sarvam_tts.SarvamTTSStream(api_key="k")
        collected = []
        async for c in s.synth("Hello world."):
            collected.append(c)

        assert collected == chunks

    @pytest.mark.asyncio
    async def test_synth_passes_voice_and_sample_rate(self, monkeypatch):
        from tools import sarvam_tts

        fake_client = FakeAsyncSarvamAI(
            api_subscription_key="k", scripted_chunks=[b"x"]
        )
        monkeypatch.setattr(
            sarvam_tts, "AsyncSarvamAI", lambda **kw: fake_client
        )

        s = sarvam_tts.SarvamTTSStream(
            api_key="k",
            voice="kabir",
            sample_rate=22050,
            target_language_code="hi-IN",
        )
        async for _ in s.synth("Test."):
            pass

        assert len(fake_client.text_to_speech.calls) == 1
        call = fake_client.text_to_speech.calls[0]
        assert call["text"] == "Test."
        assert call["speaker"] == "kabir"
        assert call["speech_sample_rate"] == 22050
        assert call["target_language_code"] == "hi-IN"
        assert call["model"] == "bulbul:v3"
        # PCM s16le maps to "linear16" in the SDK's codec enum.
        assert call["output_audio_codec"] == "linear16"

    @pytest.mark.asyncio
    async def test_multiple_synth_calls_reuse_client(self, monkeypatch):
        from tools import sarvam_tts

        ctor_calls = MagicMock()

        def fake_ctor(**kwargs):
            ctor_calls(**kwargs)
            return FakeAsyncSarvamAI(
                api_subscription_key=kwargs["api_subscription_key"],
                scripted_chunks=[b"a"],
            )

        monkeypatch.setattr(sarvam_tts, "AsyncSarvamAI", fake_ctor)

        s = sarvam_tts.SarvamTTSStream(api_key="k")
        async for _ in s.synth("one"):
            pass
        async for _ in s.synth("two"):
            pass
        async for _ in s.synth("three"):
            pass

        assert ctor_calls.call_count == 1


class TestSarvamTTSClose:
    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, monkeypatch):
        from tools import sarvam_tts

        fake_client = FakeAsyncSarvamAI(
            api_subscription_key="k", scripted_chunks=[b"x"]
        )
        monkeypatch.setattr(
            sarvam_tts, "AsyncSarvamAI", lambda **kw: fake_client
        )

        s = sarvam_tts.SarvamTTSStream(api_key="k")
        await s.close()
        await s.close()  # second call must not raise
