"""Streaming Sarvam Bulbul v3 TTS client wrapper.

Thin async wrapper around
``sarvamai.AsyncSarvamAI.text_to_speech.convert_stream``. The SDK already
returns an ``AsyncIterator[bytes]`` so this module's job is small:

- own one ``AsyncSarvamAI`` instance per stream object (so multiple
  ``synth`` calls on the same stream share one underlying client);
- bake in the v2v voice agent's defaults (``shubh`` voice, ``bulbul:v3``,
  PCM ``s16le`` at 16 kHz) while still allowing overrides;
- expose a single ``synth(text)`` async-generator entry point that
  downstream layers (chunker, LiveKit audio source) can drive.

PCM ``s16le`` maps to the SDK's ``linear16`` codec value. We do not
transcode here — callers that need a different framing (e.g. WAV-with-
header for browser playback) layer that on top.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

from sarvamai import AsyncSarvamAI


logger = logging.getLogger(__name__)


class SarvamTTSStream:
    """A reusable handle over Sarvam's streaming TTS endpoint.

    Usage::

        tts = SarvamTTSStream(api_key=..., voice="shubh")
        async for chunk in tts.synth("Hello world."):
            ...   # chunk is raw PCM s16le bytes
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "shubh",
        sample_rate: int = 16000,
        target_language_code: str = "en-IN",
        model: str = "bulbul:v3",
        output_audio_codec: str = "linear16",  # SDK name for PCM s16le
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self.voice = voice
        self.sample_rate = sample_rate
        self.target_language_code = target_language_code
        self._model = model
        self._codec = output_audio_codec

        self._client: Optional[Any] = None
        self._closed = False

    # ------------------------------------------------------------------ public

    async def synth(self, text: str) -> AsyncIterator[bytes]:
        """Stream a single utterance.

        Yields raw audio chunks in the codec configured at construction
        time (``linear16`` / PCM s16le by default).
        """
        if self._closed:
            raise RuntimeError("SarvamTTSStream is closed")
        client = self._ensure_client()
        stream = client.text_to_speech.convert_stream(
            text=text,
            target_language_code=self.target_language_code,
            speaker=self.voice,
            model=self._model,
            output_audio_codec=self._codec,
            speech_sample_rate=self.sample_rate,
        )
        async for chunk in stream:
            yield chunk

    async def close(self) -> None:
        """Drop the cached client. Idempotent.

        The Sarvam SDK uses an HTTP client under the hood; explicit close is
        not required for correctness, but releases the underlying httpx
        connection pool promptly.
        """
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is None:
            return
        # Best-effort close — older SDK builds may not expose either method.
        for closer in ("aclose", "close"):
            fn = getattr(client, closer, None)
            if fn is None:
                continue
            try:
                result = fn()
                if hasattr(result, "__await__"):
                    await result
                return
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("Sarvam TTS client close raised: %r", exc)
                return

    # ----------------------------------------------------------------- internals

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = AsyncSarvamAI(api_subscription_key=self._api_key)
        return self._client


__all__ = ["SarvamTTSStream"]
