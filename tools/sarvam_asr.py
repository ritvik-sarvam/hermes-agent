"""Streaming Sarvam Saaras v3 ASR client wrapper.

Thin async wrapper around ``sarvamai.AsyncSarvamAI.speech_to_text_streaming``.

Why a wrapper at all? Two reasons:

1. The SDK's session is an async context manager whose lifetime spans an
   ``async with`` block. Callers driving real-time audio need a queue-based
   push interface that doesn't tie them to a context manager — so we run a
   small background pump task that holds the SDK session open and forwards
   queued frames into ``socket.transcribe(...)``.
2. The SDK's response objects (``SpeechToTextStreamingResponse``) carry a
   union of transcription / events / error data. Downstream code wants a
   normalized event shape (``{"type": "final" | "speech_start" | ...,
   "text": str}``) so we don't propagate Sarvam's pydantic types throughout
   the codebase.

Audio contract: PCM ``s16le`` at the configured ``sample_rate`` (16 kHz by
default). Frames are base64-encoded for the SDK's ``transcribe()`` call.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, AsyncIterator, Optional

from sarvamai import AsyncSarvamAI


logger = logging.getLogger(__name__)


# Sentinel pushed into the frame queue to tell the pump it's time to exit.
_CLOSE_SENTINEL: object = object()


class SarvamASRStream:
    """Streaming Saaras v3 ASR session with a push-based audio interface.

    Lifecycle::

        s = SarvamASRStream(api_key=..., language_code="en-IN")
        async for ev in s.events():       # consumer task
            ...
        await s.push_pcm(frame_bytes)     # producer task
        ...
        await s.close()

    The SDK connection is opened lazily on the first ``push_pcm`` call so
    cheap construction stays cheap.
    """

    def __init__(
        self,
        *,
        api_key: str,
        language_code: str = "en-IN",
        sample_rate: int = 16000,
        model: str = "saaras:v3",
        mode: str = "transcribe",
        high_vad_sensitivity: bool = True,
        vad_signals: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self.language_code = language_code
        self.sample_rate = sample_rate
        self._model = model
        self._mode = mode
        self._high_vad = high_vad_sensitivity
        self._vad_signals = vad_signals

        self._client: Optional[Any] = None
        self._frame_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._pump_task: Optional[asyncio.Task] = None
        self._started = False
        self._closed = False

    # ------------------------------------------------------------------ public

    async def push_pcm(self, frame: bytes) -> None:
        """Queue a PCM ``s16le`` frame for transcription.

        Lazily opens the SDK session on the first call.
        """
        if self._closed:
            raise RuntimeError("SarvamASRStream is closed")
        if not self._started:
            self._start()
        await self._frame_queue.put(frame)

    async def events(self) -> AsyncIterator[dict]:
        """Yield normalized events from the upstream session.

        Event shapes::

            {"type": "speech_start"}
            {"type": "speech_end"}
            {"type": "final",   "text": "..."}
            {"type": "interim", "text": "..."}   # if the SDK ever emits one
            {"type": "error",   "text": "..."}

        Iteration ends when the underlying session closes (or ``close()`` is
        called).
        """
        if not self._started:
            # No frames pushed yet — open the session so consumers can attach
            # before the first audio frame arrives.
            self._start()

        while True:
            ev = await self._event_queue.get()
            if ev is _CLOSE_SENTINEL:  # type: ignore[comparison-overlap]
                return
            yield ev

    async def close(self) -> None:
        """Stop the pump and release the SDK session."""
        if self._closed:
            return
        self._closed = True
        if self._started:
            await self._frame_queue.put(_CLOSE_SENTINEL)
        if self._pump_task is not None:
            try:
                await asyncio.wait_for(self._pump_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._pump_task.cancel()
                try:
                    await self._pump_task
                except (asyncio.CancelledError, Exception):
                    pass
        # Wake any pending events() consumer so it sees the close.
        await self._event_queue.put(_CLOSE_SENTINEL)  # type: ignore[arg-type]

    # ----------------------------------------------------------------- internals

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        self._client = AsyncSarvamAI(api_subscription_key=self._api_key)
        self._pump_task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        """Hold the SDK session open; forward frames in, forward events out."""
        assert self._client is not None
        try:
            async with self._client.speech_to_text_streaming.connect(
                model=self._model,
                mode=self._mode,
                language_code=self.language_code,
                sample_rate=str(self.sample_rate),
                input_audio_codec="pcm_s16le",
                high_vad_sensitivity="true" if self._high_vad else "false",
                vad_signals="true" if self._vad_signals else "false",
            ) as socket:
                sender = asyncio.create_task(self._send_loop(socket))
                receiver = asyncio.create_task(self._recv_loop(socket))
                done, pending = await asyncio.wait(
                    {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                # Surface any exception from the completed task(s).
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        logger.warning("Sarvam ASR loop ended with error: %r", exc)
                        await self._event_queue.put(
                            {"type": "error", "text": str(exc)}
                        )
        except Exception as exc:  # connect failed
            logger.warning("Sarvam ASR connect failed: %r", exc)
            await self._event_queue.put({"type": "error", "text": str(exc)})
        finally:
            await self._event_queue.put(_CLOSE_SENTINEL)  # type: ignore[arg-type]

    async def _send_loop(self, socket: Any) -> None:
        while True:
            item = await self._frame_queue.get()
            if item is _CLOSE_SENTINEL:
                # Best-effort flush so any buffered audio gets a final transcript.
                try:
                    await socket.flush()
                except Exception:
                    pass
                return
            audio_b64 = base64.b64encode(item).decode("ascii")
            await socket.transcribe(
                audio=audio_b64,
                encoding="audio/wav",
                sample_rate=self.sample_rate,
            )

    async def _recv_loop(self, socket: Any) -> None:
        async for response in socket:
            normalized = self._normalize(response)
            if normalized is not None:
                await self._event_queue.put(normalized)

    @staticmethod
    def _normalize(response: Any) -> Optional[dict]:
        """Translate an SDK response into our normalized event shape.

        Sarvam's streaming response is ``{type, data}`` where ``type`` is one
        of ``data`` (transcription), ``events`` (VAD signals), or ``error``.
        """
        rtype = getattr(response, "type", None)
        data = getattr(response, "data", None)
        if rtype == "data":
            text = getattr(data, "transcript", "") or ""
            # Saaras v3 streaming only emits utterance-level finals — there is
            # no separate interim stream from the server. We tag everything
            # from the ``data`` channel as "final" and let downstream layers
            # handle smoothing.
            return {"type": "final", "text": text}
        if rtype == "events":
            sig = (
                getattr(data, "signal_type", None)
                or getattr(data, "event_type", None)
                or ""
            )
            sig_lc = str(sig).lower()
            if "start" in sig_lc:
                return {"type": "speech_start"}
            if "end" in sig_lc:
                return {"type": "speech_end"}
            return {"type": "event", "text": str(sig)}
        if rtype == "error":
            text = getattr(data, "message", None) or getattr(data, "text", "") or ""
            return {"type": "error", "text": str(text)}
        return None


__all__ = ["SarvamASRStream"]
