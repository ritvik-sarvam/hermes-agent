"""Hermes platform adapter that bridges a LiveKit room to a Hermes session.

Audio-in path (Task 5.2):
    LiveKit participant audio track  →  20ms PCM s16le @ 16 kHz
        ├─ Silero VAD  → drives TurnState (barge-in is consumed by 5.3)
        └─ SarvamASRStream  → ``{"type": "final", "text": ...}``
                              → MessageEvent → BasePlatformAdapter.handle_message

Audio-out path (Task 5.3) lives below the marked TODO and is a no-op stub here.
The skeleton ``send()`` returns success so the gateway runner's text-send code
paths don't trip on this platform — assistant audio is published as a LiveKit
audio track in 5.3.

Room-name convention: ``v2v-<user_id>-<call_id>``. Rooms not matching the
prefix are silently ignored so the same LiveKit deployment can host other
agents without us crashing on their job dispatches.

Tests in ``tests/gateway/test_voice_rtc_*.py`` mock at the LiveKit and
Sarvam boundaries — there is no live network call.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)


# Default audio contract for the audio-in path. Sarvam Saaras v3 expects
# 16 kHz mono PCM s16le; LiveKit gives us native 48 kHz frames so the
# audio-track reader is responsible for resampling — we keep this constant
# here so 5.2's resampler and 5.3's TTS use the same number.
_ASR_SAMPLE_RATE = 16000
_ASR_LANGUAGE_CODE = "en-IN"


def check_voice_rtc_requirements() -> bool:
    """Return True if every livekit module the adapter touches is importable."""
    try:
        import livekit  # noqa: F401
        import livekit.agents  # noqa: F401
        import livekit.api  # noqa: F401
        import livekit.rtc  # noqa: F401
        return True
    except ImportError:
        return False


def _parse_room_name(name: str) -> Tuple[str, str]:
    """Parse a room name of the form ``v2v-<user_id>-<call_id>``.

    Returns ``(user_id, call_id)``. Raises ``ValueError`` for malformed
    names so callers (the LiveKit job entrypoint) can ``except ValueError``
    and silently skip rooms that aren't ours.
    """
    if not isinstance(name, str) or not name.startswith("v2v-"):
        raise ValueError(f"room name {name!r} missing v2v- prefix")
    rest = name[len("v2v-"):]
    # Split into exactly two parts on the *first* dash. user_ids may
    # contain underscores or letters; call_ids are typically uuid-shaped
    # so we don't try to validate them beyond non-empty.
    if "-" not in rest:
        raise ValueError(f"room name {name!r} missing call_id segment")
    user_id, call_id = rest.split("-", 1)
    if not user_id or not call_id:
        raise ValueError(f"room name {name!r} has empty user_id or call_id")
    return user_id, call_id


class VoiceRTCAdapter(BasePlatformAdapter):
    """Streaming voice agent adapter — bridges a LiveKit room to a Hermes
    session over Sarvam ASR (in) and TTS (out, in 5.3)."""

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform.VOICE_RTC)
        extra = config.extra or {}
        self._lk_url: str = extra.get("url") or os.getenv("LIVEKIT_URL", "")
        self._lk_api_key: str = extra.get("api_key") or os.getenv("LIVEKIT_API_KEY", "")
        self._lk_api_secret: str = extra.get("api_secret") or os.getenv("LIVEKIT_API_SECRET", "")
        # Sarvam credential read at construct time but the actual ASR session
        # is opened per-call (so a key rotation hot-reload is straightforward).
        self._sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")

        # Per-call bookkeeping so disconnect() can tear down ASR sessions
        # cleanly. Keys are room names (v2v-<user_id>-<call_id>).
        self._active_calls: Dict[str, Dict[str, Any]] = {}

        # The background task running the LiveKit AgentServer / Worker. Set
        # by ``_start_worker`` and cancelled in ``disconnect``.
        self._worker_task: Optional[asyncio.Task] = None
        self._worker: Any = None  # populated by _start_worker

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Spin up the LiveKit Agents worker in the background.

        Returns True even if the worker hasn't yet finished its initial
        handshake — the gateway runner expects ``connect()`` to be
        non-blocking so it can launch multiple platforms concurrently.
        """
        try:
            self._start_worker()
        except Exception as exc:  # pragma: no cover — surfaces as fatal
            logger.exception("voice_rtc: worker startup failed: %s", exc)
            self._set_fatal_error("voice_rtc_worker", str(exc), retryable=False)
            return False

        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Cancel the worker task and close any per-call ASR streams."""
        # Tear down ASR streams first so finals in flight don't hit a dead
        # adapter.
        for room_name, state in list(self._active_calls.items()):
            asr = state.get("asr")
            if asr is not None:
                try:
                    await asr.close()
                except Exception:  # pragma: no cover — best-effort
                    logger.warning("voice_rtc: asr.close() raised for %s", room_name, exc_info=True)
            for task_name in ("asr_consumer_task", "vad_task", "audio_task"):
                t = state.get(task_name)
                if t is not None and not t.done():
                    t.cancel()
        self._active_calls.clear()

        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass
        self._worker_task = None
        self._worker = None

        self._mark_disconnected()

    # ------------------------------------------------------------------
    # Worker bootstrapping (testable seam)
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        """Construct a livekit.agents worker bound to ``_entrypoint``.

        Tests substitute this method (or replace ``_build_worker_options``
        / ``_run_worker``) so no real LiveKit handshake is performed.
        """
        # Use lazy imports so module import doesn't pull livekit in.
        from livekit import agents as lk_agents  # type: ignore

        options = lk_agents.WorkerOptions(
            entrypoint_fnc=self._entrypoint,
            ws_url=self._lk_url or None,
            api_key=self._lk_api_key or None,
            api_secret=self._lk_api_secret or None,
        )
        self._worker = self._build_worker(options)
        self._worker_task = asyncio.create_task(self._run_worker(self._worker))

    def _build_worker(self, options: Any) -> Any:
        """Construct an AgentServer from WorkerOptions (livekit-agents>=1.0).

        Wrapped in a method so tests can replace it without touching the
        private API of livekit-agents (which has changed shape across
        versions: ``Worker``  →  ``AgentServer.from_server_options``).
        """
        from livekit.agents.worker import AgentServer  # type: ignore
        return AgentServer.from_server_options(options)

    async def _run_worker(self, worker: Any) -> None:
        """Run the worker until cancelled. Background-task body."""
        try:
            await worker.run()
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover — log and exit
            logger.exception("voice_rtc: worker exited with an error")
        finally:
            try:
                await worker.aclose()
            except Exception:  # pragma: no cover — best-effort
                pass

    # ------------------------------------------------------------------
    # LiveKit job entrypoint
    # ------------------------------------------------------------------

    async def _entrypoint(self, ctx: Any) -> None:
        """Invoked once per dispatched job. ``ctx`` is a JobContext."""
        room = getattr(ctx, "room", None)
        room_name = getattr(room, "name", None) if room is not None else None
        if not room_name:
            logger.warning("voice_rtc: entrypoint received ctx without a room name")
            return
        try:
            user_id, call_id = _parse_room_name(room_name)
        except ValueError:
            # Not our room — quietly bow out so other platform agents on the
            # same LiveKit deployment can dispatch their own jobs.
            logger.debug("voice_rtc: skipping non-v2v room %s", room_name)
            return
        await self._on_room(ctx, user_id, call_id)

    async def _on_room(self, ctx: Any, user_id: str, call_id: str) -> None:
        """Per-call orchestration: open ASR, attach audio reader + VAD,
        forward finals to ``handle_message``.

        Tests drive this directly with a hand-built fake ``ctx``; the
        production path goes through ``_entrypoint`` after parsing the
        room name.
        """
        room_name = f"v2v-{user_id}-{call_id}"
        logger.info("voice_rtc: starting call %s", room_name)

        from tools.sarvam_asr import SarvamASRStream
        from tools.voice_rtc.state import TurnState

        asr = SarvamASRStream(
            api_key=self._sarvam_api_key,
            sample_rate=_ASR_SAMPLE_RATE,
            language_code=_ASR_LANGUAGE_CODE,
        )
        turn_state = TurnState()

        state: Dict[str, Any] = {
            "asr": asr,
            "turn_state": turn_state,
            "user_id": user_id,
            "call_id": call_id,
        }
        self._active_calls[room_name] = state

        # Spawn the ASR-event consumer.  It runs concurrently with the
        # audio reader; both terminate when the room ends or the adapter
        # disconnects.
        async def _asr_consumer() -> None:
            try:
                async for ev in asr.events():
                    if ev.get("type") == "final":
                        await self._deliver_final(ev.get("text", ""), user_id, room_name)
                    elif ev.get("type") == "error":
                        logger.warning("voice_rtc: asr error for %s: %s", room_name, ev)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover
                logger.exception("voice_rtc: asr consumer crashed for %s", room_name)

        state["asr_consumer_task"] = asyncio.create_task(_asr_consumer())

        # Audio-track reader runs in its own task as well — separating the
        # producer and consumer so a slow ASR doesn't backpressure RTC frames.
        state["audio_task"] = asyncio.create_task(
            self._read_room_audio(ctx, state)
        )

    async def _deliver_final(self, text: str, user_id: str, room_name: str) -> None:
        """Build a MessageEvent for an ASR final and route it through
        BasePlatformAdapter.handle_message."""
        text = (text or "").strip()
        if not text:
            return
        source = self.build_source(
            chat_id=room_name,
            chat_name=room_name,
            chat_type="dm",
            user_id=user_id,
            user_name=user_id,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=None,
        )
        # Drive the FSM: a user final transitions LISTENING → THINKING.
        # 5.3 will use this state when the LLM token stream begins.
        try:
            from tools.voice_rtc.state import Event as TurnEvent
            ts = self._active_calls.get(room_name, {}).get("turn_state")
            if ts is not None:
                ts.handle(TurnEvent.USER_FINAL)
        except Exception:  # pragma: no cover
            logger.debug("voice_rtc: turn-state advance failed", exc_info=True)
        await self.handle_message(event)

    async def _read_room_audio(self, ctx: Any, state: Dict[str, Any]) -> None:
        """Subscribe to participant audio, demux 20ms s16le frames,
        push them to ASR and the VAD instance.

        This is the LiveKit-touching surface. Implementation kept thin and
        defensive — most of the real logic lives in tested helpers
        (``_handle_audio_frame``). Tests replace the room/track iteration
        machinery so we can assert the per-frame fanout deterministically.
        """
        room = getattr(ctx, "room", None)
        if room is None:
            logger.warning("voice_rtc: ctx has no room; skipping audio read")
            return

        # Open the Silero VAD lazily; tests inject their own via state.
        if "vad" not in state:
            try:
                from livekit.agents.vad import silero  # type: ignore
                state["vad"] = silero.VAD.load()
            except Exception:
                # Without Silero the audio-in path still works (ASR-only
                # finals); barge-in is just disabled.
                state["vad"] = None

        try:
            async for frame in self._iter_audio_frames(ctx):
                await self._handle_audio_frame(frame, state)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: audio reader crashed")

    async def _iter_audio_frames(self, ctx: Any):
        """Async iterator over ``bytes`` PCM frames for the participant.

        Production version uses ``livekit.rtc`` track subscription; tests
        override this method to yield synthetic frames.

        Default no-op: yield nothing, so a missing override during tests
        does not hang.
        """
        # The real implementation will use ``rtc.AudioStream`` over the
        # participant's first audio publication and resample to 16 kHz
        # mono s16le. That belongs in 5.3 alongside the AudioSource we
        # publish for TTS — kept stubbed here so 5.2's tests can drive the
        # frame-handling path directly via ``_handle_audio_frame``.
        if False:  # pragma: no cover
            yield b""
        return

    async def _handle_audio_frame(self, frame: bytes, state: Dict[str, Any]) -> None:
        """Push a single 20ms PCM frame to ASR and the VAD."""
        asr = state.get("asr")
        if asr is not None:
            try:
                await asr.push_pcm(frame)
            except Exception:  # pragma: no cover
                logger.exception("voice_rtc: asr.push_pcm failed")

        vad = state.get("vad")
        if vad is not None:
            try:
                push = getattr(vad, "push_frame", None)
                if push is None:
                    push = getattr(vad, "feed", None)
                if push is not None:
                    res = push(frame)
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:  # pragma: no cover
                logger.exception("voice_rtc: vad push_frame failed")

    # ------------------------------------------------------------------
    # Outbound surface
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """No text-send surface for voice_rtc. The assistant's reply is
        streamed back as a LiveKit audio track in 5.3."""
        # TODO(5.3): hand ``content`` to the chunker → TTS → AudioSource pipeline.
        return SendResult(success=True, message_id="voice")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}
