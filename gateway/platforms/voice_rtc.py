"""Hermes platform adapter that bridges a LiveKit room to a Hermes session.

Audio-in path (Task 5.2 + 5.3 track subscription):
    LiveKit participant audio track  →  AudioStream(sample_rate=16000)
        ├─ Silero VAD  → drives TurnState (barge-in cancels in-flight TTS)
        └─ SarvamASRStream  → ``{"type": "final", "text": ...}``
                              → MessageEvent → BasePlatformAdapter.handle_message

Audio-out path (Task 5.3):
    LLM token stream
        →  ClauseChunker.feed/flush()
            →  SarvamTTSStream.synth(chunk)  →  PCM s16le @ 16 kHz
                →  livekit.rtc.AudioSource.capture_frame(AudioFrame)

    Barge-in: a VAD ``speech_start`` event while the FSM is THINKING or
    SPEAKING fires ``TurnState.handle(VAD_SPEECH_START)``, the audio-out
    coroutine is cancelled, the AudioSource queue is cleared, and the
    FSM advances through INTERRUPTED → CANCEL_DONE back to LISTENING.

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
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)


# Default audio contract for both paths. Sarvam Saaras v3 expects 16 kHz
# mono PCM s16le; the TTS streamer emits the same format. LiveKit gives
# us native 48 kHz frames in the inbound direction so the audio reader
# asks AudioStream to resample down.
_ASR_SAMPLE_RATE = 16000
_ASR_LANGUAGE_CODE = "en-IN"

# 20 ms of 16 kHz mono s16le = 320 samples = 640 bytes. We slice the
# Sarvam SDK's chunk output into frames of this size before pushing them
# at the AudioSource.
_TTS_FRAME_SAMPLES = 320
_TTS_FRAME_BYTES = _TTS_FRAME_SAMPLES * 2


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
    if "-" not in rest:
        raise ValueError(f"room name {name!r} missing call_id segment")
    user_id, call_id = rest.split("-", 1)
    if not user_id or not call_id:
        raise ValueError(f"room name {name!r} has empty user_id or call_id")
    return user_id, call_id


class VoiceRTCAdapter(BasePlatformAdapter):
    """Streaming voice agent adapter — bridges a LiveKit room to a Hermes
    session over Sarvam ASR (in) and TTS (out)."""

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform.VOICE_RTC)
        extra = config.extra or {}
        self._lk_url: str = extra.get("url") or os.getenv("LIVEKIT_URL", "")
        self._lk_api_key: str = extra.get("api_key") or os.getenv("LIVEKIT_API_KEY", "")
        self._lk_api_secret: str = extra.get("api_secret") or os.getenv("LIVEKIT_API_SECRET", "")
        self._sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")

        # Per-call bookkeeping. Keys are room names (v2v-<user_id>-<call_id>).
        # Values include: asr, turn_state, audio_source, tts_stream, tts_task,
        # asr_consumer_task, vad_task, audio_task, user_id, call_id.
        self._active_calls: Dict[str, Dict[str, Any]] = {}

        # Background task running the LiveKit Agents worker.
        self._worker_task: Optional[asyncio.Task] = None
        self._worker: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        try:
            self._start_worker()
        except Exception as exc:  # pragma: no cover — surfaces as fatal
            logger.exception("voice_rtc: worker startup failed: %s", exc)
            self._set_fatal_error("voice_rtc_worker", str(exc), retryable=False)
            return False
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        for room_name, state in list(self._active_calls.items()):
            # Cancel any in-flight TTS task first so it doesn't try to
            # capture into a torn-down AudioSource.
            tts_task = state.get("tts_task")
            if tts_task is not None and not tts_task.done():
                tts_task.cancel()
            asr = state.get("asr")
            if asr is not None:
                try:
                    await asr.close()
                except Exception:  # pragma: no cover
                    logger.warning("voice_rtc: asr.close() raised for %s", room_name, exc_info=True)
            tts = state.get("tts_stream")
            if tts is not None:
                try:
                    await tts.close()
                except Exception:  # pragma: no cover
                    logger.warning("voice_rtc: tts.close() raised for %s", room_name, exc_info=True)
            src = state.get("audio_source")
            if src is not None:
                try:
                    aclose = getattr(src, "aclose", None)
                    if aclose is not None:
                        result = aclose()
                        if asyncio.iscoroutine(result):
                            await result
                except Exception:  # pragma: no cover
                    pass
            for task_name in ("asr_consumer_task", "vad_task", "audio_task", "tts_task"):
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
        from livekit.agents.worker import AgentServer  # type: ignore
        return AgentServer.from_server_options(options)

    async def _run_worker(self, worker: Any) -> None:
        try:
            await worker.run()
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: worker exited with an error")
        finally:
            try:
                await worker.aclose()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    # LiveKit job entrypoint
    # ------------------------------------------------------------------

    async def _entrypoint(self, ctx: Any) -> None:
        room = getattr(ctx, "room", None)
        room_name = getattr(room, "name", None) if room is not None else None
        if not room_name:
            logger.warning("voice_rtc: entrypoint received ctx without a room name")
            return
        try:
            user_id, call_id = _parse_room_name(room_name)
        except ValueError:
            logger.debug("voice_rtc: skipping non-v2v room %s", room_name)
            return
        await self._on_room(ctx, user_id, call_id)

    async def _on_room(self, ctx: Any, user_id: str, call_id: str) -> None:
        """Per-call orchestration: open ASR + TTS + AudioSource, attach
        the audio reader and VAD/ASR fanout, forward finals to
        ``handle_message``.
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

        # Publish an outbound audio track for the agent's TTS audio.
        # Behind a method seam so existing audio-in tests (which pass a
        # vanilla MagicMock ctx.room) can stub this out without tripping
        # over the awaitable publish_track().
        try:
            await self._open_publish_audio(ctx, state)
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: failed to publish audio for %s", room_name)

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

        state["audio_task"] = asyncio.create_task(
            self._read_room_audio(ctx, state)
        )

    async def _open_publish_audio(
        self,
        ctx: Any,
        state: Dict[str, Any],
    ) -> None:
        """Construct an ``AudioSource`` + ``LocalAudioTrack`` and publish
        it on the local participant.

        Method seam: existing audio-in tests use a vanilla MagicMock
        ``ctx.room`` whose ``local_participant.publish_track`` isn't
        awaitable; those tests stub this method out directly.
        """
        room = getattr(ctx, "room", None)
        if room is None:
            return
        local = getattr(room, "local_participant", None)
        if local is None:
            return

        from livekit import rtc as lk_rtc  # type: ignore

        source = lk_rtc.AudioSource(_ASR_SAMPLE_RATE, 1)
        track = lk_rtc.LocalAudioTrack.create_audio_track("agent-voice", source)
        try:
            await local.publish_track(track, lk_rtc.TrackPublishOptions())
        except TypeError:
            # Older SDK builds accepted publish_track(track) without
            # options — fall back to the unary call.
            await local.publish_track(track)
        state["audio_source"] = source
        state["audio_track"] = track

    async def _deliver_final(self, text: str, user_id: str, room_name: str) -> None:
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
        try:
            from tools.voice_rtc.state import Event as TurnEvent
            ts = self._active_calls.get(room_name, {}).get("turn_state")
            if ts is not None:
                ts.handle(TurnEvent.USER_FINAL)
        except Exception:  # pragma: no cover
            logger.debug("voice_rtc: turn-state advance failed", exc_info=True)
        await self.handle_message(event)

    # ------------------------------------------------------------------
    # Audio-in reader
    # ------------------------------------------------------------------

    async def _read_room_audio(self, ctx: Any, state: Dict[str, Any]) -> None:
        """Subscribe to participant audio, demux 20ms s16le frames,
        push them to ASR and the VAD instance."""
        room = getattr(ctx, "room", None)
        if room is None:
            logger.warning("voice_rtc: ctx has no room; skipping audio read")
            return

        if "vad" not in state:
            try:
                # Silero ships under livekit-plugins-silero; fall back to
                # a no-op if it isn't installed (the audio-in path still
                # works without barge-in).
                from livekit.plugins import silero  # type: ignore
                state["vad"] = silero.VAD.load()
            except Exception:
                state["vad"] = None

        # Spawn a VAD-event consumer that forwards speech_start to the
        # barge-in handler. This is what wires VAD output into the FSM.
        vad = state.get("vad")
        if vad is not None:
            state["vad_stream"] = vad.stream()
            room_name = f"v2v-{state.get('user_id')}-{state.get('call_id')}"
            state["vad_task"] = asyncio.create_task(
                self._consume_vad_events(state["vad_stream"], room_name)
            )

        try:
            async for frame in self._iter_audio_frames(ctx):
                await self._handle_audio_frame(frame, state)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: audio reader crashed")

    async def _consume_vad_events(self, vad_stream: Any, room_name: str) -> None:
        """Drain the Silero VAD's event stream; on speech_start, fire
        the barge-in handler."""
        try:
            async for ev in vad_stream:
                ev_type = getattr(ev, "type", None)
                # ``VADEventType.START_OF_SPEECH`` has value 'start_of_speech'.
                # Compare on string value so the test doesn't have to import
                # the enum class.
                value = getattr(ev_type, "value", ev_type)
                if value in ("start_of_speech", "speech_start"):
                    await self._on_vad_speech_start(room_name)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: vad event consumer crashed")

    async def _iter_audio_frames(self, ctx: Any) -> AsyncIterator[bytes]:
        """Async iterator over ``bytes`` PCM frames for the participant.

        Subscribes to incoming audio tracks via ``room.on('track_subscribed')``
        and demuxes their AudioFrames. ``livekit.rtc.AudioStream`` is
        constructed with ``sample_rate=16000, num_channels=1`` so the SDK
        resamples 48 kHz mic input down to our ASR contract.
        """
        room = getattr(ctx, "room", None)
        if room is None:
            return

        from livekit import rtc as lk_rtc  # type: ignore

        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        readers: list[asyncio.Task] = []

        async def _drain_track(track: Any) -> None:
            stream = lk_rtc.AudioStream(
                track,
                sample_rate=_ASR_SAMPLE_RATE,
                num_channels=1,
            )
            try:
                async for ev in stream:
                    frame = getattr(ev, "frame", ev)
                    data = getattr(frame, "data", None)
                    if data is None:
                        continue
                    await queue.put(bytes(data))
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover
                logger.exception("voice_rtc: audio stream reader crashed")
            finally:
                try:
                    await stream.aclose()
                except Exception:  # pragma: no cover
                    pass

        def _on_track_subscribed(track, publication, participant) -> None:
            # Filter to audio kinds.  TrackKind enum values vary across
            # SDK versions, so be permissive: a video track has no useful
            # AudioStream and AudioStream() will raise — let it through
            # and the per-track exception handler logs and drops it.
            kind = getattr(track, "kind", None)
            try:
                # Accept TrackKind.KIND_AUDIO or numeric value 1, or a
                # MagicMock placeholder in tests.
                audio_kind = getattr(lk_rtc.TrackKind, "KIND_AUDIO", None)
                if audio_kind is not None and kind != audio_kind and kind != 1:
                    return
            except Exception:
                pass
            readers.append(asyncio.create_task(_drain_track(track)))

        room.on("track_subscribed", _on_track_subscribed)

        try:
            while True:
                frame_bytes = await queue.get()
                if frame_bytes is None:
                    return
                yield frame_bytes
        finally:
            try:
                room.off("track_subscribed", _on_track_subscribed)
            except Exception:  # pragma: no cover
                pass
            for t in readers:
                if not t.done():
                    t.cancel()

    async def _handle_audio_frame(self, frame: bytes, state: Dict[str, Any]) -> None:
        """Push a single 20ms PCM frame to ASR and the VAD."""
        asr = state.get("asr")
        if asr is not None:
            try:
                await asr.push_pcm(frame)
            except Exception:  # pragma: no cover
                logger.exception("voice_rtc: asr.push_pcm failed")

        # Two VAD shapes: (a) a livekit-agents VADStream consumes
        # ``rtc.AudioFrame`` via ``push_frame``; (b) tests use a simple
        # stub that takes raw bytes via ``push_frame`` or ``feed``.
        vad_stream = state.get("vad_stream")
        if vad_stream is not None:
            try:
                from livekit import rtc as lk_rtc  # type: ignore
                samples = len(frame) // 2
                af = lk_rtc.AudioFrame(
                    data=frame,
                    sample_rate=_ASR_SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=samples,
                )
                push = getattr(vad_stream, "push_frame", None)
                if push is not None:
                    res = push(af)
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:  # pragma: no cover
                logger.exception("voice_rtc: vad_stream push_frame failed")
        else:
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
    # Audio-out: token stream → chunker → TTS → AudioSource
    # ------------------------------------------------------------------

    async def on_assistant_token_stream(
        self,
        chat_id: str,
        token_iterator: Any,
    ) -> None:
        """Streaming-output hook called by the gateway runner.

        Routes the assistant's token stream through ``_on_assistant_stream``
        for the matching active call.  If no call is active for ``chat_id``
        we fall back to the base implementation (which joins and calls
        ``send``) so a stray hook invocation doesn't blow up.
        """
        if chat_id in self._active_calls:
            await self._on_assistant_stream(chat_id, token_iterator)
            return
        await super().on_assistant_token_stream(chat_id, token_iterator)

    async def _on_assistant_stream(
        self,
        room_name: str,
        token_iterator: Any,
    ) -> None:
        """Drive the chunker → TTS → AudioSource pipeline for one turn.

        Tracks ``state["tts_task"]`` so a barge-in (``_on_vad_speech_start``)
        can cancel the in-flight synthesis cleanly.
        """
        from tools.voice_rtc.chunker import ClauseChunker
        from tools.voice_rtc.state import Event as TurnEvent

        state = self._active_calls.get(room_name)
        if state is None:
            logger.debug("voice_rtc: _on_assistant_stream called for unknown room %s", room_name)
            return

        chunker = ClauseChunker()

        # The pipeline body — runs as a task so a barge-in can cancel us.
        async def _pipeline() -> None:
            tts = self._ensure_tts_stream(state)
            try:
                async for delta in token_iterator:
                    if not delta:
                        continue
                    for chunk in chunker.feed(delta):
                        await self._synth_and_publish(chunk, state, tts)
                for chunk in chunker.flush():
                    await self._synth_and_publish(chunk, state, tts)
                # Successful end of turn — advance FSM if we were SPEAKING.
                ts = state.get("turn_state")
                if ts is not None:
                    ts.handle(TurnEvent.TTS_DONE)
            except asyncio.CancelledError:
                # Barge-in cancelled us. The FSM transition to LISTENING
                # is driven by ``_on_vad_speech_start`` after we exit.
                raise

        task = asyncio.create_task(_pipeline())
        state["tts_task"] = task
        try:
            await task
        except asyncio.CancelledError:
            # Barge-in path. ``_on_vad_speech_start`` clears the source
            # queue and dispatches CANCEL_DONE — we just exit cleanly.
            pass
        finally:
            if state.get("tts_task") is task:
                state["tts_task"] = None

    def _ensure_tts_stream(self, state: Dict[str, Any]) -> Any:
        """Lazy-construct (and cache per call) a SarvamTTSStream."""
        existing = state.get("tts_stream")
        if existing is not None:
            return existing
        from tools.sarvam_tts import SarvamTTSStream
        tts = SarvamTTSStream(
            api_key=self._sarvam_api_key,
            sample_rate=_ASR_SAMPLE_RATE,
        )
        state["tts_stream"] = tts
        return tts

    async def _synth_and_publish(
        self,
        text: str,
        state: Dict[str, Any],
        tts: Any,
    ) -> None:
        """Synthesize ``text`` and push the PCM into the AudioSource as
        20 ms AudioFrames.  Dispatches TTS_FIRST_AUDIO on the first frame
        of the turn."""
        from livekit import rtc as lk_rtc  # type: ignore
        from tools.voice_rtc.state import Event as TurnEvent

        source = state.get("audio_source")
        if source is None:
            logger.debug("voice_rtc: no audio_source on call state; dropping TTS")
            return

        ts = state.get("turn_state")
        pending = bytearray()

        async def _flush_frame(buf: bytearray) -> None:
            data = bytes(buf)
            samples = len(data) // 2
            if samples == 0:
                return
            frame = lk_rtc.AudioFrame(
                data=data,
                sample_rate=_ASR_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=samples,
            )
            # Drive the FSM on first audio of the turn.
            if ts is not None and ts.state == "THINKING":
                ts.handle(TurnEvent.TTS_FIRST_AUDIO)
            await source.capture_frame(frame)

        synth_iter = tts.synth(text)
        try:
            async for chunk in synth_iter:
                if not chunk:
                    continue
                pending.extend(chunk)
                while len(pending) >= _TTS_FRAME_BYTES:
                    frame_bytes = bytes(pending[:_TTS_FRAME_BYTES])
                    del pending[:_TTS_FRAME_BYTES]
                    await _flush_frame(bytearray(frame_bytes))
            # Any tail < a full 20 ms frame still gets shipped — pad to
            # an even sample count so samples_per_channel is correct.
            if len(pending) >= 2:
                tail = bytes(pending[: (len(pending) // 2) * 2])
                pending.clear()
                await _flush_frame(bytearray(tail))
        finally:
            # Make sure the underlying SDK iterator is closed promptly on
            # cancellation so the HTTPX stream doesn't dangle.
            aclose = getattr(synth_iter, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # pragma: no cover
                    pass

    async def _on_vad_speech_start(self, room_name: str) -> None:
        """Barge-in handler. Called by the audio-in path when VAD
        detects user speech onset.  Drives the FSM through
        VAD_SPEECH_START → CANCEL_DONE and tears down any in-flight TTS."""
        from tools.voice_rtc.state import Event as TurnEvent

        state = self._active_calls.get(room_name)
        if state is None:
            return
        ts = state.get("turn_state")
        if ts is None:
            return

        # Only barge in if we are mid-turn.
        if ts.state not in ("THINKING", "SPEAKING"):
            return

        ts.handle(TurnEvent.VAD_SPEECH_START)

        tts_task = state.get("tts_task")
        if tts_task is not None and not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except (asyncio.CancelledError, Exception):
                pass

        # Drop anything queued in the AudioSource so the user doesn't
        # keep hearing the agent talking past the interruption point.
        source = state.get("audio_source")
        if source is not None:
            clear = getattr(source, "clear_queue", None)
            if clear is not None:
                try:
                    res = clear()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:  # pragma: no cover
                    pass

        ts.handle(TurnEvent.CANCEL_DONE)

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
        """One-shot fallback: feed ``content`` through the same
        chunker → TTS → AudioSource pipeline as the streaming path.

        If no active call matches ``chat_id`` (e.g. the runner sent a
        system message before the room was joined) we just return
        success — voice_rtc has no text persistence.
        """
        if not content or chat_id not in self._active_calls:
            return SendResult(success=True, message_id="voice")

        async def _single():
            yield content

        await self._on_assistant_stream(chat_id, _single())
        return SendResult(success=True, message_id="voice")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}
