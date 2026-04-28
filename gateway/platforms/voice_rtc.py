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
import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.voice_rtc_sessions import SessionRegistry

logger = logging.getLogger(__name__)


# Default idle TTL for a per-user session — long enough that a quick
# hangup-and-redial reattaches to the same agent state, short enough
# that a forgotten session doesn't leak HTTP clients indefinitely.
_SESSION_IDLE_TTL_SECONDS = 600.0

# Sarvam's public OpenAI-compatible endpoint. Used as a fallback when no
# Pravah credentials are configured. ``sarvam-m`` is the only public chat
# model that streams content directly (sarvam-30b / sarvam-105b emit
# reasoning tokens into a separate field with multi-second first-content
# latency, which is unacceptable for voice).
_DEFAULT_SARVAM_BASE_URL = "https://api.sarvam.ai/v1"
_DEFAULT_SARVAM_MODEL = "sarvam-m"

# Pravah / IndiaAI internal endpoint — non-reasoning 100B SFT variant of
# Sarvam-105B. Smoke-tested at ~503ms first token and ~561ms total for a
# greeting; no reasoning preamble. Preferred over the public endpoint
# when its env vars are set.
_DEFAULT_PRAVAH_BASE_URL = "https://api.pravah.indiaai.sarvam.ai/v1"
_DEFAULT_PRAVAH_MODEL = "Sarvam-100b@SFT-14k#64k-ctx"

# Voice-mode directive prepended to every system prompt. Without this the
# model emits XML-style tool-call markup (skills_tool, lookup_order, etc.)
# which streams into TTS as audible gibberish — there's no tool-execution
# loop in V2VAgentSession yet (deferred to the Hermes-AIAgent integration
# milestone). For now: tell the model to answer from prompt context only.
_VOICE_MODE_DIRECTIVE = """\
# Voice-mode operating directives (LOAD-BEARING — read first)

You are operating in spoken-conversation mode. Your output is synthesized
to audio in real time and played to the user.

- Answer in plain spoken English. Use short sentences and contractions.
- Do NOT emit tool-call markup of any kind. No `<tool_call>...`, no XML,
  no JSON, no function-call syntax, no markdown lists or code blocks.
  These are not interactive in voice mode and would be heard as gibberish.
- Use ONLY the information already present in this prompt (user memory,
  SOP, active skill content). Do not pretend to "look something up";
  if you don't have the answer in this prompt, say you'll follow up
  and either schedule a callback or transfer to a human.
- Pronounce order IDs naturally: "ACME order ending in three-four-five-six"
  for "ACME-3456". Never read URLs or JSON aloud.
- When you reference user-specific facts (commitments, refunds, deliveries),
  source them from the User memory section below.

---

"""


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


# Topic used for v2v transcript / observability events on the LiveKit data
# channel. The browser subscribes to ``DataReceived`` and filters on this
# topic so other apps sharing the room don't see our wire format.
_TRANSCRIPT_TOPIC = "v2v.transcript"


# Inter-clause silence pushed between TTS clauses so the agent doesn't
# sound like one breathless run-on. ``_CLAUSE`` is the default gap between
# any two adjacent clauses (commas, mid-sentence breaks); ``_SENTENCE``
# is the longer pause inserted when the just-finished clause ended with
# terminal punctuation (. ! ?). Tunable via env so an operator can ramp
# it up or down without a redeploy.
_DEFAULT_INTER_CLAUSE_SILENCE_MS = 80
_DEFAULT_INTER_SENTENCE_SILENCE_MS = 220


# Tunable VAD knobs exposed via env. Each maps to a Sarvam Saaras streaming
# connect kwarg of the same name (snake_case). The defaults below relax
# Saaras' aggressive end-of-speech detection so users get more time to
# finish a sentence before a final fires. Override any of them via env to
# tune for a specific deployment / mic / speaking style.
#
# The two main knobs to tune if speech is being cut off mid-sentence:
#   V2V_VAD_HIGH_SENSITIVITY=false     (coarse: turns down all of the above)
#   V2V_VAD_NEGATIVE_FRAMES_COUNT=40   (fine: requires N consecutive silence
#                                      frames before declaring end-of-speech.
#                                      32-frame default ≈ 640ms; 40 ≈ 800ms)
_VAD_BOOL_ENVS = {
    "V2V_VAD_HIGH_SENSITIVITY": "high_vad_sensitivity",
    "V2V_VAD_SIGNALS": "vad_signals",
}
_VAD_FLOAT_ENVS = {
    "V2V_VAD_POSITIVE_SPEECH_THRESHOLD": "positive_speech_threshold",
    "V2V_VAD_NEGATIVE_SPEECH_THRESHOLD": "negative_speech_threshold",
    "V2V_VAD_START_SPEECH_VOLUME_THRESHOLD": "start_speech_volume_threshold",
}
_VAD_INT_ENVS = {
    "V2V_VAD_MIN_SPEECH_FRAMES": "min_speech_frames",
    "V2V_VAD_FIRST_TURN_MIN_SPEECH_FRAMES": "first_turn_min_speech_frames",
    "V2V_VAD_NEGATIVE_FRAMES_COUNT": "negative_frames_count",
    "V2V_VAD_NEGATIVE_FRAMES_WINDOW": "negative_frames_window",
    "V2V_VAD_INTERRUPT_MIN_SPEECH_FRAMES": "interrupt_min_speech_frames",
    "V2V_VAD_PRE_SPEECH_PAD_FRAMES": "pre_speech_pad_frames",
    "V2V_VAD_NUM_INITIAL_IGNORED_FRAMES": "num_initial_ignored_frames",
}


def _agent_opens_call() -> bool:
    """Whether the agent should speak first when a call connects.

    Default ``True``. Disable with ``V2V_AGENT_OPENS_CALL=false`` (or any
    of ``0``, ``no``, ``off``).
    """
    val = os.environ.get("V2V_AGENT_OPENS_CALL")
    if val is None:
        return True
    return val.strip().lower() not in ("0", "false", "no", "off")


def _silence_after(text: str) -> int:
    """Return the silence (in ms) to insert AFTER a TTS clause.

    Sentence-end punctuation gets the longer gap; everything else gets the
    shorter inter-clause gap. Both are env-tunable via
    ``V2V_TTS_CLAUSE_SILENCE_MS`` and ``V2V_TTS_SENTENCE_SILENCE_MS``.
    """
    s = (text or "").rstrip()
    if not s:
        return 0
    last = s[-1]
    if last in {".", "?", "!", "…"}:
        env = os.environ.get("V2V_TTS_SENTENCE_SILENCE_MS")
        try:
            return max(0, int(env)) if env else _DEFAULT_INTER_SENTENCE_SILENCE_MS
        except ValueError:
            return _DEFAULT_INTER_SENTENCE_SILENCE_MS
    env = os.environ.get("V2V_TTS_CLAUSE_SILENCE_MS")
    try:
        return max(0, int(env)) if env else _DEFAULT_INTER_CLAUSE_SILENCE_MS
    except ValueError:
        return _DEFAULT_INTER_CLAUSE_SILENCE_MS


async def _flush_silence(flush_frame_fn: Callable[[bytearray], Awaitable[None]], ms: int) -> None:
    """Push ``ms`` of mono s16le silence at 16 kHz through the framing fn."""
    if ms <= 0:
        return
    samples = (_ASR_SAMPLE_RATE * ms) // 1000
    bytes_remaining = samples * 2
    while bytes_remaining > 0:
        chunk = min(bytes_remaining, _TTS_FRAME_BYTES)
        await flush_frame_fn(bytearray(chunk))  # zeros — bytearray default
        bytes_remaining -= chunk


def _read_vad_env() -> Dict[str, Any]:
    """Read V2V_VAD_* env vars into the kwarg shape SarvamASRStream expects.

    Default behavior when no envs are set: ``high_vad_sensitivity=False``
    (relaxed end-of-speech detection) so the user has more time to finish
    speaking. The Sarvam SDK's own defaults apply for every other knob.
    """
    out: Dict[str, Any] = {}
    if "V2V_VAD_HIGH_SENSITIVITY" not in os.environ:
        out["high_vad_sensitivity"] = False
    for env_name, kwarg in _VAD_BOOL_ENVS.items():
        val = os.environ.get(env_name)
        if val is not None:
            out[kwarg] = val.strip().lower() in ("1", "true", "yes", "on")
    for env_name, kwarg in _VAD_FLOAT_ENVS.items():
        val = os.environ.get(env_name)
        if val is not None and val.strip():
            try:
                out[kwarg] = float(val)
            except ValueError:
                logger.warning("voice_rtc: ignoring invalid float for %s=%r", env_name, val)
    for env_name, kwarg in _VAD_INT_ENVS.items():
        val = os.environ.get(env_name)
        if val is not None and val.strip():
            try:
                out[kwarg] = int(val)
            except ValueError:
                logger.warning("voice_rtc: ignoring invalid int for %s=%r", env_name, val)
    return out


_V2V_CONSOLE_LOG_PREFIXES: Tuple[str, ...] = (
    "gateway.platforms.voice_rtc",
    "run_agent",
    "agent",
    "tools.sarvam_",
    "tools.voice_rtc",
    "v2v.web",
)


class _V2VConsoleFilter(logging.Filter):
    """Allow records whose logger name matches the v2v allowlist prefixes."""

    def __init__(self, prefixes: Tuple[str, ...]) -> None:
        super().__init__()
        self._prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 — Filter contract
        name = record.name or ""
        for p in self._prefixes:
            if p.endswith("_"):
                if name.startswith(p):
                    return True
            elif name == p or name.startswith(p + "."):
                return True
        return False


def _install_console_log_handler() -> Optional[logging.StreamHandler]:
    """Attach a stdout handler to the root logger for v2v-namespaced loggers.

    Idempotent: a handler tagged with ``_v2v_console = True`` is installed at
    most once. Returns the handler (existing or freshly installed) so callers
    and tests can inspect / swap its stream.
    """
    root = logging.getLogger()
    for h in root.handlers:
        if getattr(h, "_v2v_console", False):
            return h  # type: ignore[return-value]
    level_name = os.environ.get("V2V_CONSOLE_LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        level = logging.INFO
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.addFilter(_V2VConsoleFilter(_V2V_CONSOLE_LOG_PREFIXES))
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    handler._v2v_console = True  # type: ignore[attr-defined]
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    root.addHandler(handler)
    return handler


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


# ----------------------------------------------------------------------
# V2VAgentSession — per-user agent driver backed by Hermes ``AIAgent``
# ----------------------------------------------------------------------


_DEFAULT_V2V_TOOLSETS: List[str] = [
    "skills",
    "memory",
    "file",
    "cronjob",
    "delegation",
    "v2v",
]


class V2VAgentSession:
    """Single-user, single-thread chat session that drives Hermes'
    ``AIAgent`` from the LiveKit voice path.

    Public surface kept identical to the prior Sarvam-direct implementation
    so the rest of the adapter (``_build_v2v_session``, ``_active_calls``,
    ``_feed_to_agent``, ``on_assistant_token_stream``) works unchanged::

        gen = await session.submit_user_turn("hello")
        async for tok in gen:
            ...  # feed into TTS / chunker

    Internally we run ``AIAgent.run_conversation`` (synchronous) on a
    worker thread and bridge its sync ``stream_delta_callback`` into an
    asyncio queue that ``_gen`` drains as an async iterator.
    """

    def __init__(
        self,
        *,
        user_id: str,
        api_key: str,
        model: str,
        system_prompt: str,
        base_url: str = _DEFAULT_SARVAM_BASE_URL,
        max_history_turns: int = 32,
        enabled_toolsets: Optional[List[str]] = None,
        chat_id: Optional[str] = None,
        max_tokens: int = 4096,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_complete: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
        on_skill_loaded: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        from run_agent import AIAgent  # lazy import — keeps tests' monkeypatch order simple

        self.user_id = user_id
        self.model = model
        self.system_prompt = system_prompt or ""
        self._max_history_turns = max_history_turns
        self._on_tool_start = on_tool_start
        self._on_tool_complete = on_tool_complete
        self._on_skill_loaded = on_skill_loaded

        self._agent = AIAgent(
            base_url=base_url,
            api_key=api_key,
            model=model,
            enabled_toolsets=list(enabled_toolsets) if enabled_toolsets else list(_DEFAULT_V2V_TOOLSETS),
            ephemeral_system_prompt=self.system_prompt,
            quiet_mode=True,
            verbose_logging=False,
            session_id=f"v2v-{user_id}",
            user_id=user_id,
            chat_id=chat_id or user_id,
            platform="voice_rtc",
            max_iterations=20,
            max_tokens=max_tokens,
        )

        self._history: List[Dict[str, str]] = []

    async def submit_user_turn(self, text: str) -> AsyncIterator[str]:
        """Run one turn through ``AIAgent.run_conversation`` on a worker
        thread; yield content deltas as they arrive; persist user +
        assistant in history when the turn completes."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def _on_delta(delta: str) -> None:
            if delta:
                loop.call_soon_threadsafe(queue.put_nowait, delta)

        def _on_tool_start_cb(*args, **kwargs) -> None:
            tool_name, tool_args = _extract_tool_event(args, kwargs)
            if self._on_tool_start is None:
                return
            try:
                self._on_tool_start(tool_name, tool_args)
            except Exception:  # pragma: no cover
                logger.exception("v2v: tool_start callback raised")

        def _on_tool_complete_cb(*args, **kwargs) -> None:
            tool_name, tool_args, result = _extract_tool_event(args, kwargs, with_result=True)
            if self._on_tool_complete is None:
                return
            try:
                self._on_tool_complete(tool_name, tool_args, result)
            except Exception:  # pragma: no cover
                logger.exception("v2v: tool_complete callback raised")

        self._agent.stream_delta_callback = _on_delta
        self._agent.tool_start_callback = _on_tool_start_cb
        self._agent.tool_complete_callback = _on_tool_complete_cb

        # AIAgent.run_conversation appends the user message internally
        # (see run_agent.py around line 9699), so the snapshot we hand it
        # must NOT yet contain this turn's user message — otherwise the
        # turn would be doubled in the rolling history.
        history_for_call = list(self._history)
        self._history.append({"role": "user", "content": text})

        result_holder: Dict[str, Any] = {}

        def _run_blocking() -> None:
            try:
                result_holder["result"] = self._agent.run_conversation(
                    user_message=text,
                    conversation_history=history_for_call,
                )
            except Exception as exc:  # pragma: no cover
                result_holder["error"] = exc
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        runner_task = asyncio.create_task(asyncio.to_thread(_run_blocking))

        history_ref = self._history
        max_turns = self._max_history_turns

        async def _gen() -> AsyncIterator[str]:
            collected: List[str] = []
            try:
                while True:
                    item = await queue.get()
                    if item is _SENTINEL:
                        break
                    collected.append(item)
                    yield item
            finally:
                try:
                    await runner_task
                except Exception:  # pragma: no cover
                    logger.exception("v2v: runner task await raised")
                text_out = "".join(collected).strip()
                if not text_out:
                    res = result_holder.get("result") or {}
                    text_out = (
                        res.get("final_response")
                        or res.get("response")
                        or ""
                    ).strip()
                history_ref.append({"role": "assistant", "content": text_out})
                if max_turns > 0:
                    head = history_ref[:1] if history_ref and history_ref[0].get("role") == "system" else []
                    tail = history_ref[len(head):]
                    limit = max_turns * 2
                    if len(tail) > limit:
                        del tail[: len(tail) - limit]
                    self._history = head + tail
                err = result_holder.get("error")
                if err is not None:
                    logger.warning("v2v: AIAgent.run_conversation raised: %r", err)

        return _gen()

    async def close(self) -> None:
        """Drop the underlying agent. AIAgent has no async-clean teardown;
        sessions are GC'd via the registry's idle-eviction path."""
        self._agent = None  # release the reference; no explicit close API


def _extract_tool_event(
    args: tuple,
    kwargs: Dict[str, Any],
    *,
    with_result: bool = False,
) -> tuple:
    """Normalise AIAgent tool callback invocations to ``(name, args[, result])``.

    AIAgent fires ``tool_start_callback(tc_id, name, args)`` and
    ``tool_complete_callback(tc_id, name, args, result)``; tests in this
    file may invoke the wrapper directly with the simpler
    ``(name, args[, result])`` shape. Detect by sniffing the first
    positional.
    """
    name = ""
    targs: Dict[str, Any] = {}
    result: Any = None
    if len(args) >= 4 and with_result:
        # (tc_id, name, args, result)
        name = str(args[1] or "")
        targs = args[2] if isinstance(args[2], dict) else {}
        result = args[3]
    elif len(args) >= 3 and not with_result:
        # (tc_id, name, args)
        name = str(args[1] or "")
        targs = args[2] if isinstance(args[2], dict) else {}
    elif len(args) >= 1 and isinstance(args[0], str) and (len(args) < 2 or isinstance(args[1], (dict, type(None)))):
        # (name, args[, result])
        name = args[0]
        targs = args[1] if len(args) >= 2 and isinstance(args[1], dict) else {}
        if with_result and len(args) >= 3:
            result = args[2]
    else:
        name = kwargs.get("tool_name") or kwargs.get("name") or (args[1] if len(args) > 1 else "") or ""
        ka = kwargs.get("tool_args") or kwargs.get("args") or {}
        targs = ka if isinstance(ka, dict) else {}
        if with_result:
            result = kwargs.get("result")
    if with_result:
        return name, targs, result
    return name, targs


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

        # LLM credentials — prefer Pravah (non-reasoning 100B SFT) when
        # configured; fall back to public Sarvam (sarvam-m) otherwise.
        # ASR/TTS still use SARVAM_API_KEY (those endpoints are unchanged).
        self._pravah_api_key: str = os.getenv("PRAVAH_API_KEY", "")
        self._pravah_base_url: str = (
            os.getenv("PRAVAH_BASE_URL") or _DEFAULT_PRAVAH_BASE_URL
        )
        self._pravah_model: str = os.getenv("PRAVAH_MODEL") or _DEFAULT_PRAVAH_MODEL

        # Per-call bookkeeping. Keys are room names (v2v-<user_id>-<call_id>).
        # Values include: asr, turn_state, audio_source, tts_stream, tts_task,
        # asr_consumer_task, vad_task, audio_task, user_id, call_id, session.
        self._active_calls: Dict[str, Dict[str, Any]] = {}

        # Per-user persistent agent sessions. The factory builds a
        # V2VAgentSession (Sarvam-direct OpenAI-compatible client) using
        # the system prompt assembled by ``v2v_memory_loader``. Tests
        # swap ``self._sessions._factory`` to inject a stub.
        self._v2v_data_root: Path = self._resolve_data_root(extra)
        self._v2v_global_path: Path = self._v2v_data_root / "agent_workflow.md"
        self._sessions: SessionRegistry = SessionRegistry(
            factory=self._build_v2v_session,
            idle_ttl_seconds=_SESSION_IDLE_TTL_SECONDS,
        )

        # Background task running the LiveKit Agents worker.
        self._worker_task: Optional[asyncio.Task] = None
        self._worker: Any = None

    # ------------------------------------------------------------------
    # V2V session factory + memory plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_data_root(extra: Dict[str, Any]) -> Path:
        """Pick the v2v memory-data root: explicit config > env > home."""
        cfg_root = extra.get("data_root")
        if cfg_root:
            return Path(cfg_root).expanduser()
        env_root = os.getenv("V2V_DATA_ROOT")
        if env_root:
            return Path(env_root).expanduser()
        return Path.home() / ".hermes" / "v2v"

    def _user_memory_path(self, user_id: str) -> Path:
        return self._v2v_data_root / "users" / user_id / "memory.md"

    def _resolve_router_skill_path(self) -> Optional[Path]:
        """Find ``skills/router/SKILL.md`` from env / data-root convention.

        Tried in order:

        1. ``$V2V_SKILLS_DIR/router/SKILL.md`` if the env var is set.
        2. ``$V2V_HARNESS_ROOT/skills/router/SKILL.md``.
        3. ``<data_root>/../skills/router/SKILL.md`` — both ``data/`` and
           ``skills/`` typically live in the v2v_harness repo root.

        Returns ``None`` if no candidate exists; the factory then falls
        back to ``skill_text=""`` and the agent runs on memory + the
        baked-in voice-mode directive only.
        """
        candidates: list[Path] = []
        sd = os.getenv("V2V_SKILLS_DIR")
        if sd:
            candidates.append(Path(sd).expanduser() / "router" / "SKILL.md")
        hr = os.getenv("V2V_HARNESS_ROOT")
        if hr:
            candidates.append(Path(hr).expanduser() / "skills" / "router" / "SKILL.md")
        # data_root is typically <repo>/data, so <repo>/skills is sibling.
        candidates.append(self._v2v_data_root.parent / "skills" / "router" / "SKILL.md")
        for c in candidates:
            try:
                if c.is_file():
                    return c
            except OSError:
                continue
        return None

    # ------------------------------------------------------------------
    # Data-channel transcript events
    # ------------------------------------------------------------------

    async def _publish_event(self, room_name: str, event: Dict[str, Any]) -> None:
        """Publish a JSON event to the room's LiveKit data channel.

        Used to drive the browser's transcript visualizer. Best-effort —
        in unit tests ``room.local_participant.publish_data`` is absent
        and we silently no-op so adding events doesn't tangle existing
        tests' fakes.

        Schema (all events have ``type`` and ``ts`` ISO8601 UTC):

          ``session_ready``     — call up; payload has user_id, model, skill_path
          ``user_message``      — ASR final received; payload has text
          ``assistant_chunk``   — one TTS-bound clause; payload has text
          ``assistant_done``    — turn complete; payload has full reply text
          ``barge_in``          — user interrupted mid-TTS
          ``skill_loaded``      — voice-safe router skill content size at attach
        """
        state = self._active_calls.get(room_name)
        if state is None:
            return
        room = state.get("room")
        if room is None:
            return
        local = getattr(room, "local_participant", None)
        if local is None:
            return
        publish = getattr(local, "publish_data", None)
        if publish is None:
            return
        try:
            payload_dict = dict(event)
            payload_dict["ts"] = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
            payload = json.dumps(payload_dict).encode("utf-8")
        except Exception:  # pragma: no cover
            logger.warning("voice_rtc: failed to serialize transcript event %r", event)
            return

        # The Python livekit-rtc SDK changed the publish_data signature
        # across versions: some accept a positional bytes payload + topic
        # kw, others expect a ``DataPacket``-style object. Try the most
        # common shape first; fall back to a plain positional call.
        try:
            result = publish(payload, reliable=True, topic=_TRANSCRIPT_TOPIC)
        except TypeError:
            try:
                result = publish(payload, topic=_TRANSCRIPT_TOPIC)
            except TypeError:
                try:
                    result = publish(payload)
                except Exception:  # pragma: no cover
                    logger.warning("voice_rtc: publish_data unsupported", exc_info=True)
                    return
        except Exception:  # pragma: no cover
            logger.warning("voice_rtc: publish_data raised", exc_info=True)
            return

        if asyncio.iscoroutine(result):
            try:
                await result
            except Exception:  # pragma: no cover
                logger.warning("voice_rtc: publish_data coro raised", exc_info=True)

    async def _build_v2v_session(self, user_id: str) -> "V2VAgentSession":
        """Default :class:`SessionRegistry` factory.

        Builds a system prompt from the global SOP + per-user memory +
        the voice-safe slice of the router skill, sandwiches it with the
        voice-mode directive, and constructs the per-user
        :class:`V2VAgentSession` (which wraps Hermes' ``AIAgent``)
        backed by Pravah (preferred) or public Sarvam (fallback).

        Tests swap ``self._sessions._factory`` after construction so
        this path doesn't run in CI.
        """
        from agent.v2v_memory_loader import build_system_prompt, load_router_for_voice

        skill_text = ""
        router_path = self._resolve_router_skill_path()
        if router_path is not None:
            try:
                skill_text = load_router_for_voice(router_path)
                logger.info(
                    "voice_rtc: loaded voice-safe router skill from %s (%d chars)",
                    router_path, len(skill_text),
                )
            except Exception:  # pragma: no cover
                logger.exception("voice_rtc: load_router_for_voice failed for %s", router_path)
                skill_text = ""
        else:
            logger.warning(
                "voice_rtc: no router skill found on disk; agent will run on memory + directives only"
            )

        body = build_system_prompt(
            global_path=self._v2v_global_path,
            user_path=self._user_memory_path(user_id),
            skill_text=skill_text,
        )

        recency_reminder = (
            "\n\n---\n\n"
            "REMINDER: spoken-conversation mode. Answer in plain English "
            "from the prompt context above. Do NOT emit any tool calls, "
            "XML tags, JSON, or markdown formatting in your response. "
            "If the user asks something not covered by the context, say "
            "you'll follow up rather than fabricating."
        )
        system_prompt = _VOICE_MODE_DIRECTIVE + (body or "") + recency_reminder

        toolsets_env = os.getenv("V2V_TOOLSETS")
        if toolsets_env:
            enabled_toolsets = [t.strip() for t in toolsets_env.split(",") if t.strip()]
        else:
            enabled_toolsets = list(_DEFAULT_V2V_TOOLSETS)

        tool_event_queue: asyncio.Queue = asyncio.Queue()
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        def _on_tool_start(name: str, targs: Dict[str, Any]) -> None:
            if event_loop is None:
                tool_event_queue.put_nowait(("tool_start", name, targs, None))
                return
            event_loop.call_soon_threadsafe(
                tool_event_queue.put_nowait, ("tool_start", name, targs, None)
            )

        def _on_tool_complete(name: str, targs: Dict[str, Any], result: Any) -> None:
            if event_loop is None:
                tool_event_queue.put_nowait(("tool_complete", name, targs, result))
                return
            event_loop.call_soon_threadsafe(
                tool_event_queue.put_nowait, ("tool_complete", name, targs, result)
            )

        if self._pravah_api_key:
            api_key = self._pravah_api_key
            model = self._pravah_model
            base_url = self._pravah_base_url
            llm_provider = "pravah"
        else:
            api_key = self._sarvam_api_key
            model = _DEFAULT_SARVAM_MODEL
            base_url = _DEFAULT_SARVAM_BASE_URL
            llm_provider = "sarvam-public"

        try:
            max_tokens = int(os.environ.get("V2V_MAX_OUTPUT_TOKENS", "4096"))
        except ValueError:
            max_tokens = 4096

        session = V2VAgentSession(
            user_id=user_id,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            base_url=base_url,
            enabled_toolsets=enabled_toolsets,
            chat_id=user_id,
            max_tokens=max_tokens,
            on_tool_start=_on_tool_start,
            on_tool_complete=_on_tool_complete,
        )
        try:
            session._tool_event_queue = tool_event_queue  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass

        # Stash provenance so the transcript visualiser / logs can show
        # which provider+skill+memory backed this session.
        try:
            session._loaded_skill_path = str(router_path) if router_path else None  # type: ignore[attr-defined]
            session._loaded_skill_chars = len(skill_text)  # type: ignore[attr-defined]
            session._llm_provider = llm_provider  # type: ignore[attr-defined]
            session._user_memory_path = str(self._user_memory_path(user_id))  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass
        logger.info(
            "voice_rtc: built v2v session user_id=%s provider=%s model=%s skill=%s",
            user_id, llm_provider, getattr(session, "model", "?"), router_path or "<none>",
        )
        return session

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        _install_console_log_handler()
        try:
            self._start_worker()
        except Exception as exc:  # pragma: no cover — surfaces as fatal
            logger.exception("voice_rtc: worker startup failed: %s", exc)
            self._set_fatal_error("voice_rtc_worker", str(exc), retryable=False)
            return False
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        # Release any held sessions back to the registry first; the
        # registry keeps them cached for the idle-TTL window so quick
        # reconnects re-attach to the same agent state.
        released: set[str] = set()
        for state in self._active_calls.values():
            user_id = state.get("user_id")
            if user_id and user_id not in released:
                try:
                    await self._sessions.release(user_id)
                except Exception:  # pragma: no cover
                    logger.exception(
                        "voice_rtc: session release failed for %s", user_id
                    )
                released.add(user_id)

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
            for task_name in ("asr_consumer_task", "vad_task", "audio_task", "tts_task", "tool_event_task", "greeting_task"):
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

        # Disconnect is the adapter's shutdown — drop every cached
        # agent session so we don't leak HTTP clients across runs.
        try:
            await self._sessions.close()
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: session registry close failed")

        self._mark_disconnected()

    # ------------------------------------------------------------------
    # Worker bootstrapping (testable seam)
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        from livekit import agents as lk_agents  # type: ignore

        # Use the threaded executor instead of the default process pool.
        # Our entrypoint is a bound method on this adapter; the adapter
        # holds non-picklable state (asyncio locks, the SessionRegistry,
        # logger handles), so spawning workers as separate processes
        # fails with `cannot pickle '_thread.lock' object`. Threads share
        # the parent's address space and avoid that entire class of bug.
        options = lk_agents.WorkerOptions(
            entrypoint_fnc=self._entrypoint,
            ws_url=self._lk_url or None,
            api_key=self._lk_api_key or None,
            api_secret=self._lk_api_secret or None,
            job_executor_type=lk_agents.JobExecutorType.THREAD,
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

        # LiveKit Agents v1.5+ requires the entrypoint to explicitly
        # await ctx.connect() before reading room.local_participant /
        # subscribing to tracks. Older SDKs auto-connect; the guard
        # keeps the audio-in unit tests (which pass a vanilla MagicMock
        # ctx without a real .connect) working.
        connect = getattr(ctx, "connect", None)
        if callable(connect):
            try:
                result = connect()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover
                logger.exception("voice_rtc: ctx.connect() failed for %s", room_name)
                return

        from tools.sarvam_asr import SarvamASRStream
        from tools.voice_rtc.state import TurnState

        asr = SarvamASRStream(
            api_key=self._sarvam_api_key,
            sample_rate=_ASR_SAMPLE_RATE,
            language_code=_ASR_LANGUAGE_CODE,
            **_read_vad_env(),
        )
        turn_state = TurnState()

        # Acquire the user's persistent agent session up-front so the
        # first ASR final lands in a warm session. The registry caches
        # it across calls within ``_SESSION_IDLE_TTL_SECONDS``.
        try:
            session = await self._sessions.get_or_create(user_id)
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: failed to acquire session for %s", user_id)
            session = None

        state: Dict[str, Any] = {
            "asr": asr,
            "turn_state": turn_state,
            "user_id": user_id,
            "call_id": call_id,
            "session": session,
            # Capture the room handle so _publish_event can write to the
            # data channel without re-walking ctx every time.
            "room": getattr(ctx, "room", None),
            # Per-turn assistant text accumulator — flushed on assistant_done.
            "assistant_buffer": [],
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

        # Tell the browser visualiser the call is up. Best-effort — if
        # publish_data isn't available (unit tests, older SDK), this is a
        # no-op. Includes provenance so the operator can see which skill
        # and which model are driving this call.
        try:
            await self._publish_event(
                room_name,
                {
                    "type": "session_ready",
                    "user_id": user_id,
                    "call_id": call_id,
                    "model": getattr(session, "model", None) if session else None,
                    "provider": getattr(session, "_llm_provider", None) if session else None,
                    "skill_path": getattr(session, "_loaded_skill_path", None) if session else None,
                    "skill_chars": getattr(session, "_loaded_skill_chars", 0) if session else 0,
                    "memory_path": getattr(session, "_user_memory_path", None) if session else None,
                },
            )
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: failed to publish session_ready for %s", room_name)

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

        tool_event_queue = getattr(session, "_tool_event_queue", None)
        if tool_event_queue is not None:
            state["tool_event_queue"] = tool_event_queue
            state["tool_event_task"] = asyncio.create_task(
                self._drain_tool_events(room_name, tool_event_queue)
            )

        # Opening greeting: kick off a synthetic first turn so the agent
        # speaks first ("Hi Alice, this is Acme support…") rather than
        # waiting for the user to start. Disabled by V2V_AGENT_OPENS_CALL=
        # false. Failure here is non-fatal — the call still works, just
        # without an opener.
        if _agent_opens_call() and session is not None:
            state["greeting_task"] = asyncio.create_task(
                self._deliver_opening_greeting(room_name, session, user_id)
            )

    async def _deliver_opening_greeting(
        self,
        room_name: str,
        session: Any,
        user_id: str,
    ) -> None:
        """Drive a synthetic first turn so the agent greets the caller.

        We pass a hint as the user_message — the AIAgent answers naturally
        and the response streams through the existing chunker → TTS path.
        The synthetic user turn IS persisted in history so subsequent
        turns have context (the agent knows it just greeted), but we
        DON'T publish a ``user_message`` data-channel event for it
        (the browser would render it as a bogus user bubble).
        """
        prompt = os.environ.get(
            "V2V_OPENING_GREETING",
            "[CALL_OPENED] The call just connected. Greet the caller "
            "warmly in one short sentence — introduce yourself as the "
            "support assistant, address them by name if you know it from "
            "memory, and ask how you can help. Do not list capabilities.",
        )
        try:
            logger.info("voice_rtc: delivering opening greeting room=%s", room_name)
            token_iter = await session.submit_user_turn(prompt)
            await self.on_assistant_token_stream(
                chat_id=room_name,
                token_iterator=token_iter,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: opening greeting failed for %s", room_name)

    async def _drain_tool_events(
        self,
        room_name: str,
        queue: asyncio.Queue,
    ) -> None:
        try:
            while True:
                kind, name, targs, result = await queue.get()
                if kind == "tool_start":
                    logger.info(
                        "voice_rtc: tool_start room=%s tool=%s args=%r",
                        room_name, name, targs,
                    )
                    try:
                        await self._publish_event(
                            room_name,
                            {"type": "tool_call", "name": name, "args": targs},
                        )
                    except Exception:  # pragma: no cover
                        logger.exception("voice_rtc: publish tool_call failed")
                else:
                    ok = not isinstance(result, Exception)
                    logger.info(
                        "voice_rtc: tool_complete room=%s tool=%s ok=%s",
                        room_name, name, ok,
                    )
                    try:
                        await self._publish_event(
                            room_name,
                            {"type": "tool_result", "name": name, "ok": ok},
                        )
                    except Exception:  # pragma: no cover
                        logger.exception("voice_rtc: publish tool_result failed")
                    if name in ("end_call", "agent_handover"):
                        reason = ""
                        if isinstance(targs, dict):
                            reason = str(targs.get("reason") or "")
                        try:
                            grace_ms = int(os.environ.get("V2V_HANGUP_GRACE_MS", "4000"))
                        except ValueError:
                            grace_ms = 4000
                        asyncio.create_task(
                            self._hangup_call(room_name, name, reason, grace_ms)
                        )
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: tool event drain crashed for %s", room_name)

    async def _hangup_call(
        self,
        room_name: str,
        tool_name: str,
        reason: str,
        grace_ms: int,
    ) -> None:
        """Tear down a LiveKit room after a call-control tool fired.

        Waits ``grace_ms`` so the agent's farewell sentence finishes
        synthesizing, publishes a ``call_ended`` data-channel event, and
        disconnects the room.
        """
        try:
            if grace_ms > 0:
                await asyncio.sleep(grace_ms / 1000.0)
            logger.info("voice_rtc: hangup room=%s reason=%s", room_name, reason or tool_name)
            try:
                await self._publish_event(
                    room_name,
                    {"type": "call_ended", "reason": reason or tool_name, "tool": tool_name},
                )
            except Exception:  # pragma: no cover
                logger.exception("voice_rtc: publish call_ended failed for %s", room_name)
            state = self._active_calls.get(room_name)
            if state is not None:
                tts_task = state.get("tts_task")
                if tts_task is not None and not tts_task.done():
                    tts_task.cancel()
                room = state.get("room")
                if room is not None:
                    for attr in ("disconnect", "aclose", "close"):
                        fn = getattr(room, attr, None)
                        if fn is None:
                            continue
                        try:
                            res = fn()
                            if asyncio.iscoroutine(res):
                                await res
                            break
                        except Exception:  # pragma: no cover
                            logger.warning(
                                "voice_rtc: room.%s() raised for %s", attr, room_name,
                                exc_info=True,
                            )
                self._active_calls.pop(room_name, None)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: hangup failed for %s", room_name)

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

        # Operator-visible log line — at INFO so the gateway log narrates
        # the conversation. Truncate to keep lines readable.
        preview = text if len(text) <= 160 else text[:157] + "..."
        logger.info("voice_rtc: user_final room=%s text=%r", room_name, preview)
        try:
            await self._publish_event(
                room_name,
                {"type": "user_message", "text": text},
            )
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: publish user_message failed")

        try:
            from tools.voice_rtc.state import Event as TurnEvent
            ts = self._active_calls.get(room_name, {}).get("turn_state")
            if ts is not None:
                ts.handle(TurnEvent.USER_FINAL)
        except Exception:  # pragma: no cover
            logger.debug("voice_rtc: turn-state advance failed", exc_info=True)

        # If we have a persistent V2V session for this room, drive the
        # token stream directly into the audio-out pipeline. This bypasses
        # the standard runner pipeline (Hermes AIAgent + send) — see the
        # M5.3 deferral note: the runner has no streaming-from-LiveKit-job
        # hook today.
        session = self._active_calls.get(room_name, {}).get("session")
        if session is not None:
            try:
                await self._feed_to_agent(session, text, room_name)
                return
            except Exception:  # pragma: no cover
                logger.exception(
                    "voice_rtc: agent session feed failed for %s; falling back to runner",
                    room_name,
                )

        # Fallback: synthesise a MessageEvent and run it through the
        # normal runner. Useful both when the session was unavailable
        # and for legacy tests that pre-date the v2v session wiring.
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
        await self.handle_message(event)

    async def _feed_to_agent(
        self,
        session: Any,
        text: str,
        room_name: str,
    ) -> None:
        """Submit a user turn to ``session`` and route its streaming
        tokens through ``on_assistant_token_stream`` (which the M5.3
        audio-out pipeline already consumes)."""
        logger.info(
            "voice_rtc: submit_user_turn room=%s chars=%d",
            room_name, len(text or ""),
        )
        token_iter = await session.submit_user_turn(text)
        await self.on_assistant_token_stream(
            chat_id=room_name,
            token_iterator=token_iter,
        )

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

        Drains incoming audio via ``livekit.rtc.AudioStream`` (constructed
        with ``sample_rate=16000, num_channels=1`` so the SDK resamples
        48 kHz mic input down to our ASR contract) and yields raw bytes.

        Two subscription paths run concurrently to avoid the connect-race:

        * **Future events:** ``room.on('track_subscribed', ...)`` for any
          track that subscribes after we register.
        * **Backfill scan:** by the time this method is called, ``ctx.connect()``
          has already returned and LiveKit (with ``auto_subscribe=1``) has
          subscribed to the user's mic track AND fired the
          ``track_subscribed`` event. Our listener missed it. We compensate
          by walking ``room.remote_participants`` and starting a drain for
          every already-subscribed audio track. Without this scan the
          first call's mic audio is silently dropped.
        """
        room = getattr(ctx, "room", None)
        if room is None:
            return

        from livekit import rtc as lk_rtc  # type: ignore

        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        readers: list[asyncio.Task] = []
        # Track the (participant_sid, track_sid) pairs we've already started
        # a drain for, so the future-events listener doesn't double-subscribe
        # tracks the backfill scan picked up first.
        drained_keys: set[tuple[str, str]] = set()

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

        def _is_audio_kind(track_or_pub: Any) -> bool:
            kind = getattr(track_or_pub, "kind", None)
            try:
                audio_kind = getattr(lk_rtc.TrackKind, "KIND_AUDIO", None)
                if audio_kind is not None and kind == audio_kind:
                    return True
            except Exception:
                pass
            # Permissive fallbacks: numeric 1 or string "audio" or a
            # MagicMock placeholder used by unit tests.
            return kind == 1 or kind == "audio" or kind is None

        def _start_drain(track: Any, participant: Any) -> None:
            p_sid = getattr(participant, "sid", "") or ""
            t_sid = getattr(track, "sid", "") or ""
            key = (p_sid, t_sid)
            if key in drained_keys:
                return
            drained_keys.add(key)
            logger.info(
                "voice_rtc: subscribing to audio track p=%s t=%s",
                p_sid or "?", t_sid or "?",
            )
            readers.append(asyncio.create_task(_drain_track(track)))

        def _on_track_subscribed(track, publication, participant) -> None:
            if not _is_audio_kind(track):
                return
            _start_drain(track, participant)

        room.on("track_subscribed", _on_track_subscribed)

        # Backfill: pick up any tracks LiveKit auto-subscribed during
        # ``ctx.connect()`` — that event already fired before we got here.
        try:
            participants_view = getattr(room, "remote_participants", None) or {}
            participants = (
                participants_view.values()
                if hasattr(participants_view, "values")
                else list(participants_view)
            )
            for participant in participants:
                pubs = getattr(participant, "track_publications", None) or {}
                pub_iter = pubs.values() if hasattr(pubs, "values") else list(pubs)
                for pub in pub_iter:
                    track = getattr(pub, "track", None)
                    if track is None:
                        continue
                    if not _is_audio_kind(pub) and not _is_audio_kind(track):
                        continue
                    _start_drain(track, participant)
        except Exception:  # pragma: no cover
            logger.exception("voice_rtc: backfill scan over remote_participants failed")

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
        state["assistant_buffer"] = []

        # The pipeline body — runs as a task so a barge-in can cancel us.
        async def _pipeline() -> None:
            tts = self._ensure_tts_stream(state)
            first_token_seen = False
            try:
                async for delta in token_iterator:
                    if not delta:
                        continue
                    if not first_token_seen:
                        first_token_seen = True
                        logger.info("voice_rtc: first_token room=%s", room_name)
                    for chunk in chunker.feed(delta):
                        state["assistant_buffer"].append(chunk)
                        await self._publish_event(
                            room_name,
                            {"type": "assistant_chunk", "text": chunk},
                        )
                        await self._synth_and_publish(chunk, state, tts)
                for chunk in chunker.flush():
                    state["assistant_buffer"].append(chunk)
                    await self._publish_event(
                        room_name,
                        {"type": "assistant_chunk", "text": chunk},
                    )
                    await self._synth_and_publish(chunk, state, tts)
                full_text = "".join(state["assistant_buffer"])
                await self._publish_event(
                    room_name,
                    {"type": "assistant_done", "text": full_text},
                )
                logger.info(
                    "voice_rtc: assistant_done room=%s chars=%d",
                    room_name, len(full_text),
                )
                # Successful end of turn — advance FSM if we were SPEAKING.
                ts = state.get("turn_state")
                if ts is not None:
                    ts.handle(TurnEvent.TTS_DONE)
            except asyncio.CancelledError:
                try:
                    await self._publish_event(
                        room_name,
                        {
                            "type": "assistant_cancelled",
                            "text": "".join(state.get("assistant_buffer", [])),
                        },
                    )
                except Exception:  # pragma: no cover
                    logger.debug("voice_rtc: publish assistant_cancelled failed", exc_info=True)
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
            # Inter-clause silence: agents that emit clause-by-clause TTS
            # otherwise sound breathless because each clause's audio runs
            # straight into the next. Pad with zeros — sentence boundaries
            # get a longer gap than mid-sentence clauses.
            silence_ms = _silence_after(text)
            if silence_ms > 0:
                await _flush_silence(_flush_frame, silence_ms)
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

        logger.info(
            "voice_rtc: barge_in room=%s prior_state=%s",
            room_name, ts.state,
        )
        try:
            await self._publish_event(room_name, {"type": "barge_in"})
        except Exception:  # pragma: no cover
            logger.debug("voice_rtc: publish barge_in failed", exc_info=True)

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
