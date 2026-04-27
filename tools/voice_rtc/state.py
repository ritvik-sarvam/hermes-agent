"""Turn-state machine for the streaming voice agent.

Tracks the conversational turn across four states — ``LISTENING``,
``THINKING``, ``SPEAKING``, ``INTERRUPTED`` — driven by VAD events from
ASR and "first audio" / "done" events from TTS.

It is the source of truth for whether to cancel an in-flight TTS
playback when the user starts speaking (barge-in).  The FSM is
deliberately *permissive*: unexpected events are silently ignored rather
than raising, because in a real-time pipeline we routinely see late
events from a previous turn (e.g. a ``TTS_DONE`` arriving after we've
already returned to ``LISTENING`` due to a barge-in).

Pure-Python utility — no async, no SDK calls, no external dependencies.
"""

from __future__ import annotations

from enum import Enum, auto


class Event(Enum):
    USER_FINAL = auto()
    TTS_FIRST_AUDIO = auto()
    TTS_DONE = auto()
    VAD_SPEECH_START = auto()
    CANCEL_DONE = auto()


class TurnState:
    def __init__(self) -> None:
        self.state: str = "LISTENING"
        self.cancel_requested: bool = False

    def handle(self, e: Event) -> None:
        s = self.state
        if s == "LISTENING" and e is Event.USER_FINAL:
            self.state = "THINKING"
        elif s == "THINKING" and e is Event.TTS_FIRST_AUDIO:
            self.state = "SPEAKING"
        elif s == "SPEAKING" and e is Event.TTS_DONE:
            self.state = "LISTENING"
        elif s in ("THINKING", "SPEAKING") and e is Event.VAD_SPEECH_START:
            self.state = "INTERRUPTED"
            self.cancel_requested = True
        elif s == "INTERRUPTED" and e is Event.CANCEL_DONE:
            self.state = "LISTENING"
            self.cancel_requested = False
        # otherwise: no-op (FSM is permissive — unexpected events are
        # silently ignored, e.g. late TTS_DONE from a previous turn).
