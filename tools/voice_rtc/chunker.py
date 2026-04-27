"""Clause chunker for streaming LLM output → TTS.

Sits between the LLM token stream and the TTS streamer.  Buffers incoming
``delta`` strings and emits text at "natural" boundaries so TTS can start
synthesizing audio quickly without waiting for the full LLM response.

Boundary rules (in order of precedence within a single ``feed`` call):

* Hard punctuation ``. ! ? ; \\n`` — emit immediately.
* Soft punctuation ``,`` — emit only when the current buffer prefix length
  has reached ``min_chars_for_comma``.
* Token cap — when the whitespace-token count of the buffer reaches
  ``max_tokens``, emit at the next whitespace boundary.

Additionally, ``<think>...</think>`` blocks emitted by reasoning models
(notably ``sarvam-m``) are stripped: text inside the block never reaches
the output, and emission boundary detection ignores characters inside the
block.  The opening / closing tags may be split across multiple
``feed()`` calls (e.g. ``<thi`` then ``nk>``) — partial tag matching is
handled.  If a ``<think>`` opens and is never closed, the rest of the
stream is suppressed and ``flush()`` returns ``[]``.

Pure-Python utility — no async, no SDK calls, no external dependencies.
"""

from __future__ import annotations

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
HARD_PUNCT = set(".!?;\n")


class ClauseChunker:
    def __init__(self, min_chars_for_comma: int = 40, max_tokens: int = 80) -> None:
        self.min_chars_for_comma = min_chars_for_comma
        self.max_tokens = max_tokens

        # Visible buffer (text the user will eventually hear), accumulated
        # between emission boundaries.
        self._buf: str = ""
        # Pending tail bytes that *might* be the start of a tag, held back
        # from ``_buf`` until disambiguated.  E.g. seeing "<" alone, we don't
        # know yet if it begins ``<think>``; keep it pending.
        self._pending: str = ""
        # True while we are inside a ``<think>...</think>`` block (everything
        # is suppressed).
        self._in_think: bool = False

    # ------------------------------------------------------------------ API

    def feed(self, delta: str) -> list[str]:
        """Consume an incremental piece of LLM output, return any chunks
        ready to be sent to TTS."""
        out: list[str] = []
        for ch in delta:
            self._consume_char(ch, out)
        return out

    def flush(self) -> list[str]:
        """Emit any remaining buffered text (called at end-of-stream).

        If we are still inside an unterminated ``<think>`` block, suppress
        everything and return ``[]``.
        """
        if self._in_think:
            # Unterminated reasoning block — drop everything.
            self._buf = ""
            self._pending = ""
            return []

        # Pending bytes that never resolved into a tag are real content.
        if self._pending:
            self._buf += self._pending
            self._pending = ""

        remainder = self._buf.strip()
        self._buf = ""
        return [remainder] if remainder else []

    # --------------------------------------------------------------- internal

    def _consume_char(self, ch: str, out: list[str]) -> None:
        """Process a single character, possibly producing emissions."""
        if self._in_think:
            # Looking for the end tag.  ``_pending`` holds a partial match
            # of THINK_CLOSE.
            candidate = self._pending + ch
            if THINK_CLOSE.startswith(candidate):
                if candidate == THINK_CLOSE:
                    # Closing tag fully matched — exit think mode.
                    self._in_think = False
                    self._pending = ""
                else:
                    self._pending = candidate
            else:
                # Mismatch — drop pending (it was inside the block anyway)
                # and re-evaluate ``ch`` from scratch (still inside think).
                self._pending = ""
                if THINK_CLOSE.startswith(ch):
                    self._pending = ch
                # else: just discard the character.
            return

        # Not inside a think block — watch for an opening tag.
        candidate = self._pending + ch
        if THINK_OPEN.startswith(candidate):
            if candidate == THINK_OPEN:
                # Entered a think block.  Drop the buffered tag.
                self._in_think = True
                self._pending = ""
            else:
                # Still a possible prefix of "<think>".  Hold back.
                self._pending = candidate
            return

        # ``candidate`` is not a prefix of "<think>".  Whatever was held
        # in ``_pending`` is real content; flush it into ``_buf`` (no
        # boundary characters can appear inside a candidate prefix that
        # starts with '<', so no special handling needed there), then
        # process ``ch`` itself.
        if self._pending:
            self._buf += self._pending
            self._pending = ""

        # ``ch`` itself might still start a fresh prefix of "<think>".
        if THINK_OPEN.startswith(ch):
            self._pending = ch
            return

        # Plain content character.
        self._buf += ch
        self._maybe_emit(out)

    def _maybe_emit(self, out: list[str]) -> None:
        """Check the most recent character against boundary rules."""
        if not self._buf:
            return
        last = self._buf[-1]

        if last in HARD_PUNCT:
            self._emit(out)
            return

        if last == ",":
            # Length of the buffer up to (but not including) the comma is
            # the "prefix" — the rule says emit if that prefix is long
            # enough.  We include the comma in the emitted chunk.
            prefix_len = len(self._buf) - 1
            if prefix_len >= self.min_chars_for_comma:
                self._emit(out)
            return

        # Token-cap rule: at a whitespace boundary, count tokens so far.
        if last.isspace():
            token_count = len(self._buf.split())
            if token_count >= self.max_tokens:
                self._emit(out)
            return

    def _emit(self, out: list[str]) -> None:
        chunk = self._buf.strip()
        self._buf = ""
        if chunk:
            out.append(chunk)
