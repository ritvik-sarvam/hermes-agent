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

Additionally, **suppression tags** are stripped: text inside any of the
configured ``(open, close)`` tag pairs never reaches the output, and
emission boundary detection ignores characters inside the block.  Default
suppression set covers reasoning-shape leaks AND tool-call markup, both
of which would otherwise stream into TTS as audible gibberish:

* ``<think>...</think>``        — sarvam-m / smaller reasoning models
* ``<thinking>...</thinking>``  — alt reasoning shape (e.g. Anthropic)
* ``<analysis>...</analysis>``  — alt reasoning shape
* ``<reflection>...</reflection>`` — alt reasoning shape
* ``<tool_call>...</tool_call>`` — when the model decides to invoke a
  tool but we have no execution loop yet (Pravah / sarvam-* OpenAI-compat
  endpoints sometimes emit this even with the voice-mode directive)

Suppression tag opens/closes may be split across multiple ``feed()``
calls (e.g. ``<thi`` then ``nk>``) — partial tag matching is handled.
If a suppression tag opens and is never closed, the rest of the stream
is suppressed and ``flush()`` returns ``[]``.

Pure-Python utility — no async, no SDK calls, no external dependencies.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

# Default tag pairs to suppress. Keep them lowercase; the chunker matches
# verbatim — i.e. ``<Think>`` would NOT match. LLMs have been observed
# emitting only the lowercase forms; if a model surfaces mixed case in
# practice, extend this tuple.
DEFAULT_SUPPRESS_TAGS: Tuple[Tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<analysis>", "</analysis>"),
    ("<reflection>", "</reflection>"),
    ("<tool_call>", "</tool_call>"),
)

HARD_PUNCT = set(".!?;\n")


class ClauseChunker:
    """Streaming clause chunker with suppression-tag stripping."""

    def __init__(
        self,
        min_chars_for_comma: int = 40,
        max_tokens: int = 80,
        suppress_tags: Sequence[Tuple[str, str]] = DEFAULT_SUPPRESS_TAGS,
    ) -> None:
        self.min_chars_for_comma = min_chars_for_comma
        self.max_tokens = max_tokens
        # Pre-compute lookup structures over the configured tag set.
        self._open_tags: Tuple[str, ...] = tuple(o for o, _ in suppress_tags)
        self._close_for: dict[str, str] = {o: c for o, c in suppress_tags}

        # Visible buffer (text the user will eventually hear), accumulated
        # between emission boundaries.
        self._buf: str = ""
        # Pending tail bytes that *might* be the start of a suppression tag,
        # held back from ``_buf`` until disambiguated.  E.g. seeing "<" alone,
        # we don't know yet if it begins ``<think>`` or ``<tool_call>`` or is
        # just a literal "<"; keep it pending.
        self._pending: str = ""
        # When inside a suppressed block this is the open-tag string we're
        # currently in (so we know which close-tag to look for); else ``""``.
        self._suppress_for: str = ""

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

        If we are still inside an unterminated suppression block, drop
        everything and return ``[]``.
        """
        if self._suppress_for:
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
        if self._suppress_for:
            self._consume_inside_suppression(ch)
            return
        self._consume_outside_suppression(ch, out)

    def _consume_inside_suppression(self, ch: str) -> None:
        """We're already inside an open suppression block — look for the
        matching close tag."""
        close = self._close_for[self._suppress_for]
        candidate = self._pending + ch
        if close.startswith(candidate):
            if candidate == close:
                # Fully matched — exit the suppression block.
                self._suppress_for = ""
                self._pending = ""
            else:
                self._pending = candidate
        else:
            # Mismatch — drop pending (it was inside the block anyway)
            # and re-evaluate ``ch`` from scratch (still inside).
            self._pending = ""
            if close.startswith(ch):
                self._pending = ch
            # else: discard the character.

    def _consume_outside_suppression(self, ch: str, out: list[str]) -> None:
        """We're outside any suppression — watch for an opening tag while
        also feeding ``_buf`` and emitting at boundaries."""
        candidate = self._pending + ch
        # If ``candidate`` is a prefix of ANY known open tag, hold it back.
        # If it equals one exactly, enter that suppression block.
        matching_prefix = self._tag_prefix_match(candidate)
        if matching_prefix is not None:
            full_match = matching_prefix == candidate and candidate in self._open_tags
            if full_match:
                self._suppress_for = candidate
                self._pending = ""
            else:
                self._pending = candidate
            return

        # ``candidate`` is not a prefix of any open tag.  Whatever was held
        # in ``_pending`` is real content; flush it, then process ``ch``.
        if self._pending:
            self._buf += self._pending
            self._pending = ""
            # Note: the held-back text could not have contained a boundary
            # punctuation since all tag prefixes start with "<".

        # ``ch`` itself might still start a fresh prefix of an open tag.
        if self._tag_prefix_match(ch) is not None:
            self._pending = ch
            return

        # Plain content character.
        self._buf += ch
        self._maybe_emit(out)

    def _tag_prefix_match(self, s: str) -> str | None:
        """Return the longest open-tag string that ``s`` is a prefix of, or
        ``None`` if ``s`` is not a prefix of any open tag.

        We return *some* matched prefix (deterministic order: the input ``s``
        itself if any open tag starts with it). The caller only needs to know
        "is this a possible tag-start? and if so is it a complete open tag?"
        """
        for opener in self._open_tags:
            if opener.startswith(s):
                return s
        return None

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
