"""Per-user session registry for the voice_rtc adapter (Milestone 6.2).

A single :class:`SessionRegistry` instance lives on each
:class:`gateway.platforms.voice_rtc.VoiceRTCAdapter` and caches the
``V2VAgentSession`` (or any session-shaped dict / object) for each
``user_id``. Identical user_ids share the underlying session across
calls, so a user who hangs up and immediately calls back re-attaches to
the same agent state.

The registry is intentionally agnostic about what's *inside* a session
— it just calls a user-supplied async ``factory(user_id)`` to create
one and (if the session has an ``async close()``) calls it on eviction.

Lifecycle:

* ``get_or_create(user_id)`` returns the cached session if any (and
  bumps its in-use refcount), otherwise constructs a fresh one via the
  factory.
* ``release(user_id)`` decrements the refcount; when the count reaches
  zero the session is marked idle with the current monotonic time.
* ``evict_idle()`` reaps any sessions whose idle window exceeds
  ``idle_ttl_seconds``. Active sessions (refcount > 0) are never evicted.
* ``close()`` disposes every session unconditionally.

All public methods are coroutine functions. Internal mutation is guarded
by an :class:`asyncio.Lock` so concurrent connect/disconnect from the
LiveKit worker can't race the eviction sweeper.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


SessionFactory = Callable[[str], Awaitable[Any]]


@dataclass
class _Entry:
    session: Any
    in_use: int = 0
    last_release: float = 0.0  # monotonic timestamp; 0 ⇒ never released


class SessionRegistry:
    """Cache of per-user agent sessions with idle-TTL eviction.

    Parameters
    ----------
    factory:
        Async callable taking ``user_id`` and returning the session
        object (or dict) to cache. Called once per user_id while the
        session is live.
    idle_ttl_seconds:
        After ``release()`` drops the in-use count to zero, the session
        stays cached for at least this many seconds before being eligible
        for eviction by ``evict_idle()``.
    time_fn:
        Optional clock override. Defaults to :func:`time.monotonic`. Tests
        inject a deterministic clock so they don't have to actually wait.
    """

    def __init__(
        self,
        *,
        factory: SessionFactory,
        idle_ttl_seconds: float,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self._factory = factory
        self._idle_ttl = float(idle_ttl_seconds)
        self._now = time_fn or time.monotonic
        self._entries: Dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_create(self, user_id: str) -> Any:
        """Return the cached session for ``user_id`` or build a new one.

        Bumps the in-use refcount; pair with :meth:`release` when the
        caller is done.
        """
        # Fast path: already cached.
        async with self._lock:
            entry = self._entries.get(user_id)
            if entry is not None:
                entry.in_use += 1
                return entry.session

        # Slow path: construct outside the lock so a slow factory (which
        # might itself take I/O) doesn't serialise unrelated user_ids.
        session = await self._factory(user_id)

        async with self._lock:
            # Another caller may have raced us; if so, drop our session
            # and use theirs. The orphan ``session`` we just built is
            # disposed if it has a close() method.
            existing = self._entries.get(user_id)
            if existing is not None:
                existing.in_use += 1
                await _maybe_close(session)
                return existing.session
            self._entries[user_id] = _Entry(session=session, in_use=1)
            return session

    async def release(self, user_id: str) -> None:
        """Decrement the in-use refcount; mark idle when it hits zero."""
        async with self._lock:
            entry = self._entries.get(user_id)
            if entry is None:
                return
            if entry.in_use > 0:
                entry.in_use -= 1
            if entry.in_use == 0:
                entry.last_release = self._now()

    async def evict_idle(self) -> None:
        """Drop sessions idle for longer than ``idle_ttl_seconds``.

        Active sessions (in_use > 0) are never evicted regardless of
        ``last_release``. Each evicted session has ``close()`` awaited
        if it exposes one.
        """
        now = self._now()
        to_close: list[tuple[str, Any]] = []
        async with self._lock:
            for user_id, entry in list(self._entries.items()):
                if entry.in_use > 0:
                    continue
                if entry.last_release == 0.0:
                    # Never released — should not happen if in_use==0,
                    # but be defensive.
                    continue
                if now - entry.last_release >= self._idle_ttl:
                    to_close.append((user_id, entry.session))
                    self._entries.pop(user_id, None)

        for user_id, session in to_close:
            try:
                await _maybe_close(session)
            except Exception:  # pragma: no cover
                logger.exception(
                    "voice_rtc.sessions: close() raised for user %s during eviction",
                    user_id,
                )

    async def close(self) -> None:
        """Dispose every session unconditionally and clear the registry."""
        async with self._lock:
            entries = list(self._entries.items())
            self._entries.clear()
        for user_id, entry in entries:
            try:
                await _maybe_close(entry.session)
            except Exception:  # pragma: no cover
                logger.exception(
                    "voice_rtc.sessions: close() raised for user %s during shutdown",
                    user_id,
                )

    # ------------------------------------------------------------------
    # Introspection (for tests / metrics)
    # ------------------------------------------------------------------

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)


async def _maybe_close(session: Any) -> None:
    """Call ``session.close()`` if it exposes one and is awaitable.

    Sessions can be either dicts (with a ``close`` callable under that
    key) or objects with a bound ``close`` method. We accept both so the
    registry stays decoupled from the V2VAgentSession concrete shape.
    """
    closer = None
    if isinstance(session, dict):
        closer = session.get("close")
    else:
        closer = getattr(session, "close", None)

    if closer is None:
        return
    try:
        result = closer()
        if asyncio.iscoroutine(result):
            await result
    except Exception:  # pragma: no cover
        logger.exception("voice_rtc.sessions: session close() raised")


__all__ = ["SessionRegistry", "SessionFactory"]
