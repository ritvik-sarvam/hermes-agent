"""Tests for ``gateway.platforms.voice_rtc_sessions.SessionRegistry`` (Task 6.2).

The registry is keyed by ``user_id``: identical user_id requests share a
single underlying session object across calls, releases mark a session
idle but keep it cached for an idle-TTL window so quick reconnects
re-attach to the same agent state, and ``evict_idle()`` reaps anything
older than the TTL. ``close()`` disposes everything synchronously.

Construction of the actual ``V2VAgentSession`` is delegated to a factory
callable injected at construction — the registry doesn't know or care
what's inside the session dict.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.platforms.voice_rtc_sessions import SessionRegistry


@pytest.mark.asyncio
async def test_get_or_create_returns_same_session_for_same_user():
    factory_calls = []

    async def factory(user_id):
        factory_calls.append(user_id)
        return {"user_id": user_id, "agent": object()}

    reg = SessionRegistry(factory=factory, idle_ttl_seconds=600)
    s1 = await reg.get_or_create("u1")
    s2 = await reg.get_or_create("u1")
    assert s1 is s2
    assert factory_calls == ["u1"]


@pytest.mark.asyncio
async def test_different_users_get_different_sessions():
    async def factory(user_id):
        return {"user_id": user_id, "agent": object()}

    reg = SessionRegistry(factory=factory, idle_ttl_seconds=600)
    s1 = await reg.get_or_create("u1")
    s2 = await reg.get_or_create("u2")
    assert s1 is not s2


@pytest.mark.asyncio
async def test_release_marks_idle_but_doesnt_drop_immediately():
    async def factory(user_id):
        return {"user_id": user_id}

    reg = SessionRegistry(factory=factory, idle_ttl_seconds=600)
    s = await reg.get_or_create("u1")
    await reg.release("u1")
    again = await reg.get_or_create("u1")
    assert again is s  # still cached during idle window


@pytest.mark.asyncio
async def test_idle_eviction_after_ttl():
    times = [1000.0]

    async def factory(user_id):
        return {"user_id": user_id}

    reg = SessionRegistry(
        factory=factory,
        idle_ttl_seconds=600,
        time_fn=lambda: times[0],
    )
    await reg.get_or_create("u1")
    await reg.release("u1")
    times[0] = 1700.0  # 700s later, past TTL
    await reg.evict_idle()

    # Next get_or_create should construct a fresh session.
    factory_called = []

    async def factory2(user_id):
        factory_called.append(user_id)
        return {"user_id": user_id}

    reg._factory = factory2  # swap factory to detect re-construction
    fresh = await reg.get_or_create("u1")
    assert factory_called == ["u1"]
    assert fresh["user_id"] == "u1"


@pytest.mark.asyncio
async def test_idle_eviction_does_not_evict_active_sessions():
    times = [1000.0]

    async def factory(user_id):
        return {"user_id": user_id}

    reg = SessionRegistry(factory=factory, idle_ttl_seconds=600, time_fn=lambda: times[0])
    s1 = await reg.get_or_create("u1")
    # Don't release — stay active.
    times[0] = 9999.0
    await reg.evict_idle()
    again = await reg.get_or_create("u1")
    assert again is s1


@pytest.mark.asyncio
async def test_close_disposes_all_sessions():
    closed = []

    async def factory(user_id):
        async def close():
            closed.append(user_id)

        return {"user_id": user_id, "close": close}

    reg = SessionRegistry(factory=factory, idle_ttl_seconds=600)
    await reg.get_or_create("u1")
    await reg.get_or_create("u2")
    await reg.close()
    assert sorted(closed) == ["u1", "u2"]


@pytest.mark.asyncio
async def test_evict_idle_calls_close():
    closed = []

    async def factory(user_id):
        async def close():
            closed.append(user_id)

        return {"user_id": user_id, "close": close}

    times = [1000.0]
    reg = SessionRegistry(factory=factory, idle_ttl_seconds=10, time_fn=lambda: times[0])
    await reg.get_or_create("u1")
    await reg.release("u1")
    times[0] = 2000.0
    await reg.evict_idle()
    assert closed == ["u1"]


@pytest.mark.asyncio
async def test_in_use_count_prevents_eviction_during_concurrent_calls():
    """If user holds two simultaneous calls, releasing one must not
    cause eviction — the other call still references the session."""
    times = [1000.0]

    async def factory(user_id):
        return {"user_id": user_id}

    reg = SessionRegistry(factory=factory, idle_ttl_seconds=10, time_fn=lambda: times[0])
    s1 = await reg.get_or_create("u1")
    s2 = await reg.get_or_create("u1")
    assert s1 is s2

    await reg.release("u1")  # one call gone, one still using

    times[0] = 9999.0
    await reg.evict_idle()  # in-use → not evicted

    again = await reg.get_or_create("u1")
    assert again is s1
