"""Skeleton tests for the voice_rtc platform adapter (Task 5.1).

Verifies:
- Platform.VOICE_RTC enum entry is wired
- check_voice_rtc_requirements() returns True with livekit deps installed
- VoiceRTCAdapter constructs from a PlatformConfig and connect()/disconnect()
  toggle the connected flag without raising
- Gateway runner's _create_adapter() returns a VoiceRTCAdapter for
  Platform.VOICE_RTC

Tests run with no env vars set; nothing here touches the network.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


def test_platform_enum_entry_exists():
    assert Platform.VOICE_RTC.value == "voice_rtc"


def test_check_voice_rtc_requirements_true_when_livekit_installed():
    from gateway.platforms.voice_rtc import check_voice_rtc_requirements
    # The hermes-agent fork installs livekit-agents/api/livekit at dev time
    # (Path A: declared as a `voice_rtc` extra in pyproject).
    assert check_voice_rtc_requirements() is True


def test_adapter_constructs_without_env_vars(monkeypatch):
    # Strip any LiveKit/Sarvam env that might be set on dev machines.
    for var in (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "SARVAM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    from gateway.platforms.voice_rtc import VoiceRTCAdapter

    config = PlatformConfig(enabled=True, extra={})
    adapter = VoiceRTCAdapter(config)
    assert adapter.platform is Platform.VOICE_RTC
    assert adapter._lk_url == ""
    assert adapter._lk_api_key == ""
    assert adapter._lk_api_secret == ""


def test_adapter_reads_credentials_from_extra(monkeypatch):
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    from gateway.platforms.voice_rtc import VoiceRTCAdapter

    config = PlatformConfig(
        enabled=True,
        extra={
            "url": "wss://example.livekit.cloud",
            "api_key": "ak_x",
            "api_secret": "sk_x",
        },
    )
    adapter = VoiceRTCAdapter(config)
    assert adapter._lk_url == "wss://example.livekit.cloud"
    assert adapter._lk_api_key == "ak_x"
    assert adapter._lk_api_secret == "sk_x"


def test_connect_disconnect_clean(monkeypatch):
    """connect() marks connected; disconnect() marks disconnected — without
    actually starting a LiveKit worker."""
    from gateway.platforms.voice_rtc import VoiceRTCAdapter

    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    adapter = VoiceRTCAdapter(PlatformConfig(enabled=True, extra={}))

    async def _go():
        # Patch the worker-startup hook so connect() does not try to
        # contact a real LiveKit server.
        adapter._start_worker = MagicMock()  # type: ignore[method-assign]
        ok = await adapter.connect()
        assert ok is True
        assert adapter.is_connected is True
        await adapter.disconnect()
        assert adapter.is_connected is False

    asyncio.run(_go())


def test_runner_create_adapter_returns_voice_rtc():
    """gateway.run.GatewayRunner._create_adapter routes Platform.VOICE_RTC
    to VoiceRTCAdapter."""
    from gateway.run import GatewayRunner
    from gateway.platforms.voice_rtc import VoiceRTCAdapter

    # Bypass full GatewayRunner __init__ — _create_adapter only reads
    # ``self.config`` for a couple of optional defaults.
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = False
    runner.config.thread_sessions_per_user = False

    config = PlatformConfig(enabled=True, extra={})
    adapter = runner._create_adapter(Platform.VOICE_RTC, config)
    assert isinstance(adapter, VoiceRTCAdapter)
