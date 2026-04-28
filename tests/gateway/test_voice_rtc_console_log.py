"""Tests for the v2v console log handler installer."""

from __future__ import annotations

import io
import logging

import pytest

from gateway.platforms.voice_rtc import _install_console_log_handler


def _remove_v2v_handlers() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_v2v_console", False):
            root.removeHandler(h)


@pytest.fixture(autouse=True)
def _clean_root_handler():
    _remove_v2v_handlers()
    original_level = logging.getLogger().level
    yield
    _remove_v2v_handlers()
    logging.getLogger().setLevel(original_level)


def test_install_console_log_handler_is_idempotent():
    h1 = _install_console_log_handler()
    h2 = _install_console_log_handler()
    assert h1 is h2
    root = logging.getLogger()
    tagged = [h for h in root.handlers if getattr(h, "_v2v_console", False)]
    assert len(tagged) == 1


def test_install_console_log_handler_emits_v2v_records():
    handler = _install_console_log_handler()
    stream = io.StringIO()
    handler.stream = stream

    logger = logging.getLogger("gateway.platforms.voice_rtc")
    logger.info("hello world from v2v")

    out = stream.getvalue()
    assert "hello world from v2v" in out
    assert "gateway.platforms.voice_rtc" in out


def test_install_console_log_handler_filters_unrelated_loggers():
    handler = _install_console_log_handler()
    stream = io.StringIO()
    handler.stream = stream

    logging.getLogger("some.unrelated.logger").info("should be filtered out")

    assert "should be filtered out" not in stream.getvalue()


def test_install_console_log_handler_honors_env_level(monkeypatch):
    monkeypatch.setenv("V2V_CONSOLE_LOG_LEVEL", "DEBUG")
    handler = _install_console_log_handler()
    assert handler.level == logging.DEBUG

    stream = io.StringIO()
    handler.stream = stream

    logger = logging.getLogger("gateway.platforms.voice_rtc")
    logger.debug("debug-level record")
    assert "debug-level record" in stream.getvalue()
