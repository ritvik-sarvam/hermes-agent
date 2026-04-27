"""Tests for Sarvam LLM provider registration.

Sarvam exposes an OpenAI-compatible chat completions endpoint
(https://api.sarvam.ai/v1) authenticated via a bearer token. We register
it as an ``openai_chat`` overlay so Hermes can reach it through the
existing OpenAI-style transport.

Only the minimal contract the v2v voice agent depends on is covered here:
the auth-key registry entry, the providers.py overlay, and the static
model list (which the model picker reads).
"""

from __future__ import annotations

import pytest


class TestSarvamAuthRegistry:
    """``hermes_cli.auth.PROVIDER_REGISTRY`` must know about ``sarvam``."""

    def test_registered(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert "sarvam" in PROVIDER_REGISTRY

    def test_auth_type_is_api_key(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert PROVIDER_REGISTRY["sarvam"].auth_type == "api_key"

    def test_inference_base_url(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert PROVIDER_REGISTRY["sarvam"].inference_base_url == "https://api.sarvam.ai/v1"

    def test_api_key_env_vars(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert PROVIDER_REGISTRY["sarvam"].api_key_env_vars == ("SARVAM_API_KEY",)

    def test_base_url_env_var(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert PROVIDER_REGISTRY["sarvam"].base_url_env_var == "SARVAM_LLM_BASE_URL"


class TestSarvamProvidersOverlay:
    """``hermes_cli.providers.HERMES_OVERLAYS`` must register sarvam as openai_chat."""

    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "sarvam" in HERMES_OVERLAYS

    def test_transport_is_openai_chat(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert HERMES_OVERLAYS["sarvam"].transport == "openai_chat"

    def test_base_url_env_var(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert HERMES_OVERLAYS["sarvam"].base_url_env_var == "SARVAM_LLM_BASE_URL"

    def test_base_url_override_points_at_sarvam_api(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        # Sarvam isn't in models.dev, so the overlay must carry the URL itself.
        assert HERMES_OVERLAYS["sarvam"].base_url_override == "https://api.sarvam.ai/v1"

    def test_api_key_env_var_in_overlay(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        # The providers overlay tracks "extra" env vars beyond models.dev. Since
        # Sarvam isn't in models.dev, SARVAM_API_KEY needs to live here for
        # ``get_provider`` to surface it on the resolved ProviderDef.
        assert "SARVAM_API_KEY" in HERMES_OVERLAYS["sarvam"].extra_env_vars

    def test_get_provider_resolves(self):
        from hermes_cli.providers import get_provider

        pdef = get_provider("sarvam")
        assert pdef is not None
        assert pdef.id == "sarvam"
        assert pdef.transport == "openai_chat"
        assert pdef.base_url == "https://api.sarvam.ai/v1"
        assert "SARVAM_API_KEY" in pdef.api_key_env_vars

    def test_api_mode_is_chat_completions(self):
        from hermes_cli.providers import HERMES_OVERLAYS, TRANSPORT_TO_API_MODE

        overlay = HERMES_OVERLAYS["sarvam"]
        assert TRANSPORT_TO_API_MODE[overlay.transport] == "chat_completions"


class TestSarvamModelCatalog:
    """The static model picker fallback must list sarvam-m as the default."""

    def test_known_models(self):
        from hermes_cli.models import _PROVIDER_MODELS

        assert "sarvam" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["sarvam"]
        assert "sarvam-m" in models
        # sarvam-m is the only voice-suitable model. The reasoning models
        # (sarvam-30b, sarvam-105b) stream into delta.reasoning_content with
        # multi-second TTFB — wrong for voice — but we list them so the picker
        # surfaces them for non-voice use.
        assert "sarvam-30b" in models
        assert "sarvam-105b" in models

    def test_default_model_is_sarvam_m(self):
        from hermes_cli.models import _PROVIDER_MODELS

        # Convention used elsewhere in this file: index 0 is the default.
        assert _PROVIDER_MODELS["sarvam"][0] == "sarvam-m"


class TestSarvamCredentials:
    """Auth helpers should resolve the Sarvam key + base URL correctly."""

    def test_resolve_credentials_default_url(self, monkeypatch):
        from hermes_cli.auth import resolve_api_key_provider_credentials

        monkeypatch.setenv("SARVAM_API_KEY", "test-sarvam-key-12345678")
        monkeypatch.delenv("SARVAM_LLM_BASE_URL", raising=False)

        creds = resolve_api_key_provider_credentials("sarvam")
        assert creds["api_key"] == "test-sarvam-key-12345678"
        assert creds["base_url"] == "https://api.sarvam.ai/v1"

    def test_resolve_credentials_custom_url(self, monkeypatch):
        from hermes_cli.auth import resolve_api_key_provider_credentials

        monkeypatch.setenv("SARVAM_API_KEY", "test-sarvam-key-12345678")
        monkeypatch.setenv("SARVAM_LLM_BASE_URL", "https://staging.sarvam.example/v1")

        creds = resolve_api_key_provider_credentials("sarvam")
        assert creds["base_url"] == "https://staging.sarvam.example/v1"
