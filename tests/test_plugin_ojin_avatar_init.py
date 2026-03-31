"""Tests for AvatarSession constructor and OjinException."""

import os
from unittest.mock import patch

import pytest

from livekit.plugins.ojin import AvatarSession, OjinException


def test_requires_api_key_and_config_id():
    """Missing api_key and config_id (no env) should raise OjinException(retryable=False)."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove any OJIN_ env vars that might be set
        env = {k: v for k, v in os.environ.items() if not k.startswith("OJIN_")}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(OjinException) as exc_info:
                AvatarSession()
            assert exc_info.value.retryable is False


def test_accepts_explicit_args():
    """Passing api_key and config_id explicitly should succeed."""
    session = AvatarSession(api_key="test-key", config_id="test-config")
    assert session._api_key == "test-key"
    assert session._config_id == "test-config"


def test_ws_url_default():
    """Default ws_url should be wss://models.ojin.ai/realtime."""
    session = AvatarSession(api_key="test-key", config_id="test-config")
    assert session._ws_url == "wss://models.ojin.ai/realtime"


def test_ws_url_custom():
    """Custom ws_url should be used when provided."""
    session = AvatarSession(
        api_key="test-key", config_id="test-config", ws_url="wss://custom.example.com"
    )
    assert session._ws_url == "wss://custom.example.com"


def test_env_fallback():
    """api_key and config_id should fall back to env vars."""
    with patch.dict(
        os.environ, {"OJIN_API_KEY": "env-key", "OJIN_CONFIG_ID": "env-config"}, clear=False
    ):
        session = AvatarSession()
        assert session._api_key == "env-key"
        assert session._config_id == "env-config"


def test_ojin_exception_attributes():
    """OjinException should carry retryable, code, and origin."""
    exc = OjinException("test error", retryable=True, code="TEST", origin="unit_test")
    assert str(exc) == "test error"
    assert exc.retryable is True
    assert exc.code == "TEST"
    assert exc.origin == "unit_test"
