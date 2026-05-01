"""Tests for ElevenLabs STT plugin configuration options."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents.types import NOT_GIVEN
from livekit.plugins.elevenlabs import STT


def test_keyterms_default():
    stt = STT(api_key="test-key")
    assert stt._opts.keyterms is NOT_GIVEN


def test_keyterms_set():
    stt = STT(api_key="test-key", keyterms=["LiveKit", "Scribe"])
    assert stt._opts.keyterms == ["LiveKit", "Scribe"]


def test_keyterms_update():
    stt = STT(api_key="test-key")
    stt.update_options(keyterms=["foo", "bar"])
    assert stt._opts.keyterms == ["foo", "bar"]


def test_no_verbatim_default():
    stt = STT(api_key="test-key")
    assert stt._opts.no_verbatim is NOT_GIVEN


def test_no_verbatim_set():
    stt = STT(api_key="test-key", no_verbatim=True)
    assert stt._opts.no_verbatim is True


def test_no_verbatim_update():
    stt = STT(api_key="test-key")
    stt.update_options(no_verbatim=False)
    assert stt._opts.no_verbatim is False


def test_enable_logging_default():
    stt = STT(api_key="test-key")
    assert stt._opts.enable_logging is NOT_GIVEN


def test_enable_logging_set():
    stt = STT(api_key="test-key", enable_logging=False)
    assert stt._opts.enable_logging is False


def test_enable_logging_update():
    stt = STT(api_key="test-key")
    stt.update_options(enable_logging=True)
    assert stt._opts.enable_logging is True


def test_combined_options():
    stt = STT(
        api_key="test-key",
        keyterms=["alpha", "beta"],
        no_verbatim=True,
        enable_logging=False,
    )
    assert stt._opts.keyterms == ["alpha", "beta"]
    assert stt._opts.no_verbatim is True
    assert stt._opts.enable_logging is False


def _make_session_mock(captured: dict[str, str]) -> MagicMock:
    async def fake_ws_connect(url: str, **kwargs):
        captured["url"] = url
        return MagicMock()

    session = MagicMock()
    session.ws_connect = AsyncMock(side_effect=fake_ws_connect)
    return session


@pytest.mark.asyncio
async def test_realtime_query_string_includes_new_params():
    """Verify keyterms, no_verbatim, and enable_logging appear in the realtime WS URL."""
    captured: dict[str, str] = {}
    stt = STT(
        api_key="test-key",
        model_id="scribe_v2_realtime",
        language_code="en",
        keyterms=["hello world", "scribe&v2"],
        no_verbatim=True,
        enable_logging=False,
        http_session=_make_session_mock(captured),
    )
    stream = stt.stream()
    try:
        await stream._connect_ws()
    finally:
        await stream.aclose()

    url = captured["url"]
    assert "keyterms=hello%20world" in url
    assert "keyterms=scribe%26v2" in url
    assert "no_verbatim=true" in url
    assert "enable_logging=false" in url


@pytest.mark.asyncio
async def test_realtime_query_string_omits_unset_params():
    """Unset params should not appear in the realtime WS URL."""
    captured: dict[str, str] = {}
    stt = STT(
        api_key="test-key",
        model_id="scribe_v2_realtime",
        language_code="en",
        http_session=_make_session_mock(captured),
    )
    stream = stt.stream()
    try:
        await stream._connect_ws()
    finally:
        await stream.aclose()

    url = captured["url"]
    assert "keyterms=" not in url
    assert "no_verbatim=" not in url
    assert "enable_logging=" not in url
