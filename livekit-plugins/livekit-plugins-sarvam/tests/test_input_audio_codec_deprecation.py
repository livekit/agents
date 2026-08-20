"""Tests for the deprecated `input_audio_codec` no-op warning.

Sarvam removed server-side support for `input_audio_codec` in January 2026; the
plugin now ignores it and warns exactly when a caller passes it explicitly.
"""

from __future__ import annotations

import logging

import pytest

from livekit.plugins.sarvam.stt import STT

pytestmark = pytest.mark.unit


def test_init_warns_when_input_audio_codec_given(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        STT(api_key="test-key", input_audio_codec="linear16")
    assert any("input_audio_codec" in r.message for r in caplog.records)


def test_init_does_not_warn_when_input_audio_codec_omitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        STT(api_key="test-key")
    assert not any("input_audio_codec" in r.message for r in caplog.records)


async def test_stream_warns_when_input_audio_codec_given(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stt_instance = STT(api_key="test-key")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        stream = stt_instance.stream(input_audio_codec="linear16")
    try:
        assert any("input_audio_codec" in r.message for r in caplog.records)
    finally:
        await stream.aclose()


async def test_stream_does_not_warn_when_input_audio_codec_omitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stt_instance = STT(api_key="test-key")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        stream = stt_instance.stream()
    try:
        assert not any("input_audio_codec" in r.message for r in caplog.records)
    finally:
        await stream.aclose()
