"""Tests for Rime TTS plugin defaults."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.plugin("rime")


def test_default_model_and_speaker() -> None:
    from livekit.plugins.rime import TTS

    tts = TTS(api_key="test-key")

    assert tts.model == "coda"
    assert tts._opts.speaker == "wawona"
