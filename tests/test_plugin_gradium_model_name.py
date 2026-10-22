"""Gradium TTS and STT report the configured model name in metrics and traces."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_tts_model_reports_model_name():
    from livekit.plugins.gradium import TTS

    assert TTS(api_key="test-key", model_name="my-model").model == "my-model"
    assert TTS(api_key="test-key").model == "default"


def test_stt_model_reports_model_name():
    from livekit.plugins.gradium import STT

    assert STT(api_key="test-key", model_name="my-model").model == "my-model"
    assert STT(api_key="test-key").model == "default"
