"""The SDK must attach its VAD for models the provider does not endpoint.

``openai/gpt-live-transcribe`` and ``openai/gpt-realtime-whisper`` reject server
turn detection (see OpenAI's VAD guide), so the inference gateway can only
finalize a turn when the client commits (``session.finalize``). The SDK sends
that commit from its VAD path (``SpeechStream.vad_task``), so it must attach a
VAD for these models. Without one they stream interim transcripts and never
produce a final, so user turns never commit.
"""

from __future__ import annotations

import logging

import pytest

from livekit.agents.inference.stt import STT, _resolve_vad_for_model
from livekit.agents.inference.vad import VAD

pytestmark = pytest.mark.unit


@pytest.fixture
def vad_instance() -> VAD:
    return VAD()


def test_openai_live_transcribe_gets_a_vad():
    assert _resolve_vad_for_model("openai/gpt-live-transcribe", None) is not None


def test_openai_realtime_whisper_gets_a_vad():
    assert _resolve_vad_for_model("openai/gpt-realtime-whisper", None) is not None


def test_speechmatics_rt_still_gets_a_vad():
    assert _resolve_vad_for_model("speechmatics/rt", None) is not None


def test_openai_live_transcribe_keeps_a_caller_vad(vad_instance, caplog):
    with caplog.at_level(logging.WARNING):
        assert _resolve_vad_for_model("openai/gpt-live-transcribe", vad_instance) is vad_instance
    assert "will be ignored" not in caplog.text


def test_server_side_endpointing_models_drop_a_caller_vad(vad_instance, caplog):
    for model in ("deepgram/nova-3", "assemblyai/universal-3-5-pro", "speechmatics/linden-1"):
        with caplog.at_level(logging.WARNING):
            assert _resolve_vad_for_model(model, vad_instance) is None
    assert "will be ignored" in caplog.text


def test_stt_constructor_attaches_a_vad_for_openai_live_transcribe():
    stt = STT(
        model="openai/gpt-live-transcribe",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
    )
    assert stt._vad is not None
