from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.unit


def _make_frame(byte: int = 0x11, samples: int = 160, sample_rate: int = 16000) -> rtc.AudioFrame:
    data = bytes([byte, byte]) * samples
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


def _make_recognition() -> AudioRecognition:
    """Build an AudioRecognition stub with just the attributes ``push_audio`` reads."""
    ar = object.__new__(AudioRecognition)
    ar._sample_rate = None  # type: ignore[attr-defined]
    ar._stt_pipeline = MagicMock()  # type: ignore[attr-defined]
    # the input anchor lives on the pipeline (see _STTPipeline.input_started_at)
    ar._stt_pipeline.input_started_at = None  # type: ignore[attr-defined]
    ar._vad_ch = MagicMock()  # type: ignore[attr-defined]
    ar._interruption_ch = MagicMock()  # type: ignore[attr-defined]
    ar._session = MagicMock()  # type: ignore[attr-defined]
    ar._turn_detector_stream = None  # type: ignore[attr-defined]
    return ar


def test_push_audio_routes_real_frame_everywhere_by_default() -> None:
    ar = _make_recognition()
    frame = _make_frame()

    ar._push_audio(frame)

    ar._stt_pipeline.audio_ch.send_nowait.assert_called_once_with(frame)
    ar._vad_ch.send_nowait.assert_called_once_with(frame)
    ar._interruption_ch.send_nowait.assert_called_once_with(frame)


def test_push_audio_substitutes_stt_frame() -> None:
    ar = _make_recognition()
    real = _make_frame(byte=0x11)
    silence = _make_frame(byte=0x00)

    ar._push_audio(real, stt_frame=silence)

    ar._stt_pipeline.audio_ch.send_nowait.assert_called_once_with(silence)
    ar._vad_ch.send_nowait.assert_called_once_with(real)
    ar._interruption_ch.send_nowait.assert_called_once_with(real)


def test_push_audio_skips_optional_consumers_when_unset() -> None:
    ar = _make_recognition()
    ar._stt_pipeline = None  # type: ignore[attr-defined]
    ar._vad_ch = None  # type: ignore[attr-defined]
    ar._interruption_ch = None  # type: ignore[attr-defined]

    # Should not raise even when every downstream consumer is absent.
    ar._push_audio(_make_frame())


def test_push_audio_records_sample_rate_and_input_start() -> None:
    ar = _make_recognition()
    frame = _make_frame(sample_rate=24000)

    ar._push_audio(frame)

    assert ar._sample_rate == 24000  # type: ignore[attr-defined]
    assert ar._input_started_at is not None  # type: ignore[attr-defined]


def _make_activity() -> AgentActivity:
    activity = object.__new__(AgentActivity)
    activity._started = True
    activity._session = MagicMock()
    activity._session.agent_state = "listening"
    activity._session.amd.enabled = True
    activity._session.amd.started = True
    activity._current_speech = None
    activity._rt_session = MagicMock()
    activity._audio_recognition = MagicMock()
    return activity


def test_amd_pre_answer_gate_discards_audio_for_all_consumers() -> None:
    activity = _make_activity()
    activity._session.amd.started = False
    activity.push_audio(_make_frame())
    activity._session.amd.push_audio.assert_not_called()
    activity._audio_recognition._push_audio.assert_not_called()
    activity._rt_session.push_audio.assert_not_called()

    activity._session.amd.started = True
    frame = _make_frame()
    activity.push_audio(frame)
    activity._session.amd.push_audio.assert_called_once_with(frame)
    activity._audio_recognition._push_audio.assert_called_once_with(frame, stt_frame=None)
    activity._rt_session.push_audio.assert_called_once_with(frame)


@pytest.mark.parametrize("guard", ["aec_warmup", "uninterruptible"])
def test_activity_sends_the_same_muted_audio_to_amd_and_session_stt(guard: str) -> None:
    activity = _make_activity()
    if guard == "aec_warmup":
        activity._session.agent_state = "speaking"
        activity._session._aec_warmup_remaining = 1
        activity._session._aec_warmup_timer = object()
    else:
        activity._current_speech = SimpleNamespace(
            done=lambda: False, interrupted=False, allow_interruptions=False
        )
        activity._session.options.interruption = {"discard_audio_if_uninterruptible": True}
    frame = _make_frame()
    activity.push_audio(frame)
    muted = activity._session.amd.push_audio.call_args.args[0]
    assert bytes(muted.data) == bytes(len(frame.data) * frame.data.itemsize)
    activity._audio_recognition._push_audio.assert_called_once_with(frame, stt_frame=muted)
    activity._rt_session.push_audio.assert_called_once_with(muted)
