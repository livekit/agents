from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents.utils import aio
from livekit.agents.voice.agent_session import AgentSession
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
    ar._session.amd.push_audio.assert_called_once_with(frame)
    ar._interruption_ch.send_nowait.assert_called_once_with(frame)


def test_push_audio_substitutes_stt_frame_only_on_stt_path() -> None:
    ar = _make_recognition()
    real = _make_frame(byte=0x11)
    silence = _make_frame(byte=0x00)

    ar._push_audio(real, stt_frame=silence)

    # STT pipeline sees the substitute (silence), nothing else does.
    ar._stt_pipeline.audio_ch.send_nowait.assert_called_once_with(silence)
    ar._vad_ch.send_nowait.assert_called_once_with(real)
    ar._session.amd.push_audio.assert_called_once_with(real)
    ar._interruption_ch.send_nowait.assert_called_once_with(real)


def test_push_audio_skips_optional_consumers_when_unset() -> None:
    ar = _make_recognition()
    ar._stt_pipeline = None  # type: ignore[attr-defined]
    ar._vad_ch = None  # type: ignore[attr-defined]
    ar._interruption_ch = None  # type: ignore[attr-defined]
    ar._session.amd = None

    # Should not raise even when every downstream consumer is absent.
    ar._push_audio(_make_frame())


def test_push_audio_records_sample_rate_and_input_start() -> None:
    ar = _make_recognition()
    frame = _make_frame(sample_rate=24000)

    ar._push_audio(frame)

    assert ar._sample_rate == 24000  # type: ignore[attr-defined]
    assert ar._input_started_at is not None  # type: ignore[attr-defined]


def test_push_audio_keeps_vad_when_stt_channel_closed() -> None:
    ar = _make_recognition()
    ar._stt_pipeline.audio_ch.send_nowait.side_effect = aio.ChanClosed
    frame = _make_frame()

    ar._push_audio(frame)

    ar._vad_ch.send_nowait.assert_called_once_with(frame)
    ar._session.amd.push_audio.assert_called_once_with(frame)
    ar._interruption_ch.send_nowait.assert_called_once_with(frame)


async def test_forward_audio_task_survives_push_audio_error() -> None:
    from .fake_io import FakeAudioInput

    session = object.__new__(AgentSession)
    pushed: list[rtc.AudioFrame] = []

    class _Activity:
        def __init__(self) -> None:
            self._n = 0

        def push_audio(self, frame: rtc.AudioFrame) -> None:
            self._n += 1
            if self._n == 1:
                raise RuntimeError("stt fanout failed")
            pushed.append(frame)

    session._activity = _Activity()  # type: ignore[attr-defined]
    audio = FakeAudioInput()
    session._input = SimpleNamespace(audio=audio)  # type: ignore[assignment]

    task = asyncio.create_task(AgentSession._forward_audio_task(session))
    audio.push(0.05)
    audio.push(0.05)
    for _ in range(50):
        if len(pushed) == 1:
            break
        await asyncio.sleep(0)
    audio._audio_ch.close()
    await asyncio.wait_for(task, timeout=2)
    assert len(pushed) == 1
