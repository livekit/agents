from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.unit


class _TrackingEvent(asyncio.Event):
    def __init__(self) -> None:
        super().__init__()
        self.wait_calls = 0

    async def wait(self) -> bool:
        self.wait_calls += 1
        return await super().wait()


def _make_recognition() -> tuple[AudioRecognition, _TrackingEvent]:
    recognition = AudioRecognition.__new__(AudioRecognition)
    final_received = _TrackingEvent()
    final_received.set()

    recognition._stt = object()
    recognition._closing = asyncio.Event()
    recognition._vad = object()
    recognition._turn_detection_mode = "manual"
    recognition._last_final_transcript_time = time.time() - 1.0
    recognition._last_speaking_time = recognition._last_final_transcript_time
    recognition._user_silence_ev = asyncio.Event()
    recognition._speaking = False
    recognition._final_transcript_received = final_received
    recognition._audio_transcript = "cached transcript"
    recognition._audio_interim_transcript = ""
    recognition._sample_rate = 16000
    recognition._push_audio = MagicMock()
    recognition._hooks = MagicMock()
    recognition._run_eou_detection = MagicMock()
    recognition._commit_user_turn_atask = None
    recognition._user_turn_committed = False

    return recognition, final_received


async def test_commit_user_turn_reuses_stale_final_after_manual_audio_detached() -> None:
    recognition, final_received = _make_recognition()

    transcript = await recognition._commit_user_turn(
        audio_detached=True,
        transcript_timeout=0.02,
    )

    assert transcript == "cached transcript"
    assert final_received.wait_calls == 0
    recognition._push_audio.assert_not_called()


async def test_commit_user_turn_flushes_when_speech_follows_cached_final() -> None:
    recognition, final_received = _make_recognition()
    assert recognition._last_final_transcript_time is not None
    recognition._last_speaking_time = recognition._last_final_transcript_time + 0.1

    future = recognition._commit_user_turn(
        audio_detached=True,
        transcript_timeout=1.0,
        stt_flush_duration=0.2,
    )
    for _ in range(3):
        if final_received.wait_calls:
            break
        await asyncio.sleep(0)

    assert not future.done()
    assert final_received.wait_calls == 1
    recognition._audio_transcript = "cached transcript with tail"
    final_received.set()

    assert await future == "cached transcript with tail"
    recognition._push_audio.assert_called_once()


async def test_commit_user_turn_flushes_without_vad_to_confirm_cached_final() -> None:
    recognition, final_received = _make_recognition()
    recognition._vad = None

    future = recognition._commit_user_turn(
        audio_detached=True,
        transcript_timeout=1.0,
        stt_flush_duration=0.2,
    )
    for _ in range(3):
        if final_received.wait_calls:
            break
        await asyncio.sleep(0)

    assert not future.done()
    assert final_received.wait_calls == 1
    recognition._audio_transcript = "cached transcript with tail"
    final_received.set()

    assert await future == "cached transcript with tail"
    recognition._push_audio.assert_called_once()


async def test_commit_user_turn_flushes_when_cached_final_is_whitespace() -> None:
    recognition, final_received = _make_recognition()
    recognition._audio_transcript = " \t "

    future = recognition._commit_user_turn(
        audio_detached=True,
        transcript_timeout=1.0,
        stt_flush_duration=0.2,
    )
    for _ in range(3):
        if final_received.wait_calls:
            break
        await asyncio.sleep(0)

    assert not future.done()
    assert final_received.wait_calls == 1
    recognition._audio_transcript = "fresh transcript"
    final_received.set()

    assert await future == "fresh transcript"
    recognition._push_audio.assert_called_once()


@pytest.mark.parametrize(
    ("audio_detached", "manual_mode", "has_final", "has_interim"),
    [
        pytest.param(False, True, True, False, id="audio-still-attached"),
        pytest.param(True, True, True, True, id="interim-still-pending"),
        pytest.param(True, True, False, False, id="no-final-yet"),
        pytest.param(True, False, True, False, id="non-manual-session-close"),
    ],
)
async def test_commit_user_turn_waits_when_another_final_may_be_needed(
    audio_detached: bool,
    manual_mode: bool,
    has_final: bool,
    has_interim: bool,
) -> None:
    recognition, final_received = _make_recognition()
    recognition._turn_detection_mode = "manual" if manual_mode else "vad"
    recognition._last_final_transcript_time = time.time() - 1.0 if has_final else None
    recognition._audio_transcript = "cached transcript" if has_final else ""
    recognition._audio_interim_transcript = "partial transcript" if has_interim else ""

    future = recognition._commit_user_turn(
        audio_detached=audio_detached,
        transcript_timeout=1.0,
        stt_flush_duration=0.2,
    )
    for _ in range(3):
        if final_received.wait_calls:
            break
        await asyncio.sleep(0)

    assert not future.done()
    assert final_received.wait_calls == 1

    recognition._audio_transcript = "completed transcript"
    recognition._audio_interim_transcript = ""
    final_received.set()

    assert await future == "completed transcript"
    if audio_detached:
        recognition._push_audio.assert_called_once()
    else:
        recognition._push_audio.assert_not_called()
