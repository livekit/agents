"""Manual turn commits must not wait the full transcript_timeout for a final that is already in hand."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.audio_eot


def _recognition(*, transcript: str = "hello") -> AudioRecognition:
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._stt = MagicMock()
    ar._closing = asyncio.Event()
    ar._stt_pipeline = None
    ar._stt_flush_requested = False
    ar._final_transcript_received = asyncio.Event()
    ar._audio_interim_transcript = ""
    ar._audio_transcript = transcript
    ar._user_turn_committed = False
    ar._commit_user_turn_atask = None
    ar._sample_rate = 16000
    ar._hooks = MagicMock()
    ar._hooks.retrieve_chat_ctx.return_value = MagicMock()
    ar._run_eou_detection = MagicMock()
    ar._push_audio = MagicMock()
    return ar


async def test_commit_does_not_wait_the_full_timeout_for_an_already_finalized_turn() -> None:
    """A final emitted after the last frame is the turn's transcript; no second final is due."""
    ar = _recognition()
    ar._last_final_transcript_time = time.time() - 0.6  # older than the 0.5s heuristic
    ar._last_audio_time = ar._last_final_transcript_time - 0.1  # nothing was pushed since

    started = time.monotonic()
    fut = ar._commit_user_turn(audio_detached=True, transcript_timeout=10.0)

    assert await asyncio.wait_for(fut, timeout=3.0) == "hello"
    assert time.monotonic() - started < 3.0


async def test_commit_still_waits_briefly_for_a_second_final() -> None:
    """Streaming STT can finalize a second phrase for audio it already accepted, so keep a grace window."""
    ar = _recognition()
    ar._last_final_transcript_time = time.time() - 0.6
    ar._last_audio_time = ar._last_final_transcript_time - 0.1

    fut = ar._commit_user_turn(audio_detached=True, transcript_timeout=10.0)

    async def _second_final() -> None:
        await asyncio.sleep(0.1)
        ar._audio_transcript = "hello world"
        ar._final_transcript_received.set()

    task = asyncio.create_task(_second_final())
    assert await asyncio.wait_for(fut, timeout=3.0) == "hello world"
    await task


async def test_commit_waits_the_full_timeout_when_audio_arrived_after_the_final() -> None:
    """Speech the STT has not finalized yet still needs the flush and the full wait."""
    ar = _recognition()
    ar._last_final_transcript_time = time.time() - 0.6
    ar._last_audio_time = time.time()  # audio pushed after that final

    started = time.monotonic()
    fut = ar._commit_user_turn(audio_detached=True, transcript_timeout=0.4)
    assert await asyncio.wait_for(fut, timeout=2.0) == "hello"
    assert time.monotonic() - started >= 0.4
    ar._push_audio.assert_called()


async def test_a_recent_final_that_predates_pushed_audio_still_waits() -> None:
    """Age is not the signal: a final older than the pushed audio is stale however recent it is."""
    ar = _recognition()
    ar._last_final_transcript_time = time.time() - 0.1  # inside the 0.5s heuristic
    ar._last_audio_time = time.time()  # but speech was pushed after it

    started = time.monotonic()
    fut = ar._commit_user_turn(audio_detached=True, transcript_timeout=0.4)
    assert await asyncio.wait_for(fut, timeout=2.0) == "hello"
    assert time.monotonic() - started >= 0.4
    ar._push_audio.assert_called()


async def test_commit_waits_when_an_interim_transcript_is_pending() -> None:
    """An interim means STT is still working on this utterance, so the final is still coming."""
    ar = _recognition(transcript="hello")
    ar._audio_interim_transcript = "wor"
    ar._last_final_transcript_time = time.time() - 0.6
    ar._last_audio_time = ar._last_final_transcript_time - 0.1

    started = time.monotonic()
    fut = ar._commit_user_turn(audio_detached=True, transcript_timeout=0.4)
    result = await asyncio.wait_for(fut, timeout=2.0)
    assert time.monotonic() - started >= 0.4
    assert result == "hello wor"
