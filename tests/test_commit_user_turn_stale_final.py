"""Regression tests for the stale-final commit stall (livekit/agents#7517).

On manual turn-taking, ``commit_user_turn`` used to discard a final transcript
it already held whenever the last final was older than 0.5s, then wait the full
``transcript_timeout`` for a second final that a streaming STT will never emit
for audio it already closed. The commit now reuses the in-hand final when the
STT has emitted nothing since it.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = [pytest.mark.unit]


def _make_recognition() -> AudioRecognition:
    """Build an AudioRecognition stub with just the attributes _commit_user_turn reads."""
    ar = object.__new__(AudioRecognition)  # type: ignore[attr-defined]
    ar._session = MagicMock()  # type: ignore[attr-defined]
    ar._session._root_span_context = None
    ar._stt = MagicMock()  # type: ignore[attr-defined]
    ar._closing = asyncio.Event()  # type: ignore[attr-defined]
    ar._turn_detection_mode = "manual"  # type: ignore[attr-defined]
    ar._vad_base_turn_detection = False  # type: ignore[attr-defined]  # manual mode
    ar._last_language = ""  # type: ignore[attr-defined]
    ar._turn_detector = None  # type: ignore[attr-defined]
    ar._turn_detector_stream = None  # type: ignore[attr-defined]
    ar._end_of_turn_task = None  # type: ignore[attr-defined]
    ar._last_speaking_time = None  # type: ignore[attr-defined]
    ar._speech_start_time = None  # type: ignore[attr-defined]
    ar._vad_speech_started = False  # type: ignore[attr-defined]
    ar._last_emitted_prediction = None  # type: ignore[attr-defined]
    ar._user_silence_ev = asyncio.Event()  # type: ignore[attr-defined]
    ar._user_silence_ev.set()
    ar._turn_tracker = MagicMock()  # type: ignore[attr-defined]
    ar._user_turn_start = None  # type: ignore[attr-defined]
    ar._eou_wait_span = MagicMock()  # type: ignore[attr-defined]
    ar._eou_wait_span.is_recording.return_value = False
    ar._endpointing = MagicMock()  # type: ignore[attr-defined]
    ar._endpointing.min_delay = 0.0
    ar._stt_request_ids = []  # type: ignore[attr-defined]
    ar._session.options.turn_handling = {
        "user_turn_limit": {"max_words": None, "max_duration": None}
    }
    ar._word_tokenizer = MagicMock()  # type: ignore[attr-defined]
    ar._user_turn_span = None  # type: ignore[attr-defined]
    ar._stt_model = None  # type: ignore[attr-defined]
    ar._stt_provider = None  # type: ignore[attr-defined]
    ar._stt_aligned_transcript = False  # type: ignore[attr-defined]
    ar._stt_events = []  # type: ignore[attr-defined]
    ar._eou_wait_started_at_ns = None  # type: ignore[attr-defined]
    ar._eou_wait_rearms = 0  # type: ignore[attr-defined]
    ar._eou_wait_not_committed = 0  # type: ignore[attr-defined]
    ar._user_turn_resumes = 0  # type: ignore[attr-defined]
    ar._eou_wait_floor_ns = None  # type: ignore[attr-defined]
    ar._eou_detection_span = None  # type: ignore[attr-defined]
    ar._turn_backchannel_over_agent = False  # type: ignore[attr-defined]
    ar._overlap_in_current_turn = False  # type: ignore[attr-defined]
    ar._turn_backchannel = False  # type: ignore[attr-defined]
    ar._transcription_timeout_handle = None  # type: ignore[attr-defined]
    ar._tasks = set()  # type: ignore[attr-defined]
    ar._user_turn_committed = False  # type: ignore[attr-defined]
    ar._commit_user_turn_atask = None  # type: ignore[attr-defined]
    ar._final_transcript_received = asyncio.Event()  # type: ignore[attr-defined]
    ar._last_final_transcript_time = None  # type: ignore[attr-defined]
    ar._audio_interim_transcript = ""  # type: ignore[attr-defined]
    ar._audio_transcript = ""  # type: ignore[attr-defined]
    ar._audio_preflight_transcript = ""  # type: ignore[attr-defined]
    ar._final_transcript_confidence = []  # type: ignore[attr-defined]
    ar._sample_rate = None  # type: ignore[attr-defined]
    ar._stt_consumer_atask = None  # type: ignore[attr-defined]
    ar._stt_pipeline = None  # type: ignore[attr-defined]
    ar._transcript_buffer = MagicMock()  # type: ignore[attr-defined]
    ar._hooks = MagicMock()  # type: ignore[attr-defined]
    ar._hooks.retrieve_chat_ctx.return_value = MagicMock()
    return ar


def _emit_final(ar: AudioRecognition, text: str, age_s: float) -> None:
    """Simulate a final transcript that arrived `age_s` seconds ago."""
    ar._final_transcript_received.set()
    ar._audio_transcript = text
    ar._last_final_transcript_time = time.time() - age_s
    # the tests stub out the pipeline; default to "no audio trail" (None) so
    # they exercise the reuse path exactly as before the last_audio_at guard
    ar._stt_pipeline = None


@pytest.mark.asyncio
async def test_stale_final_commits_immediately():
    """Final in hand >0.5s ago + no interim since: commit returns at once, no timeout wait."""
    ar = _make_recognition()
    _emit_final(ar, "hello world", age_s=1.2)

    t0 = time.monotonic()
    fut = ar._commit_user_turn(audio_detached=False, transcript_timeout=30.0)
    transcript = await asyncio.wait_for(fut, timeout=2.0)
    elapsed = time.monotonic() - t0

    assert transcript == "hello world"
    assert elapsed < 1.0, f"commit stalled {elapsed:.2f}s despite final in hand (#7517)"


@pytest.mark.asyncio
async def test_fresh_final_resolves_immediately():
    """A recent final (<0.5s) skips the flush-wait in current code too — unchanged."""
    ar = _make_recognition()
    _emit_final(ar, "partial end", age_s=0.1)

    t0 = time.monotonic()
    fut = ar._commit_user_turn(audio_detached=False, transcript_timeout=0.3)
    transcript = await asyncio.wait_for(fut, timeout=2.0)
    elapsed = time.monotonic() - t0

    assert transcript == "partial end"
    assert elapsed < 0.2, "fresh final path was already immediate; must stay so"


@pytest.mark.asyncio
async def test_interim_since_final_still_waits_timeout():
    """An interim after the final means STT is still working: flush-wait runs (bounded)."""
    ar = _make_recognition()
    _emit_final(ar, "first part", age_s=1.5)
    ar._audio_interim_transcript = "first part and"

    t0 = time.monotonic()
    fut = ar._commit_user_turn(audio_detached=False, transcript_timeout=0.3)
    transcript = await asyncio.wait_for(fut, timeout=2.0)
    elapsed = time.monotonic() - t0

    assert "first part and" in transcript  # interim folded into the commit
    assert elapsed >= 0.25, "interim-since-final must still go through the flush-wait"


@pytest.mark.asyncio
async def test_no_final_at_all_still_waits_timeout():
    """No final received: unchanged behavior (flush + bounded wait, empty transcript)."""
    ar = _make_recognition()

    t0 = time.monotonic()
    fut = ar._commit_user_turn(audio_detached=False, transcript_timeout=0.3)
    transcript = await asyncio.wait_for(fut, timeout=2.0)
    elapsed = time.monotonic() - t0

    assert transcript == ""
    assert elapsed >= 0.25, "no final must still go through the flush-wait"


@pytest.mark.asyncio
async def test_audio_since_final_takes_flush_wait():
    """Devin review bug 1: audio reached the STT after the final (no interim,
    e.g. interim_results=False) — the turn is NOT complete, keep the flush-wait."""
    ar = _make_recognition()
    _emit_final(ar, "book", age_s=0.4)  # final in hand, recent
    pipeline = MagicMock()
    pipeline.last_audio_at = time.time()  # audio pushed after that final
    ar._stt_pipeline = pipeline

    t0 = time.monotonic()
    fut = ar._commit_user_turn(audio_detached=False, transcript_timeout=0.3)
    transcript = await asyncio.wait_for(fut, timeout=2.0)
    elapsed = time.monotonic() - t0

    assert transcript == "book"  # the old final is still what we hold
    assert elapsed >= 0.25, "pending audio without final must go through the flush-wait"


@pytest.mark.asyncio
async def test_stale_final_event_from_prior_turn_waits_for_new_final():
    """Devin review bug 2: the final event survived a previous commit — a new
    manual turn with no final of its own must NOT reuse the stale event."""
    ar = _make_recognition()
    # prior turn: final arrived and was committed (cleanup resets timestamps
    # but the event stays set) — the new turn starts empty
    ar._final_transcript_received.set()
    ar._last_final_transcript_time = None  # reset by commit cleanup
    ar._stt_pipeline = None

    t0 = time.monotonic()
    fut = ar._commit_user_turn(audio_detached=False, transcript_timeout=0.3)
    transcript = await asyncio.wait_for(fut, timeout=2.0)
    elapsed = time.monotonic() - t0

    assert transcript == ""
    assert elapsed >= 0.25, "stale final event must not shortcut the new turn"


@pytest.mark.asyncio
async def test_clear_user_turn_releases_stale_final_event():
    """After _clear_user_turn the stale final event must be released, so the
    next commit waits for its own final (the event no longer leaks)."""
    ar = _make_recognition()
    ar._final_transcript_received.set()
    ar._last_final_transcript_time = time.time()
    ar._final_transcript_confidence = [0.9]
    # attributes _clear_user_turn touches beyond the commit stub
    ar._audio_preflight_transcript = ""
    ar._speech_start_time = None
    ar._last_speaking_time = None
    ar._vad_speech_started = False
    ar._user_turn_committed = False
    ar._last_emitted_prediction = None
    ar._turn_detector_stream = None
    ar._active_vad_speech_started_at = None
    ar._last_final_transcript_time = None
    # main's _clear_user_turn resets the STT (update_stt round-trip); with a
    # MagicMock stt that would spawn a real pipeline + consumer task. The
    # behavior under test is the final-event release, not the STT reset.
    ar._update_stt = lambda _stt: None  # type: ignore[method-assign]

    ar._clear_user_turn()

    assert not ar._final_transcript_received.is_set(), (
        "cleared turn must release the final event (it belonged to that turn)"
    )


@pytest.mark.asyncio
async def test_repeated_commit_cancels_previous():
    """A second commit cancels the first pending one (existing contract, kept)."""
    ar = _make_recognition()
    _emit_final(ar, "hello again", age_s=1.0)

    first = ar._commit_user_turn(audio_detached=False, transcript_timeout=5.0)
    second = ar._commit_user_turn(audio_detached=False, transcript_timeout=5.0)
    transcript = await asyncio.wait_for(second, timeout=2.0)

    assert transcript == "hello again"
    assert first.done() and first.cancelled()
