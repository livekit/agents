"""Unit tests for the end-of-turn timing metric computation.

These exercise ``_compute_end_of_turn_metrics`` in isolation, with crafted
timestamps (no audio, no STT/VAD). They pin down the behaviour described in
issue #6093: when the internal ``_last_speaking_time`` anchor (reported as
``stopped_speaking_at``) is stale and predates the start of the current turn,
the previous code emitted wildly inflated ``transcription_delay`` /
``end_of_turn_delay`` values (often >200s) instead of skipping the calculation.
"""

from __future__ import annotations

import pytest

from livekit.agents.voice.audio_recognition import _compute_end_of_turn_metrics

pytestmark = pytest.mark.unit


def test_normal_turn_produces_small_bounded_delays() -> None:
    """A well-ordered turn yields the expected sub-second delays."""
    started = 1000.0
    stopped = 1005.0  # finished speaking 5s after starting
    last_final = 1005.2  # final transcript landed 0.2s later
    now = 1005.4  # turn committed 0.4s after the user stopped

    metrics = _compute_end_of_turn_metrics(
        speech_start_time=started,
        last_speaking_time=stopped,
        last_final_transcript_time=last_final,
        now=now,
    )

    assert metrics.started_speaking_at == started
    assert metrics.stopped_speaking_at == stopped
    assert metrics.transcription_delay == pytest.approx(0.2)
    assert metrics.end_of_turn_delay == pytest.approx(0.4)


def test_stale_anchor_predating_turn_start_is_skipped() -> None:
    """Regression for issue #6093.

    When the turn detector commits a turn whose ``_last_speaking_time`` anchor was
    never refreshed for this segment, the anchor can be from a much earlier point
    in the session and predate ``speech_start_time``. The old code passed the
    not-None guard and computed ``end_of_turn_delay = now - last_speaking_time``,
    yielding ~220s. The metric must instead be skipped (left as ``None``) rather
    than reported as a bogus huge value.
    """
    # numbers mirror the issue payload: stopped_speaking_at ~220s before the start
    started = 1781342804.815377
    stopped = 1781342584.6181495  # ~220s BEFORE `started` — stale anchor
    last_final = 1781342804.9027314
    now = 1781342804.9027314

    metrics = _compute_end_of_turn_metrics(
        speech_start_time=started,
        last_speaking_time=stopped,
        last_final_transcript_time=last_final,
        now=now,
    )

    assert metrics.started_speaking_at is None
    assert metrics.stopped_speaking_at is None
    assert metrics.transcription_delay is None
    assert metrics.end_of_turn_delay is None


def test_anchor_equal_to_start_is_accepted() -> None:
    """An anchor exactly at the turn start is valid (boundary, delay == 0)."""
    started = 2000.0
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=started,
        last_speaking_time=started,
        last_final_transcript_time=started,
        now=started + 0.3,
    )

    assert metrics.started_speaking_at == started
    assert metrics.stopped_speaking_at == started
    assert metrics.transcription_delay == 0.0
    assert metrics.end_of_turn_delay == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("speech_start_time", "last_speaking_time", "last_final_transcript_time"),
    [
        (None, 1005.0, 1005.2),
        (1000.0, None, 1005.2),
        (1000.0, 1005.0, None),
    ],
)
def test_missing_anchor_is_skipped(
    speech_start_time: float | None,
    last_speaking_time: float | None,
    last_final_transcript_time: float | None,
) -> None:
    """Any missing anchor skips the calculation (unreliable VAD/STT timing)."""
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=speech_start_time,
        last_speaking_time=last_speaking_time,
        last_final_transcript_time=last_final_transcript_time,
        now=1006.0,
    )

    assert metrics.started_speaking_at is None
    assert metrics.stopped_speaking_at is None
    assert metrics.transcription_delay is None
    assert metrics.end_of_turn_delay is None


def test_provider_stop_time_is_preferred_over_the_anchor() -> None:
    """The metrics report where the user actually stopped talking.

    In stt turn detection the endpointing anchor moves to the arrival of an
    untimestamped ``END_OF_SPEECH`` (an explicit endpointing signal), but the
    provider's word end from the last transcript is the better reported value.
    """
    started = 1000.0
    anchor = 1005.6  # END_OF_SPEECH arrival, 0.6 s after the last word
    provider = 1005.0  # the word end, from the last final transcript

    metrics = _compute_end_of_turn_metrics(
        speech_start_time=started,
        last_speaking_time=anchor,
        provider_speaking_time=provider,
        last_final_transcript_time=1005.2,
        now=1005.8,
    )

    assert metrics.stopped_speaking_at == provider
    assert metrics.transcription_delay == pytest.approx(0.2)
    assert metrics.end_of_turn_delay == pytest.approx(0.8)


def test_provider_stop_time_predating_the_segment_falls_back_to_the_anchor() -> None:
    """A provider value left over from an earlier segment is dropped.

    The same consistency rule as the stale anchor above applies: the value cannot
    predate the turn it is reported for, so the endpointing anchor is used instead
    of skipping the metrics altogether.
    """
    started = 1010.0  # a new segment opened after the stale provider value
    anchor = 1012.5

    metrics = _compute_end_of_turn_metrics(
        speech_start_time=started,
        last_speaking_time=anchor,
        provider_speaking_time=1005.0,  # from the previous segment
        last_final_transcript_time=1012.6,
        now=1012.8,
    )

    assert metrics.stopped_speaking_at == anchor
    assert metrics.transcription_delay == pytest.approx(0.1)
    assert metrics.end_of_turn_delay == pytest.approx(0.3)


def test_stale_provider_value_does_not_rescue_a_stale_anchor() -> None:
    """Both values stale: skipped like any out-of-order anchor."""
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=1010.0,
        last_speaking_time=1005.0,
        provider_speaking_time=1001.0,
        last_final_transcript_time=1012.6,
        now=1012.8,
    )

    assert metrics.stopped_speaking_at is None
    assert metrics.end_of_turn_delay is None
