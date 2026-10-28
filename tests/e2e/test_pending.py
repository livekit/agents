"""Journeys blocked on named implementation slices; never counted as passing coverage."""

import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="I-8: tasks/get or subscribe must settle worker-lost without a new lane acquisition",
)
async def test_worker_loss_settles_on_read_without_another_message() -> None:
    """Kill A after service entry; B must report worker-lost on read alone.

    Must observe A's exit AND reaping and expire its lease. Do not send a new message
    to trigger orphan settlement; the ordinary crash/next-turn journey covers that case.
    """
    pytest.fail("enable this journey with I-8 read-side orphan settlement")


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="H-13/K-23: real-room caller audio-stop bound and calibrated speech driver",
)
async def test_barge_in_stops_audio_at_receiving_caller() -> None:
    """Live room/worker/RTC; scripted model/audio. Start speech, then publish interruption.

    Measure caller-side interrupting-speech onset to last received agent frame against
    K-23's published bound. Require received speech before interruption; a history flag
    or silence from an unstarted agent is insufficient.
    """
    pytest.fail("enable with real-room speech injection and K-23 bound")


@pytest.mark.xfail(
    run=False, strict=True, reason="H-6/G-6: assembled recording and call worker lifecycle"
)
async def test_spoken_call_produces_readable_recording_with_expected_media() -> None:
    """Live RTC/worker; scripted providers. Caller hears reply; recording decodes with audio.

    Missing/empty/corrupt recording fails. Fixtures must delete room, close caller and
    release recorder/avatar services on failure or cancellation.
    """
    pytest.fail("enable with recording/call lifecycle slices")


@pytest.mark.xfail(
    run=False,
    strict=True,
    reason="L-6/I-20: console through assembled authorized thread and A2A APIs",
)
async def test_console_chat_and_call_share_thread() -> None:
    """Live console/worker/room; type, call and reconnect, retaining the same thread/history."""
    pytest.fail("enable with console and thread assembly")
