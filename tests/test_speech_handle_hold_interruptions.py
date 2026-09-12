"""``SpeechHandle.hold_interruptions()``: the counted, public interruption hold.

The count is kept beside the assigned ``allow_interruptions`` value rather than
overwriting it. Overwriting it is what the private hold used to do, and it lost
information in both directions: an assignment during a hold defeated the hold, and
the last release discarded whatever had been assigned since. Both are covered below.
"""

from __future__ import annotations

import pytest

from livekit.agents.voice.speech_handle import SpeechHandle

pytestmark = pytest.mark.unit


def _handle(*, allow_interruptions: bool = True) -> SpeechHandle:
    return SpeechHandle.create(allow_interruptions=allow_interruptions)


async def test_a_hold_makes_the_speech_uninterruptible_and_restores_it() -> None:
    handle = _handle()

    with handle.hold_interruptions() as held:
        assert held is handle
        assert handle.allow_interruptions is False
        assert handle.interruptions_held is True
        with pytest.raises(RuntimeError, match="does not allow interruptions"):
            handle.interrupt()

    assert handle.allow_interruptions is True
    assert handle.interruptions_held is False
    handle.interrupt()
    assert handle.interrupted


async def test_overlapping_holds_release_only_on_the_last_one() -> None:
    handle = _handle()

    with handle.hold_interruptions():
        with handle.hold_interruptions():
            assert handle.allow_interruptions is False
        assert handle.allow_interruptions is False, "the outer holder still holds"

    assert handle.allow_interruptions is True
    handle._mark_done()


async def test_a_hold_does_not_make_an_uninterruptible_speech_interruptible() -> None:
    """The release restores nothing -- it only stops holding."""
    handle = _handle(allow_interruptions=False)

    with handle.hold_interruptions():
        assert handle.allow_interruptions is False

    assert handle.allow_interruptions is False
    handle._mark_done()


async def test_an_assignment_during_a_hold_cannot_defeat_the_hold() -> None:
    """The regression. The hold used to overwrite ``allow_interruptions``, so assigning
    it back to True during a hold made the speech interruptible again -- the hold was
    silently gone while its holder still believed it was protected."""
    handle = _handle()

    with handle.hold_interruptions():
        handle.allow_interruptions = True

        assert handle.allow_interruptions is False, "still held"
        with pytest.raises(RuntimeError, match="does not allow interruptions"):
            handle.interrupt()

    handle._mark_done()


async def test_an_assignment_during_a_hold_survives_the_release() -> None:
    """And the other direction: the release must not discard it."""
    handle = _handle()

    with handle.hold_interruptions():
        handle.allow_interruptions = False

    assert handle.allow_interruptions is False, "the assignment outlives the hold"
    handle._mark_done()


async def test_a_forced_interrupt_lands_through_a_hold() -> None:
    """A hold survives barge-in; it does not make a speech impossible to stop."""
    handle = _handle()

    with handle.hold_interruptions():
        handle.interrupt(force=True)
        assert handle.interrupted


async def test_a_release_after_a_forced_interrupt_does_not_raise() -> None:
    handle = _handle()

    with handle.hold_interruptions():
        handle.interrupt(force=True)

    assert handle.interrupted
    assert handle.interruptions_held is False


async def test_an_already_interrupted_speech_refuses_to_be_held() -> None:
    """Holding a speech that has already been cut off would report protection that
    cannot exist, and leave the holder waiting on a speech that is not playing."""
    handle = _handle()
    handle.interrupt()

    with pytest.raises(RuntimeError, match="already interrupted"):
        with handle.hold_interruptions():
            pass

    assert handle.interruptions_held is False, "the failed hold left the count balanced"
    handle._mark_done()


async def test_a_hold_released_by_an_exception_still_releases() -> None:
    handle = _handle()

    with pytest.raises(ValueError, match="boom"):
        with handle.hold_interruptions():
            raise ValueError("boom")

    assert handle.interruptions_held is False
    assert handle.allow_interruptions is True
    handle._mark_done()


async def test_repeated_hold_cycles_do_not_accumulate() -> None:
    handle = _handle()

    for _ in range(3):
        with handle.hold_interruptions():
            assert handle.allow_interruptions is False
        assert handle.allow_interruptions is True

    assert handle.interruptions_held is False
    handle._mark_done()


async def test_interruptions_held_is_narrower_than_allow_interruptions() -> None:
    """A speech configured uninterruptible is not a *held* speech, and the realtime
    barge-in path tells them apart to decide whether a refusal is expected."""
    handle = _handle(allow_interruptions=False)

    assert handle.allow_interruptions is False
    assert handle.interruptions_held is False
    handle._mark_done()
