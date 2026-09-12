"""Public SpeechHandle.hold_interruptions() context manager (issue #7191)."""

from __future__ import annotations

import pytest

from livekit.agents.voice.speech_handle import SpeechHandle

pytestmark = pytest.mark.unit


def test_hold_interruptions_disallows_then_restores() -> None:
    handle = SpeechHandle.create(allow_interruptions=True)

    with handle.hold_interruptions() as held:
        assert held is handle
        assert handle.allow_interruptions is False
        with pytest.raises(RuntimeError, match="does not allow interruptions"):
            handle.interrupt()

    assert handle.allow_interruptions is True
    handle.interrupt()
    assert handle.interrupted


def test_hold_interruptions_nested_restores_once() -> None:
    handle = SpeechHandle.create(allow_interruptions=True)

    with handle.hold_interruptions():
        assert handle.allow_interruptions is False
        with handle.hold_interruptions():
            assert handle.allow_interruptions is False
        # inner release must not restore while outer still holds
        assert handle.allow_interruptions is False

    assert handle.allow_interruptions is True


def test_hold_interruptions_preserves_prior_false() -> None:
    handle = SpeechHandle.create(allow_interruptions=False)

    with handle.hold_interruptions():
        assert handle.allow_interruptions is False

    assert handle.allow_interruptions is False


def test_force_interrupt_bypasses_hold() -> None:
    handle = SpeechHandle.create(allow_interruptions=True)

    with handle.hold_interruptions():
        handle.interrupt(force=True)
        assert handle.interrupted


def test_allow_interruptions_true_during_hold_restores_after_release() -> None:
    handle = SpeechHandle.create(allow_interruptions=False)

    with handle.hold_interruptions():
        assert handle.allow_interruptions is False
        handle.allow_interruptions = True
        # assignment is deferred; hold stays effective
        assert handle.allow_interruptions is False
        with pytest.raises(RuntimeError, match="does not allow interruptions"):
            handle.interrupt()

    assert handle.allow_interruptions is True
    handle.interrupt()
    assert handle.interrupted


def test_allow_interruptions_false_during_nested_holds_restores_after_final() -> None:
    handle = SpeechHandle.create(allow_interruptions=True)

    with handle.hold_interruptions():
        assert handle.allow_interruptions is False
        handle.allow_interruptions = True
        assert handle.allow_interruptions is False
        with handle.hold_interruptions():
            handle.allow_interruptions = False
            assert handle.allow_interruptions is False
        # outer still holds; still not interruptible
        assert handle.allow_interruptions is False
        with pytest.raises(RuntimeError, match="does not allow interruptions"):
            handle.interrupt()

    assert handle.allow_interruptions is False
    with pytest.raises(RuntimeError, match="does not allow interruptions"):
        handle.interrupt()


def test_force_interrupt_still_works_after_deferred_true_assignment() -> None:
    handle = SpeechHandle.create(allow_interruptions=True)

    with handle.hold_interruptions():
        handle.allow_interruptions = True
        assert handle.allow_interruptions is False
        handle.interrupt(force=True)
        assert handle.interrupted

