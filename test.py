import pytest

from livekit.agents.voice import InterruptionFilter


def test_default_initialization():
    interruption_filter = InterruptionFilter()

    assert interruption_filter.enabled is True
    assert len(interruption_filter.ignore_words) > 0
    assert "yeah" in interruption_filter.ignore_words
    assert "ok" in interruption_filter.ignore_words


def test_custom_ignore_words():
    custom_words = ["yeah", "ok", "sure"]
    interruption_filter = InterruptionFilter(ignore_words=custom_words)

    assert interruption_filter.ignore_words == {"yeah", "ok", "sure"}


def test_disabled_filter():
    interruption_filter = InterruptionFilter(enabled=False)

    assert (
        interruption_filter.should_ignore_interruption("yeah", agent_is_speaking=True)
        is False
    )
    assert (
        interruption_filter.should_ignore_interruption("stop", agent_is_speaking=True)
        is False
    )


def test_backchanneling_while_speaking():
    interruption_filter = InterruptionFilter()

    assert interruption_filter.should_ignore_interruption("yeah", True) is True
    assert interruption_filter.should_ignore_interruption("ok", True) is True
    assert interruption_filter.should_ignore_interruption("hmm", True) is True
    assert interruption_filter.should_ignore_interruption("uh-huh", True) is True

    assert interruption_filter.should_ignore_interruption("yeah okay", True) is True
    assert interruption_filter.should_ignore_interruption("hmm right", True) is True


def test_backchanneling_while_silent():
    interruption_filter = InterruptionFilter()

    assert interruption_filter.should_ignore_interruption("yeah", False) is False
    assert interruption_filter.should_ignore_interruption("ok", False) is False
    assert interruption_filter.should_ignore_interruption("hmm", False) is False


def test_real_interruptions():
    interruption_filter = InterruptionFilter()

    assert interruption_filter.should_ignore_interruption("wait", True) is False
    assert interruption_filter.should_ignore_interruption("stop", True) is False
    assert interruption_filter.should_ignore_interruption("no", True) is False
    assert interruption_filter.should_ignore_interruption("hold on", True) is False


def test_mixed_input():
    interruption_filter = InterruptionFilter()

    assert interruption_filter.should_ignore_interruption("yeah wait", True) is False
    assert interruption_filter.should_ignore_interruption("ok but stop", True) is False
    assert interruption_filter.should_ignore_interruption("hmm no", True) is False


def test_case_insensitivity():
    interruption_filter = InterruptionFilter(case_sensitive=False)

    assert interruption_filter.should_ignore_interruption("YEAH", True) is True
    assert interruption_filter.should_ignore_interruption("Yeah", True) is True
    assert interruption_filter.should_ignore_interruption("yEaH", True) is True


def test_case_sensitivity():
    interruption_filter = InterruptionFilter(
        ignore_words=["yeah", "ok"], case_sensitive=True
    )

    assert interruption_filter.should_ignore_interruption("yeah", True) is True
    assert interruption_filter.should_ignore_interruption("YEAH", True) is False


def test_empty_input():
    interruption_filter = InterruptionFilter()

    assert interruption_filter.should_ignore_interruption("", True) is False
    assert interruption_filter.should_ignore_interruption("   ", True) is False


def test_punctuation_handling():
    interruption_filter = InterruptionFilter()

    assert interruption_filter.should_ignore_interruption("yeah.", True) is True
    assert interruption_filter.should_ignore_interruption("ok!", True) is True
    assert interruption_filter.should_ignore_interruption("hmm?", True) is True
    assert interruption_filter.should_ignore_interruption("yeah, ok", True) is True


def test_add_remove_ignore_word():
    interruption_filter = InterruptionFilter(ignore_words=["yeah"])

    assert interruption_filter.should_ignore_interruption("ok", True) is False

    interruption_filter.add_ignore_word("ok")
    assert interruption_filter.should_ignore_interruption("ok", True) is True

    interruption_filter.remove_ignore_word("yeah")
    assert interruption_filter.should_ignore_interruption("yeah", True) is False
