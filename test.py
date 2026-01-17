import pytest

from livekit.agents.voice import InterruptionFilter


class TestInterruptionFilter:
    """Unit tests for the InterruptionFilter class."""

    def test_default_initialization(self):
        filter = InterruptionFilter()
        assert filter.enabled is True
        assert len(filter.ignore_words) > 0
        assert "yeah" in filter.ignore_words
        assert "ok" in filter.ignore_words

    def test_custom_ignore_words(self):
        custom_words = ["yeah", "ok", "sure"]
        filter = InterruptionFilter(ignore_words=custom_words)
        assert filter.ignore_words == {"yeah", "ok", "sure"}

    def test_disabled_filter(self):
        filter = InterruptionFilter(enabled=False)

        assert filter.should_ignore_interruption("yeah", agent_is_speaking=True) is False
        assert filter.should_ignore_interruption("stop", agent_is_speaking=True) is False

    def test_backchanneling_while_speaking(self):
        filter = InterruptionFilter()

        assert filter.should_ignore_interruption("yeah", True) is True
        assert filter.should_ignore_interruption("ok", True) is True
        assert filter.should_ignore_interruption("hmm", True) is True
        assert filter.should_ignore_interruption("uh-huh", True) is True

        assert filter.should_ignore_interruption("yeah okay", True) is True
        assert filter.should_ignore_interruption("hmm right", True) is True

    def test_backchanneling_while_silent(self):
        filter = InterruptionFilter()

        assert filter.should_ignore_interruption("yeah", False) is False
        assert filter.should_ignore_interruption("ok", False) is False
        assert filter.should_ignore_interruption("hmm", False) is False

    def test_real_interruptions(self):
        filter = InterruptionFilter()

        assert filter.should_ignore_interruption("wait", True) is False
        assert filter.should_ignore_interruption("stop", True) is False
        assert filter.should_ignore_interruption("no", True) is False
        assert filter.should_ignore_interruption("hold on", True) is False

    def test_mixed_input(self):
        filter = InterruptionFilter()

        assert filter.should_ignore_interruption("yeah wait", True) is False
        assert filter.should_ignore_interruption("ok but stop", True) is False
        assert filter.should_ignore_interruption("hmm no", True) is False

    def test_case_insensitivity(self):
        filter = InterruptionFilter(case_sensitive=False)

        assert filter.should_ignore_interruption("YEAH", True) is True
        assert filter.should_ignore_interruption("Yeah", True) is True
        assert filter.should_ignore_interruption("yEaH", True) is True

    def test_case_sensitivity(self):
        filter = InterruptionFilter(ignore_words=["yeah", "ok"], case_sensitive=True)

        assert filter.should_ignore_interruption("yeah", True) is True
        assert filter.should_ignore_interruption("YEAH", True) is False

    def test_empty_input(self):
        filter = InterruptionFilter()

        assert filter.should_ignore_interruption("", True) is False
        assert filter.should_ignore_interruption("   ", True) is False

    def test_punctuation_handling(self):
        filter = InterruptionFilter()

        assert filter.should_ignore_interruption("yeah.", True) is True
        assert filter.should_ignore_interruption("ok!", True) is True
        assert filter.should_ignore_interruption("hmm?", True) is True
        assert filter.should_ignore_interruption("yeah, ok", True) is True

    def test_add_remove_ignore_word(self):
        filter = InterruptionFilter(ignore_words=["yeah"])

        assert filter.should_ignore_interruption("ok", True) is False

        filter.add_ignore_word("ok")
        assert filter.should_ignore_interruption("ok", True) is True

        filter.remove_ignore_word("yeah")
        assert filter.should_ignore_interruption("yeah", True) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
