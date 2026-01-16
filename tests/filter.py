import pytest
from livekit.agents.voice import InterruptionFilter


class TestInterruptionFilter:
    """Unit tests for the InterruptionFilter class."""

    def test_default_initialization(self):
        """Test that the filter initializes with default settings."""
        filter = InterruptionFilter()
        assert filter.enabled is True
        assert len(filter.ignore_words) > 0
        assert "yeah" in filter.ignore_words
        assert "ok" in filter.ignore_words

    def test_custom_ignore_words(self):
        """Test initialization with custom ignore words."""
        custom_words = ["yeah", "ok", "sure"]
        filter = InterruptionFilter(ignore_words=custom_words)
        assert filter.ignore_words == {"yeah", "ok", "sure"}

    def test_disabled_filter(self):
        """Test that disabled filter allows all interruptions."""
        filter = InterruptionFilter(enabled=False)

        # Should allow all interruptions when disabled
        assert filter.should_ignore_interruption("yeah", agent_is_speaking=True) is False
        assert filter.should_ignore_interruption("stop", agent_is_speaking=True) is False

    def test_backchanneling_while_speaking(self):
        """Test that backchanneling is ignored when agent is speaking."""
        filter = InterruptionFilter()

        # Single backchanneling words
        assert filter.should_ignore_interruption("yeah", agent_is_speaking=True) is True
        assert filter.should_ignore_interruption("ok", agent_is_speaking=True) is True
        assert filter.should_ignore_interruption("hmm", agent_is_speaking=True) is True
        assert filter.should_ignore_interruption("uh-huh", agent_is_speaking=True) is True

        # Multiple backchanneling words
        assert filter.should_ignore_interruption("yeah okay", agent_is_speaking=True) is True
        assert filter.should_ignore_interruption("hmm right", agent_is_speaking=True) is True

    def test_backchanneling_while_silent(self):
        """Test that backchanneling is processed when agent is silent."""
        filter = InterruptionFilter()

        # Should process all input when agent is not speaking
        assert filter.should_ignore_interruption("yeah", agent_is_speaking=False) is False
        assert filter.should_ignore_interruption("ok", agent_is_speaking=False) is False
        assert filter.should_ignore_interruption("hmm", agent_is_speaking=False) is False

    def test_real_interruptions(self):
        """Test that real interruptions are not ignored."""
        filter = InterruptionFilter()

        # Commands should not be ignored
        assert filter.should_ignore_interruption("wait", agent_is_speaking=True) is False
        assert filter.should_ignore_interruption("stop", agent_is_speaking=True) is False
        assert filter.should_ignore_interruption("no", agent_is_speaking=True) is False
        assert filter.should_ignore_interruption("hold on", agent_is_speaking=True) is False

    def test_mixed_input(self):
        """Test that mixed input (backchanneling + command) is not ignored."""
        filter = InterruptionFilter()

        # Mixed input should not be ignored (contains non-backchannel words)
        assert filter.should_ignore_interruption("yeah wait", agent_is_speaking=True) is False
        assert filter.should_ignore_interruption("ok but stop", agent_is_speaking=True) is False
        assert filter.should_ignore_interruption("hmm no", agent_is_speaking=True) is False
        assert filter.should_ignore_interruption("yeah okay but wait", agent_is_speaking=True) is False

    def test_case_insensitivity(self):
        """Test that matching is case-insensitive by default."""
        filter = InterruptionFilter(case_sensitive=False)

        assert filter.should_ignore_interruption("YEAH", agent_is_speaking=True) is True
        assert filter.should_ignore_interruption("Yeah", agent_is_speaking=True) is True
        assert filter.should_ignore_interruption("yEaH", agent_is_speaking=True) is True

    def test_case_sensitivity(self):
        """Test case-sensitive matching when enabled."""
        filter = InterruptionFilter(
            ignore_words=["yeah", "ok"],
            case_sensitive=True
        )

        assert filter.should_ignore_interruption("yeah", agent_is_speaking=True) is True
        assert filter.should_ignore_interruption("YEAH", agent_is_speaking=True) is False

    def test_empty_input(self):
        """Test handling of empty or whitespace input."""
        filter = InterruptionFilter()

        assert filter.should_ignore_interruption("", agent_is_speaking=True) is False
        assert filter.should_ignore_interruption("   ", agent_is_speaking=True) is False

    def test_punctuation_handling(self):
        """Test that punctuation is handled correctly."""
        filter = InterruptionFilter()

        # Punctuation should be removed
        assert filter.should_ignore_interruption("yeah.", agent_is_speaking=True) is True
        assert filter.should_ignore_interruption("ok!", agent_is_speaking=True) is True
        assert filter.should_ignore_interruption("hmm?", agent_is_speaking=True) is True
        assert filter.should_ignore_interruption("yeah, ok", agent_is_speaking=True) is True

    def test_add_ignore_word(self):
        """Test adding words to the ignore list."""
        filter = InterruptionFilter(ignore_words=["yeah"])

        assert filter.should_ignore_interruption("ok", agent_is_speaking=True) is False

        filter.add_ignore_word("ok")
        assert filter.should_ignore_interruption("ok", agent_is_speaking=True) is True

    def test_remove_ignore_word(self):
        """Test removing words from the ignore list."""
        filter = InterruptionFilter(ignore_words=["yeah", "ok"])

        assert filter.should_ignore_interruption("yeah", agent_is_speaking=True) is True

        filter.remove_ignore_word("yeah")
        assert filter.should_ignore_interruption("yeah", agent_is_speaking=True) is False

    def test_enable_disable(self):
        """Test enabling and disabling the filter."""
        filter = InterruptionFilter(enabled=True)

        assert filter.should_ignore_interruption("yeah", agent_is_speaking=True) is True

        filter.set_enabled(False)
        assert filter.should_ignore_interruption("yeah", agent_is_speaking=True) is False

        filter.set_enabled(True)
        assert filter.should_ignore_interruption("yeah", agent_is_speaking=True) is True


# Integration test scenarios matching the assignment requirements
class TestIntegrationScenarios:
    """Integration tests matching the assignment test scenarios."""

    def test_scenario_1_long_explanation(self):
        """
        Scenario 1: The Long Explanation
        Context: Agent is reading a long paragraph about history.
        User Action: User says "Okay... yeah... uh-huh" while Agent is talking.
        Result: Agent audio does not break. It ignores the user input completely.
        """
        filter = InterruptionFilter()
        agent_speaking = True

        # Test individual backchanneling
        assert filter.should_ignore_interruption("Okay", agent_speaking) is True
        assert filter.should_ignore_interruption("yeah", agent_speaking) is True
        assert filter.should_ignore_interruption("uh-huh", agent_speaking) is True

        # Test combined backchanneling
        assert filter.should_ignore_interruption("Okay yeah uh-huh", agent_speaking) is True

    def test_scenario_2_passive_affirmation(self):
        """
        Scenario 2: The Passive Affirmation
        Context: Agent asks "Are you ready?" and goes silent.
        User Action: User says "Yeah."
        Result: Agent processes "Yeah" as an answer and proceeds.
        """
        filter = InterruptionFilter()
        agent_speaking = False  # Agent is silent

        # Should NOT ignore when agent is silent
        assert filter.should_ignore_interruption("Yeah", agent_speaking) is False

    def test_scenario_3_the_correction(self):
        """
        Scenario 3: The Correction
        Context: Agent is counting "One, two, three..."
        User Action: User says "No stop."
        Result: Agent cuts off immediately.
        """
        filter = InterruptionFilter()
        agent_speaking = True

        # Should NOT ignore real interruption commands
        assert filter.should_ignore_interruption("No stop", agent_speaking) is False
        assert filter.should_ignore_interruption("No", agent_speaking) is False
        assert filter.should_ignore_interruption("stop", agent_speaking) is False

    def test_scenario_4_mixed_input(self):
        """
        Scenario 4: The Mixed Input
        Context: Agent is speaking.
        User Action: User says "Yeah okay but wait."
        Result: Agent stops (because "but wait" is not in the ignore list).
        """
        filter = InterruptionFilter()
        agent_speaking = True

        # Should NOT ignore mixed input with commands
        assert filter.should_ignore_interruption("Yeah okay but wait", agent_speaking) is False
        assert filter.should_ignore_interruption("yeah wait", agent_speaking) is False
        assert filter.should_ignore_interruption("ok but stop", agent_speaking) is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
Footer
© 2026 GitHub, Inc.
Footer navigation
Terms
Privacy
Se
