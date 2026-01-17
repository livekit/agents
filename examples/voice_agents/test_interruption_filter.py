"""
Unit tests for the InterruptionFilter module.

Run with: pytest test_interruption_filter.py -v
"""

import pytest
from interruption_filter import InterruptionFilter


class TestInterruptionFilter:
    """Test suite for the InterruptionFilter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.filter = InterruptionFilter()
    
    # Test Scenario 1: Backchannel during agent speech -> should NOT interrupt
    def test_backchannel_during_agent_speech_single_word(self):
        """Single backchannel word while agent is speaking should be blocked."""
        assert self.filter.should_interrupt("yeah", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("ok", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("hmm", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("uh-huh", agent_is_speaking=True) == False
    
    def test_backchannel_during_agent_speech_multi_word(self):
        """Multiple backchannel words while agent is speaking should be blocked."""
        assert self.filter.should_interrupt("yeah okay", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("mmhmm right", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("ok uh huh", agent_is_speaking=True) == False
    
    def test_backchannel_with_punctuation(self):
        """Backchannel with punctuation should still be recognized."""
        assert self.filter.should_interrupt("yeah!", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("ok...", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("hmm,", agent_is_speaking=True) == False
    
    # Test Scenario 2: Command words during agent speech -> SHOULD interrupt
    def test_command_during_agent_speech(self):
        """Command words while agent is speaking should interrupt."""
        assert self.filter.should_interrupt("stop", agent_is_speaking=True) == True
        assert self.filter.should_interrupt("wait", agent_is_speaking=True) == True
        assert self.filter.should_interrupt("no", agent_is_speaking=True) == True
        assert self.filter.should_interrupt("pause", agent_is_speaking=True) == True
    
    def test_command_phrases(self):
        """Multi-word command phrases should interrupt."""
        assert self.filter.should_interrupt("wait a second", agent_is_speaking=True) == True
        assert self.filter.should_interrupt("hold on", agent_is_speaking=True) == True
        assert self.filter.should_interrupt("stop please", agent_is_speaking=True) == True
    
    # Test Scenario 3: Backchannel when agent is silent -> should process normally
    def test_backchannel_when_agent_silent(self):
        """Backchannel when agent is not speaking should be processed."""
        assert self.filter.should_interrupt("yeah", agent_is_speaking=False) == True
        assert self.filter.should_interrupt("ok", agent_is_speaking=False) == True
        assert self.filter.should_interrupt("hmm", agent_is_speaking=False) == True
    
    # Test Scenario 4: Mixed input -> should interrupt
    def test_mixed_backchannel_and_command(self):
        """Mixed backchannel + command should interrupt."""
        assert self.filter.should_interrupt("yeah but wait", agent_is_speaking=True) == True
        assert self.filter.should_interrupt("ok stop", agent_is_speaking=True) == True
        assert self.filter.should_interrupt("yeah okay but no", agent_is_speaking=True) == True
    
    def test_mixed_backchannel_and_content(self):
        """Backchannel + substantial content should interrupt."""
        assert self.filter.should_interrupt("yeah I have a question", agent_is_speaking=True) == True
        assert self.filter.should_interrupt("ok what about this", agent_is_speaking=True) == True
    
    # Test edge cases
    def test_empty_transcript(self):
        """Empty transcript should not interrupt."""
        assert self.filter.should_interrupt("", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("   ", agent_is_speaking=True) == False
    
    def test_case_insensitive(self):
        """Filter should be case-insensitive."""
        assert self.filter.should_interrupt("YEAH", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("Ok", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("HMM", agent_is_speaking=True) == False
        assert self.filter.should_interrupt("STOP", agent_is_speaking=True) == True
    
    def test_long_backchannel_sequence(self):
        """Very long sequence of backchannel words should be blocked."""
        long_backchannel = "yeah ok hmm right"
        assert self.filter.should_interrupt(long_backchannel, agent_is_speaking=True) == False
    
    def test_substantial_input_interrupts(self):
        """Substantial non-backchannel input should interrupt."""
        assert self.filter.should_interrupt("tell me about that", agent_is_speaking=True) == True
        assert self.filter.should_interrupt("can you explain", agent_is_speaking=True) == True
        assert self.filter.should_interrupt("I don't understand", agent_is_speaking=True) == True
    
    def test_word_threshold(self):
        """Very long phrases should interrupt even if they contain backchannel words."""
        # Default threshold is 5 words
        long_phrase = "yeah " * 6 + "something else"
        assert self.filter.should_interrupt(long_phrase, agent_is_speaking=True) == True
    
    # Test custom configuration
    def test_custom_backchannel_words(self):
        """Custom backchannel words should work."""
        custom_filter = InterruptionFilter(backchannel_words={"cool", "nice", "great"})
        assert custom_filter.should_interrupt("cool", agent_is_speaking=True) == False
        assert custom_filter.should_interrupt("yeah", agent_is_speaking=True) == True  # Not in custom set
    
    def test_custom_command_words(self):
        """Custom command words should work."""
        custom_filter = InterruptionFilter(command_words={"interrupt", "cancel"})
        assert custom_filter.should_interrupt("interrupt", agent_is_speaking=True) == True
        assert custom_filter.should_interrupt("stop", agent_is_speaking=True) == False  # Not in custom set
    
    # Test dynamic word management
    def test_add_backchannel_word(self):
        """Adding backchannel words dynamically should work."""
        self.filter.add_backchannel_word("cool")
        assert self.filter.should_interrupt("cool", agent_is_speaking=True) == False
    
    def test_add_command_word(self):
        """Adding command words dynamically should work."""
        self.filter.add_command_word("interrupt")
        assert self.filter.should_interrupt("interrupt", agent_is_speaking=True) == True
    
    def test_remove_backchannel_word(self):
        """Removing backchannel words should work."""
        self.filter.remove_backchannel_word("yeah")
        assert self.filter.should_interrupt("yeah", agent_is_speaking=True) == True
    
    def test_is_backchannel_word(self):
        """Checking individual words should work."""
        assert self.filter.is_backchannel_word("yeah") == True
        assert self.filter.is_backchannel_word("stop") == False
        assert self.filter.is_backchannel_word("hello") == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
