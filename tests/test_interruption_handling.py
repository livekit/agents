"""
Tests for interruption filtering functionality.
"""
import pytest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.interruption_filter import (
    ConfigManager,
    InterruptionFilter,
    InterruptionDecision,
    InterruptionFilteredSession,
)


class TestConfigManager:
    """Test configuration management."""
    
    def test_default_ignored_words(self):
        """Test that default ignored words are loaded."""
        config = ConfigManager()
        ignored = config.get_ignored_words()
        
        assert 'uh' in ignored
        assert 'umm' in ignored
        assert 'haan' in ignored
    
    def test_is_ignored_word(self):
        """Test word checking is case-insensitive."""
        config = ConfigManager()
        
        assert config.is_ignored_word('UH')
        assert config.is_ignored_word('Umm')
        assert config.is_ignored_word('  hmm  ')
    
    def test_confidence_threshold_from_env(self, monkeypatch):
        """Test loading confidence threshold from environment."""
        monkeypatch.setenv('CONFIDENCE_THRESHOLD', '0.8')
        config = ConfigManager()
        
        assert config.get_confidence_threshold() == 0.8


class TestInterruptionFilter:
    """Test interruption filtering logic."""
    
    def test_pass_through_when_agent_not_speaking(self):
        """Test that all events pass through when agent is quiet."""
        filter = InterruptionFilter()
        filter.set_agent_speaking_state(False)
        
        decision = filter.should_allow_interruption('uh', 0.9, True)
        assert decision == InterruptionDecision.PASS_THROUGH
    
    def test_ignore_low_confidence_filler(self):
        """Test that low confidence fillers are ignored."""
        filter = InterruptionFilter()
        filter.set_agent_speaking_state(True)
        
        decision = filter.should_allow_interruption('umm', 0.3, False)
        assert decision == InterruptionDecision.IGNORE
    
    def test_allow_high_confidence_filler(self):
        """Test that high confidence fillers can interrupt."""
        filter = InterruptionFilter()
        filter.set_agent_speaking_state(True)
        
        decision = filter.should_allow_interruption('hmm', 0.95, True)
        assert decision == InterruptionDecision.ALLOW
    
    def test_allow_non_filler_interruption(self):
        """Test that non-filler words trigger interruption."""
        filter = InterruptionFilter()
        filter.set_agent_speaking_state(True)
        
        decision = filter.should_allow_interruption('wait', 0.8, True)
        assert decision == InterruptionDecision.ALLOW
        
        decision = filter.should_allow_interruption('stop', 0.7, True)
        assert decision == InterruptionDecision.ALLOW
    
    def test_ignore_empty_transcription(self):
        """Test that empty transcriptions are ignored."""
        filter = InterruptionFilter()
        filter.set_agent_speaking_state(True)
        
        decision = filter.should_allow_interruption('', 0.9, True)
        assert decision == InterruptionDecision.IGNORE
        
        decision = filter.should_allow_interruption('   ', 0.9, True)
        assert decision == InterruptionDecision.IGNORE


class TestDynamicConfiguration:
    """Test dynamic configuration updates."""
    
    def test_add_ignored_word(self, monkeypatch):
        """Test adding words dynamically."""
        monkeypatch.setenv('ENABLE_DYNAMIC_UPDATES', 'true')
        config = ConfigManager()
        
        config.add_ignored_word('okay')
        assert config.is_ignored_word('okay')
    
    def test_remove_ignored_word(self, monkeypatch):
        """Test removing words dynamically."""
        monkeypatch.setenv('ENABLE_DYNAMIC_UPDATES', 'true')
        config = ConfigManager()
        
        config.remove_ignored_word('uh')
        assert not config.is_ignored_word('uh')
    
    def test_dynamic_updates_disabled_by_default(self):
        """Test that dynamic updates throw error when disabled."""
        config = ConfigManager()
        
        with pytest.raises(RuntimeError):
            config.add_ignored_word('test')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
