import asyncio
import pytest
from unittest.mock import Mock, AsyncMock
from filler_filter import FillerWordFilter
from config import config

class TestFillerFilter:
    """Test cases for filler word filter"""
    
    @pytest.fixture
    def filter(self):
        stt_mock = Mock()
        tts_mock = Mock()
        return FillerWordFilter(stt_mock, tts_mock)
    
    @pytest.mark.asyncio
    async def test_filler_ignored_when_agent_speaking(self, filter):
        filter.set_agent_speaking(True)
        
        transcription = Mock()
        transcription.text = "umm hmm"
        transcription.confidence = 0.9
        
        should_interrupt, reason = filter.should_interrupt(transcription)
        assert not should_interrupt
        assert reason == "filler_words"
    
    @pytest.mark.asyncio
    async def test_interruption_trigger_detected(self, filter):
        filter.set_agent_speaking(True)
        
        transcription = Mock()
        transcription.text = "wait stop that"
        transcription.confidence = 0.9
        
        should_interrupt, reason = filter.should_interrupt(transcription)
        assert should_interrupt
        assert reason == "interruption_trigger"
    
    @pytest.mark.asyncio
    async def test_filler_allowed_when_agent_quiet(self, filter):
        filter.set_agent_speaking(False)
        
        transcription = Mock()
        transcription.text = "umm"
        transcription.confidence = 0.9
        
        should_interrupt, reason = filter.should_interrupt(transcription)
        assert should_interrupt
        assert reason == "agent_not_speaking"
    
    @pytest.mark.asyncio
    async def test_mixed_filler_and_command(self, filter):
        filter.set_agent_speaking(True)
        
        transcription = Mock()
        transcription.text = "umm okay stop now"
        transcription.confidence = 0.9
        
        should_interrupt, reason = filter.should_interrupt(transcription)
        assert should_interrupt
        assert reason == "interruption_trigger"
    
    @pytest.mark.asyncio
    async def test_low_confidence_ignored(self, filter):
        filter.set_agent_speaking(True)
        
        transcription = Mock()
        transcription.text = "some speech"
        transcription.confidence = 0.5  # Below threshold
        
        should_interrupt, reason = filter.should_interrupt(transcription)
        assert not should_interrupt
        assert reason == "low_confidence"

def test_dynamic_config_update():
    """Test dynamic configuration updates"""
    initial_count = len(config.ignored_words)
    
    # Add new filler word
    config.update_ignored_words(["newfiller"])
    assert "newfiller" in config.ignored_words
    assert len(config.ignored_words) == initial_count + 1
    
    # Remove filler word
    config.remove_ignored_word("newfiller")
    assert "newfiller" not in config.ignored_words

if __name__ == "__main__":
    # Run simple demonstration
    print("Filler Filter Test Demonstration")
    print(f"Ignored words: {config.ignored_words}")
    print(f"Interruption triggers: {config.interruption_triggers}")
    print(f"Confidence threshold: {config.confidence_threshold}")