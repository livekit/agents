"""
Unit tests for Intelligent Interruption Handler
"""

import pytest
import asyncio
from interrupt_handler import (
    IntelligentInterruptionHandler,
    InterruptionType,
    InterruptionEvent
)


class TestInterruptionHandler:
    """Test suite for IntelligentInterruptionHandler"""
    
    @pytest.fixture
    def handler(self):
        """Create a handler instance for testing"""
        return IntelligentInterruptionHandler(
            ignored_words=['uh', 'umm', 'hmm', 'haan'],
            confidence_threshold=0.6,
            enable_logging=False
        )
    
    def test_initialization(self, handler):
        """Test handler initializes correctly"""
        assert handler.confidence_threshold == 0.6
        assert len(handler.ignored_words) >= 4
        assert 'uh' in handler.ignored_words
        assert not handler.is_agent_speaking()
    
    def test_agent_speaking_state(self, handler):
        """Test agent speaking state tracking"""
        assert not handler.is_agent_speaking()
        
        handler.set_agent_speaking(True)
        assert handler.is_agent_speaking()
        
        handler.set_agent_speaking(False)
        assert not handler.is_agent_speaking()
    
    def test_classify_filler_only(self, handler):
        """Test classification of filler-only input"""
        result = handler.classify_interruption("uh", 1.0)
        assert result == InterruptionType.FILLER
        
        result = handler.classify_interruption("umm hmm", 1.0)
        assert result == InterruptionType.FILLER
        
        result = handler.classify_interruption("uh uh uh", 1.0)
        assert result == InterruptionType.FILLER
    
    def test_classify_genuine(self, handler):
        """Test classification of genuine speech"""
        result = handler.classify_interruption("wait a second", 1.0)
        assert result == InterruptionType.GENUINE
        
        result = handler.classify_interruption("stop", 1.0)
        assert result == InterruptionType.GENUINE
        
        result = handler.classify_interruption("no not that one", 1.0)
        assert result == InterruptionType.GENUINE
    
    def test_classify_mixed(self, handler):
        """Test classification of mixed input"""
        result = handler.classify_interruption("umm okay stop", 1.0)
        assert result == InterruptionType.MIXED
        
        result = handler.classify_interruption("uh wait", 1.0)
        assert result == InterruptionType.MIXED
    
    @pytest.mark.asyncio
    async def test_filler_ignored_when_agent_speaking(self, handler):
        """Test fillers are ignored when agent is speaking"""
        handler.set_agent_speaking(True)
        
        should_process, event = await handler.should_process_interruption("umm", 1.0)
        
        assert not should_process
        assert event.type == InterruptionType.FILLER
        assert event.agent_was_speaking
    
    @pytest.mark.asyncio
    async def test_genuine_processed_when_agent_speaking(self, handler):
        """Test genuine speech stops agent"""
        handler.set_agent_speaking(True)
        
        should_process, event = await handler.should_process_interruption("wait stop", 1.0)
        
        assert should_process
        assert event.type == InterruptionType.GENUINE
        assert event.agent_was_speaking
    
    @pytest.mark.asyncio
    async def test_all_speech_processed_when_agent_quiet(self, handler):
        """Test all speech processed when agent is quiet"""
        handler.set_agent_speaking(False)
        
        # Even fillers are processed when agent is quiet
        should_process, event = await handler.should_process_interruption("umm", 1.0)
        assert should_process
        
        should_process, event = await handler.should_process_interruption("hello", 1.0)
        assert should_process
    
    @pytest.mark.asyncio
    async def test_low_confidence_ignored(self, handler):
        """Test low confidence speech is ignored"""
        handler.set_agent_speaking(True)
        
        should_process, event = await handler.should_process_interruption(
            "stop now", 
            confidence=0.3
        )
        
        assert not should_process
        assert event.confidence == 0.3
    
    @pytest.mark.asyncio
    async def test_mixed_input_processed(self, handler):
        """Test mixed input with command is processed"""
        handler.set_agent_speaking(True)
        
        should_process, event = await handler.should_process_interruption(
            "umm okay stop",
            1.0
        )
        
        assert should_process
        assert event.type == InterruptionType.MIXED
    
    def test_add_ignored_words(self, handler):
        """Test dynamically adding ignored words"""
        initial_count = len(handler.ignored_words)
        
        handler.add_ignored_words(['achha', 'theek'])
        
        assert len(handler.ignored_words) == initial_count + 2
        assert 'achha' in handler.ignored_words
        assert 'theek' in handler.ignored_words
    
    def test_remove_ignored_words(self, handler):
        """Test dynamically removing ignored words"""
        handler.add_ignored_words(['test'])
        assert 'test' in handler.ignored_words
        
        handler.remove_ignored_words(['test'])
        assert 'test' not in handler.ignored_words
    
    def test_case_insensitive_matching(self, handler):
        """Test case insensitive word matching"""
        result = handler.classify_interruption("UMM", 1.0)
        assert result == InterruptionType.FILLER
        
        result = handler.classify_interruption("Uh Hmm", 1.0)
        assert result == InterruptionType.FILLER
    
    def test_statistics_tracking(self, handler):
        """Test statistics are tracked correctly"""
        stats = handler.get_stats()
        assert stats['total_interruptions'] == 0
        assert stats['filler_ignored'] == 0
        assert stats['genuine_processed'] == 0
        
        handler.set_agent_speaking(True)
        asyncio.run(handler.should_process_interruption("umm", 1.0))
        
        stats = handler.get_stats()
        assert stats['total_interruptions'] == 1
        assert stats['filler_ignored'] == 1
    
    def test_reset_statistics(self, handler):
        """Test statistics reset"""
        handler.set_agent_speaking(True)
        asyncio.run(handler.should_process_interruption("umm", 1.0))
        
        handler.reset_stats()
        stats = handler.get_stats()
        
        assert stats['total_interruptions'] == 0
        assert stats['filler_ignored'] == 0
    
    def test_empty_transcript(self, handler):
        """Test handling of empty transcript"""
        result = handler.classify_interruption("", 1.0)
        assert result == InterruptionType.FILLER
        
        result = handler.classify_interruption("   ", 1.0)
        assert result == InterruptionType.FILLER
    
    def test_punctuation_handling(self, handler):
        """Test transcript with punctuation"""
        result = handler.classify_interruption("wait!", 1.0)
        assert result == InterruptionType.GENUINE
        
        result = handler.classify_interruption("umm...", 1.0)
        assert result == InterruptionType.FILLER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])