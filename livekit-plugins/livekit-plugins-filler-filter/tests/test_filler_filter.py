"""
Unit tests for Filler Filter
Run with: pytest tests/test_filler_filter.py -v
"""
import pytest
import asyncio
from livekit.agents import stt

from livekit_plugins.filler_filter.config import FillerFilterConfig, reset_config
from livekit_plugins.filler_filter.state_tracker import AgentStateTracker
from livekit_plugins.filler_filter.utils import (
    normalize_text,
    is_filler_only,
    contains_command,
    should_filter
)
from livekit_plugins.filler_filter.filler_filter import FillerFilterWrapper


class TestConfiguration:
    """Test configuration management"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = reset_config()
        assert 'uh' in config.ignored_words
        assert 'um' in config.ignored_words
        assert config.confidence_threshold == 0.3
        assert config.enabled == True
    
    @pytest.mark.asyncio
    async def test_runtime_update(self):
        """Test runtime configuration updates"""
        config = reset_config()
        
        # Add words
        await config.add_words(['test1', 'test2'])
        assert 'test1' in config.ignored_words
        
        # Remove words
        await config.remove_words(['test1'])
        assert 'test1' not in config.ignored_words
        assert 'test2' in config.ignored_words


class TestUtils:
    """Test utility functions"""
    
    def test_normalize_text(self):
        """Test text normalization"""
        assert normalize_text("Hello, World!") == "hello world"
        assert normalize_text("  Multiple   Spaces  ") == "multiple spaces"
    
    def test_is_filler_only(self):
        """Test filler detection"""
        filler_words = {'uh', 'um', 'hmm'}
        
        # Filler only
        assert is_filler_only("uh", filler_words) == True
        assert is_filler_only("um hmm", filler_words) == True
        assert is_filler_only("uh um uh", filler_words) == True
        
        # Contains content
        assert is_filler_only("uh hello", filler_words) == False
        assert is_filler_only("wait um", filler_words) == False
    
    def test_contains_command(self):
        """Test command word detection"""
        assert contains_command("wait a second") == True
        assert contains_command("stop talking") == True
        assert contains_command("no not that") == True
        assert contains_command("uh hmm") == False
    
    def test_should_filter_logic(self):
        """Test filtering decision logic"""
        filler_words = {'uh', 'um', 'hmm'}
        
        # Agent not speaking - never filter
        assert should_filter("uh", 0.9, False, filler_words) == False
        
        # Agent speaking + filler only - filter
        assert should_filter("uh", 0.9, True, filler_words) == True
        
        # Agent speaking + contains command - don't filter
        assert should_filter("wait um", 0.9, True, filler_words) == False
        
        # Low confidence - filter
        assert should_filter("anything", 0.1, True, filler_words) == True


@pytest.mark.asyncio
class TestStateTracker:
    """Test agent state tracking"""
    
    async def test_state_changes(self):
        """Test state change tracking"""
        tracker = AgentStateTracker()
        
        # Initial state
        assert await tracker.is_speaking() == False
        
        # Change to speaking
        await tracker.set_speaking(True)
        assert await tracker.is_speaking() == True
        
        # Change to listening
        await tracker.set_speaking(False)
        assert await tracker.is_speaking() == False
    
    async def test_history_tracking(self):
        """Test state change history"""
        tracker = AgentStateTracker(max_history=10)
        
        # Make some state changes
        await tracker.set_speaking(True)
        await tracker.set_speaking(False)
        await tracker.set_speaking(True)
        
        history = await tracker.get_history()
        assert len(history) == 3
