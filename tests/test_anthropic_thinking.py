"""Test Anthropic thinking configuration functionality."""

import pytest
import os
import sys
from unittest.mock import patch

# Add the livekit-agents path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'livekit-agents'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'livekit-plugins', 'livekit-plugins-anthropic'))

try:
    from livekit.plugins.anthropic import LLM, ThinkingConfig, ThinkingConfigDict
    from livekit.agents.llm import ChatContext
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


def test_thinking_config_types():
    """Test that thinking configuration types are properly defined."""
    # Test ThinkingConfig TypedDict
    thinking_config: ThinkingConfig = {
        "type": "enabled",
        "budget_tokens": 5000
    }
    
    assert thinking_config["type"] == "enabled"
    assert thinking_config["budget_tokens"] == 5000
    
    # Test ThinkingConfigDict alias
    thinking_dict: ThinkingConfigDict = {
        "type": "enabled",
        "budget_tokens": 3000
    }
    
    assert thinking_dict["type"] == "enabled"
    assert thinking_dict["budget_tokens"] == 3000


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
def test_llm_creation_with_thinking():
    """Test that LLM can be created with thinking configuration."""
    
    # Test with dict configuration
    llm_with_dict = LLM(
        thinking={"type": "enabled", "budget_tokens": 5000}
    )
    
    assert llm_with_dict._opts.thinking is not None
    assert llm_with_dict._opts.thinking["type"] == "enabled"
    assert llm_with_dict._opts.thinking["budget_tokens"] == 5000
    
    # Test with TypedDict configuration
    thinking_config: ThinkingConfig = {
        "type": "enabled",
        "budget_tokens": 3000
    }
    
    llm_with_typed = LLM(thinking=thinking_config)
    
    assert llm_with_typed._opts.thinking is not None
    assert llm_with_typed._opts.thinking["type"] == "enabled"
    assert llm_with_typed._opts.thinking["budget_tokens"] == 3000
    
    # Test without thinking configuration
    llm_without_thinking = LLM()
    
    # Should have NOT_GIVEN value for thinking
    from livekit.agents.types import NOT_GIVEN
    assert llm_without_thinking._opts.thinking is NOT_GIVEN


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
def test_thinking_config_validation():
    """Test that thinking configuration is properly validated."""
    
    # Valid configurations should work
    valid_configs = [
        {"type": "enabled", "budget_tokens": 1000},
        {"type": "enabled", "budget_tokens": 50000},
    ]
    
    for config in valid_configs:
        llm = LLM(thinking=config)
        assert llm._opts.thinking == config
        
    # Test that LLM creation works with various budget token values
    for budget in [1024, 5000, 10000, 32000]:
        llm = LLM(thinking={"type": "enabled", "budget_tokens": budget})
        assert llm._opts.thinking["budget_tokens"] == budget 