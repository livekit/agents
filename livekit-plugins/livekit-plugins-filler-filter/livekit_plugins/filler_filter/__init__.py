"""
Filler Filter Package
Main exports for the filler filter module
"""
from .config import FillerFilterConfig, get_config, reset_config
from .state_tracker import AgentStateTracker
from .filler_filter import FillerFilterWrapper
from .utils import (
    normalize_text,
    extract_words,
    is_filler_only,
    contains_command,
    should_filter
)

__all__ = [
    'FillerFilterConfig',
    'get_config',
    'reset_config',
    'AgentStateTracker',
    'FillerFilterWrapper',
    'normalize_text',
    'extract_words',
    'is_filler_only',
    'contains_command',
    'should_filter',
]

__version__ = '1.0.0'
