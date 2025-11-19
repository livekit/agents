"""
LiveKit Voice Interruption Filter
Intelligently handles interruptions by filtering filler words.
"""
from .config_manager import ConfigManager
from .interruption_filter import InterruptionFilter, InterruptionDecision
from .agent_wrapper import with_interruption_filter, InterruptionFilteredSession

__all__ = [
    'ConfigManager',
    'InterruptionFilter',
    'InterruptionDecision',
    'with_interruption_filter',
    'InterruptionFilteredSession',
]

__version__ = '1.0.0'
