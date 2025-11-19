"""
LiveKit Voice Interruption Filter
Intelligently handles interruptions by filtering filler words.
"""

from .agent_wrapper import InterruptionFilteredSession, with_interruption_filter
from .config_manager import ConfigManager
from .interruption_filter import InterruptionDecision, InterruptionFilter

__all__ = [
    "ConfigManager",
    "InterruptionFilter",
    "InterruptionDecision",
    "with_interruption_filter",
    "InterruptionFilteredSession",
]

__version__ = "1.0.0"
