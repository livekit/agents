"""
Helper functions to add interruption filtering to any AgentSession.
"""

import logging
from typing import Optional

from livekit.agents import AgentSession

from .config_manager import ConfigManager
from .interruption_filter import InterruptionFilter

logger = logging.getLogger(__name__)


def add_interruption_filter(
    session: AgentSession,
    config_manager: Optional[ConfigManager] = None,
    interruption_filter: Optional[InterruptionFilter] = None,
) -> AgentSession:
    """
    Add interruption filtering capability to an existing AgentSession.

    Usage:
        session = AgentSession(vad=vad, stt=stt, llm=llm, tts=tts)
        session = add_interruption_filter(session)
    """
    config = config_manager or ConfigManager()
    filter_obj = interruption_filter or InterruptionFilter(config)

    # Store filter on the session
    session._interruption_filter = filter_obj
    session._interruption_config = config

    logger.info("Interruption filtering enabled for AgentSession")
    logger.info(f"Ignored words: {config.get_ignored_words()}")
    logger.info(f"Confidence threshold: {config.get_confidence_threshold()}")

    return session
