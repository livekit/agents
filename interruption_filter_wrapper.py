import asyncio
import logging
from typing import Callable

from interruption_handler import IntelligentInterruptionHandler
from config import InterruptionConfig

logger = logging.getLogger(__name__)

class InterruptionFilterWrapper:
    """
    Wrapper that adds interruption filtering as a layer on top of LiveKit AgentSession.
    Does NOT modify core agent logic - just intercepts and filters events.
    """
    
    def __init__(self, session, config: InterruptionConfig = None):
        self.session = session
        self.config = config or InterruptionConfig.from_env()
        self.handler = IntelligentInterruptionHandler(self.config)
        
        logger.info(f"Interruption filter initialized with ignored words: {self.config.ignored_words}")
    
    async def set_agent_speaking(self, speaking: bool):
        """Update agent speaking state"""
        await self.handler.set_agent_speaking(speaking)
    
    async def filter_user_speech(self, transcription: str, confidence: float = 1.0) -> bool:
        """
        Filter layer for user speech. Call this before processing user input.
        
        Returns:
            True: Allow this speech to interrupt/proceed
            False: Ignore this speech (filler detected)
        """
        return await self.handler.should_interrupt(transcription, confidence)
    
    def get_logs(self):
        """Get event logs for debugging"""
        return self.handler.get_event_log()
    
    def update_ignored_words(self, new_words: list):
        """Dynamically update ignored words (bonus feature)"""
        self.handler.update_ignored_words(new_words)
        