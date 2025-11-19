"""
Middleware to add interruption filtering to LiveKit AgentSession.
"""
import logging
from typing import Optional
from functools import wraps

from .interruption_filter import InterruptionFilter, InterruptionDecision
from .config_manager import ConfigManager


logger = logging.getLogger(__name__)


def with_interruption_filter(
    config_manager: Optional[ConfigManager] = None,
    interruption_filter: Optional[InterruptionFilter] = None
):
    """
    Decorator to add interruption filtering to an agent entrypoint.
    
    Usage:
        @with_interruption_filter()
        async def entrypoint(ctx: JobContext):
            # Your agent code here
    """
    def decorator(entrypoint_func):
        @wraps(entrypoint_func)
        async def wrapper(ctx):
            # Initialize filter
            config = config_manager or ConfigManager()
            filter_obj = interruption_filter or InterruptionFilter(config)
            
            # Store in context for access in the entrypoint
            ctx.interruption_filter = filter_obj
            ctx.interruption_config = config
            
            logger.info("🎤 Interruption filtering enabled")
            logger.info(f"   Ignored words: {', '.join(list(config.get_ignored_words())[:5])}...")
            logger.info(f"   Confidence threshold: {config.get_confidence_threshold()}")
            
            # Call original entrypoint
            return await entrypoint_func(ctx)
        
        return wrapper
    return decorator


class InterruptionFilteredSession:
    """
    Wrapper class that can be used to manually add filtering to a session.
    This doesn't extend AgentSession, just provides helper methods.
    """
    
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        interruption_filter: Optional[InterruptionFilter] = None
    ):
        self.config = config_manager or ConfigManager()
        self.filter = interruption_filter or InterruptionFilter(self.config)
        self._is_speaking = False
    
    def should_interrupt(self, text: str, confidence: float = 0.8, is_final: bool = True) -> bool:
        """
        Check if the given text should trigger an interruption.
        
        Returns:
            True if interruption should be allowed, False otherwise
        """
        decision = self.filter.should_allow_interruption(
            transcription_text=text,
            confidence=confidence,
            is_final=is_final
        )
        
        self.filter.log_decision(
            decision=decision,
            transcription_text=text,
            confidence=confidence,
            metadata={'is_final': is_final}
        )
        
        if decision == InterruptionDecision.IGNORE:
            logger.info(f"🚫 Filtered filler word: '{text}'")
            return False
        elif decision == InterruptionDecision.ALLOW:
            logger.info(f"✅ Allowing interruption: '{text}'")
            return True
        else:  # PASS_THROUGH
            return True
    
    def set_speaking(self, is_speaking: bool):
        """Update agent speaking state."""
        self._is_speaking = is_speaking
        self.filter.set_agent_speaking_state(is_speaking)
        logger.debug(f"Agent speaking state: {is_speaking}")
    
    def get_config(self) -> ConfigManager:
        """Get the configuration manager."""
        return self.config
    
    def get_filter(self) -> InterruptionFilter:
        """Get the interruption filter."""
        return self.filter
