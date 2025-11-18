"""
Agent State Tracker
Monitors agent speaking state through event listeners
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime
from collections import deque


logger = logging.getLogger(__name__)


class AgentStateTracker:
    """
    Tracks agent speaking state in real-time using event listeners.
    Thread-safe implementation using asyncio.Lock.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize the state tracker
        
        Args:
            max_history: Maximum number of state changes to keep in history
        """
        self._is_speaking = False
        self._lock = asyncio.Lock()
        self._state_history = deque(maxlen=max_history)
        logger.info("AgentStateTracker initialized")
    
    async def set_speaking(self, is_speaking: bool, timestamp: Optional[datetime] = None):
        """
        Update agent speaking state (thread-safe)
        
        Args:
            is_speaking: Whether the agent is currently speaking
            timestamp: Timestamp of the state change (defaults to now)
        """
        async with self._lock:
            if self._is_speaking != is_speaking:
                old_state = self._is_speaking
                self._is_speaking = is_speaking
                
                # Record state change
                ts = timestamp or datetime.now()
                self._state_history.append({
                    'timestamp': ts,
                    'old_state': 'speaking' if old_state else 'listening',
                    'new_state': 'speaking' if is_speaking else 'listening'
                })
                
                logger.debug(
                    f"Agent state changed: "
                    f"{'listening' if old_state else 'speaking'} -> "
                    f"{'speaking' if is_speaking else 'listening'} at {ts}"
                )
    
    async def is_speaking(self) -> bool:
        """
        Get current speaking state (thread-safe)
        
        Returns:
            True if agent is currently speaking, False otherwise
        """
        async with self._lock:
            return self._is_speaking
    
    def is_speaking_sync(self) -> bool:
        """
        Get current speaking state (synchronous, non-thread-safe)
        Use only when you cannot await (e.g., in non-async contexts)
        
        Returns:
            True if agent is currently speaking, False otherwise
        """
        return self._is_speaking
    
    async def get_history(self, limit: Optional[int] = None):
        """
        Get state change history
        
        Args:
            limit: Maximum number of entries to return (None for all)
        
        Returns:
            List of state change records
        """
        async with self._lock:
            history = list(self._state_history)
            if limit:
                return history[-limit:]
            return history
    
    def attach_to_session(self, session):
        """
        Attach event listeners to an AgentSession
        
        Args:
            session: LiveKit AgentSession instance
        """
        @session.on("agent_state_changed")
        def on_agent_state_changed(event):
            """Handler for agent state changes"""
            # The event contains state information
            is_speaking = event.state == "speaking"
            
            # Schedule the async state update
            asyncio.create_task(self.set_speaking(is_speaking))
            
            logger.debug(f"Agent state event: {event.state}")
        
        logger.info("State tracker attached to session")
