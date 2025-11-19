"""
Core interruption filtering logic.
Determines whether a transcription event should trigger an agent interruption.
"""

import logging
from enum import Enum
from typing import Optional

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class InterruptionDecision(Enum):
    """Possible decisions for handling interruptions."""

    ALLOW = "allow"  # Allow the interruption
    IGNORE = "ignore"  # Ignore as filler word
    PASS_THROUGH = "pass_through"  # Agent not speaking, pass through


class InterruptionFilter:
    """Filters interruptions based on configured rules."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self._agent_speaking = False

    def set_agent_speaking_state(self, is_speaking: bool) -> None:
        """Update the agent's speaking state."""
        self._agent_speaking = is_speaking

    def is_agent_speaking(self) -> bool:
        """Check if agent is currently speaking."""
        return self._agent_speaking

    def should_allow_interruption(
        self, transcription_text: str, confidence: float, is_final: bool = False
    ) -> InterruptionDecision:
        """
        Determine if the transcription should trigger an interruption.

        Args:
            transcription_text: The transcribed text
            confidence: ASR confidence score (0-1)
            is_final: Whether this is a final transcription

        Returns:
            InterruptionDecision indicating the action to take
        """
        # If agent is not speaking, pass through all events
        if not self._agent_speaking:
            logger.debug(f"Agent not speaking, passing through: '{transcription_text}'")
            return InterruptionDecision.PASS_THROUGH

        # Normalize the text
        normalized_text = transcription_text.lower().strip()

        # Empty transcriptions should be ignored
        if not normalized_text:
            return InterruptionDecision.IGNORE

        # Check if it's an ignored word/phrase
        is_ignored = self.config.is_ignored_word(normalized_text)
        confidence_threshold = self.config.get_confidence_threshold()

        # Decision logic
        if is_ignored:
            # For ignored words, check confidence
            # Only ignore if confidence is below threshold (uncertain ASR)
            # OR if it's not a final transcription
            if confidence < confidence_threshold or not is_final:
                logger.info(
                    f"Ignoring filler word: '{transcription_text}' "
                    f"(confidence: {confidence:.2f}, final: {is_final})"
                )
                return InterruptionDecision.IGNORE
            else:
                # High confidence on a filler word that's final - might be intentional
                # e.g., user repeatedly saying "wait wait wait"
                logger.info(
                    f"Allowing high-confidence filler: '{transcription_text}' "
                    f"(confidence: {confidence:.2f})"
                )
                return InterruptionDecision.ALLOW
        else:
            # Not an ignored word - this is a real interruption
            logger.info(
                f"Allowing interruption: '{transcription_text}' (confidence: {confidence:.2f})"
            )
            return InterruptionDecision.ALLOW

    def log_decision(
        self,
        decision: InterruptionDecision,
        transcription_text: str,
        confidence: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log the interruption decision for debugging."""
        log_data = {
            "decision": decision.value,
            "text": transcription_text,
            "confidence": confidence,
            "agent_speaking": self._agent_speaking,
        }

        if metadata:
            log_data.update(metadata)

        logger.debug(f"Interruption decision: {log_data}")
