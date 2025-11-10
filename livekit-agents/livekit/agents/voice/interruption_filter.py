import re
import os
import logging
from typing import Literal, List

Decision = Literal["IGNORE", "INTERRUPT", "PASS"]

class InterruptionFilter:
    """
    InterruptionFilter distinguishes between filler words and meaningful interruptions
    during live conversations.

    - Ignores fillers when the agent is speaking.
    - Allows genuine speech interruptions instantly.
    - Configurable filler list and confidence threshold.
    """

    def __init__(self, ignored_words: List[str] = None, confidence_threshold: float = 0.6):
        self.ignored_words = set(
            ignored_words or os.getenv("IGNORED_WORDS", "uh,umm,hmm,haan").split(",")
        )
        self.conf_threshold = confidence_threshold
        self.logger = logging.getLogger("InterruptionFilter")
        self.logger.setLevel(logging.INFO)

    def update_ignored_words(self, new_list: List[str]):
        """Dynamically update ignored filler words."""
        self.ignored_words = set(new_list)
        self.logger.info(f"Updated ignored words list: {self.ignored_words}")

    def _tokenize(self, text: str):
        """Extract alphanumeric tokens in lowercase."""
        return re.findall(r"\w+", text.lower())

    def is_filler_only(self, text: str) -> bool:
        """Return True if the entire transcript is composed of filler words."""
        words = self._tokenize(text)
        return len(words) > 0 and all(w in self.ignored_words for w in words)

    def contains_meaningful_word(self, text: str) -> bool:
        """Return True if transcript has at least one non-filler word."""
        words = self._tokenize(text)
        return any(w not in self.ignored_words for w in words)

    def evaluate(self, transcript: str, confidence: float, agent_speaking: bool) -> Decision:
        """
        Main evaluation logic:
        - When agent is speaking:
            - Ignore low-confidence or filler-only phrases.
            - Interrupt if contains any meaningful word.
        - When agent is quiet:
            - Always pass to downstream logic.
        """
        if not transcript.strip():
            return "PASS"

        text = transcript.strip().lower()

        if agent_speaking:
            if confidence < self.conf_threshold:
                self.logger.debug(f"Ignored low-confidence: {text}")
                return "IGNORE"

            if self.is_filler_only(text):
                self.logger.info(f"Ignored filler during speech: {text}")
                return "IGNORE"

            if self.contains_meaningful_word(text):
                self.logger.info(f"Detected valid interruption: {text}")
                return "INTERRUPT"

        # When agent is silent, we let normal processing continue
        return "PASS"
