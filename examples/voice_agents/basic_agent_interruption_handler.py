# interruption_handler.py
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Set

logger = logging.getLogger("interrupt-filter")

def _split_csv(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]

@dataclass
class InterruptionPolicy:
    """Configurable policy for distinguishing fillers vs. real interruptions."""
    ignored_fillers: Set[str] = field(default_factory=lambda: {"uh", "umm", "um", "hmm", "haan", "huh"})
    command_keywords: Set[str] = field(default_factory=lambda: {
        "stop", "wait", "hold on", "pause", "no", "not that", "one second",
        "listen", "excuse me", "can i", "okay stop", "hang on"
    })
    min_confidence: float = 0.55  # below => treat as noise when agent is speaking
    mixed_requires_command: bool = True  # if fillers + command => interrupt

    @classmethod
    def from_env(cls) -> "InterruptionPolicy":
        fillers = set(_split_csv(os.getenv("IGNORED_FILLERS")))
        commands = set(_split_csv(os.getenv("INTERRUPT_COMMANDS")))
        min_conf = os.getenv("ASR_MIN_CONFIDENCE")
        return cls(
            ignored_fillers=(fillers or cls().ignored_fillers),
            command_keywords=(commands or cls().command_keywords),
            min_confidence=float(min_conf) if min_conf else cls().min_confidence,
        )

class InterruptionDecision:
    IGNORE = "ignore"          # ignore & (optionally) resume TTS
    INTERRUPT = "interrupt"    # stop the agent immediately
    REGISTER = "register"      # accept as normal speech event (when agent silent)

class InterruptionFilter:
    """
    Stateless text-level filter. You tell it:
      - the ASR text and confidence
      - whether the agent is currently speaking
    and it tells you what to do.
    """
    def __init__(self, policy: InterruptionPolicy | None = None):
        self.policy = policy or InterruptionPolicy.from_env()

    def update_policy(
        self,
        ignored_fillers: Iterable[str] | None = None,
        command_keywords: Iterable[str] | None = None,
        min_confidence: float | None = None,
    ):
        if ignored_fillers is not None:
            self.policy.ignored_fillers = {w.strip().lower() for w in ignored_fillers if w.strip()}
        if command_keywords is not None:
            self.policy.command_keywords = {w.strip().lower() for w in command_keywords if w.strip()}
        if min_confidence is not None:
            self.policy.min_confidence = float(min_confidence)

    def _normalize(self, text: str) -> str:
        # Lowercase + strip punctuation except spaces
        text = text.strip().lower()
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _token_set(self, text: str) -> Set[str]:
        return set(self._normalize(text).split())

    def is_fillers_only(self, text: str) -> bool:
        norm = self._normalize(text)
        if not norm:
            return True
        toks = self._token_set(norm)
        # If every token is an ignored filler, treat as filler-only
        return len(toks) > 0 and toks.issubset(self.policy.ignored_fillers)

    def contains_command(self, text: str) -> bool:
        norm = self._normalize(text)
        if not norm:
            return False
        # Match either keywords as tokens or phrase substrings
        for cmd in self.policy.command_keywords:
            if " " in cmd:
                if cmd in norm:
                    return True
            else:
                if cmd in self._token_set(norm):
                    return True
        return False

    def decide(self, *, text: str, confidence: float | None, agent_speaking: bool) -> str:
        conf = confidence if confidence is not None else 1.0

        if not agent_speaking:
            # Agent is quiet => register speech (even if it's a filler)
            return InterruptionDecision.REGISTER

        # Agent is speaking:
        if conf < self.policy.min_confidence:
            logger.debug(f"ASR confidence {conf:.2f} below {self.policy.min_confidence:.2f} -> IGNORE")
            return InterruptionDecision.IGNORE

        if self.contains_command(text):
            logger.debug("Detected command while agent speaking -> INTERRUPT")
            return InterruptionDecision.INTERRUPT

        if self.is_fillers_only(text):
            logger.debug("Fillers-only while agent speaking -> IGNORE")
            return InterruptionDecision.IGNORE

        # Normal speech with adequate confidence -> INTERRUPT (real interruption)
        return InterruptionDecision.INTERRUPT
