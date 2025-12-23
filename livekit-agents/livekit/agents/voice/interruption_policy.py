import re
from typing import Iterable, Optional

# User's default list
DEFAULT_BACKCHANNELS = [
    "mm-hmm", "uh-huh", "hmm", "mhm", "hm", "uh", "um", "huh", "mm", "ah", "oh", "eh", "mhmm", "mmhmm", "mmm", "aha", "yup", "ya",
    "yeah", "yep", "yes", "yea", "ok", "okay",
    "right", "sure", "alright", "cool", "fine",
    "i see", "got it", "gotcha", "understood",
]

# Extended colloquial/agreement list
EXTENDED_BACKCHANNELS = [
    "exactly", "precisely", "indeed", "certainly",
    "fair enough", "that's true", "that is true", "true", "valid point",
    "i agree", "agreed", "no doubt", "absolutely", "absolutely correct",
    "for sure", "no way", "right on", "you bet",
    "go on", "continue", "keep going", "and then",
    "interesting", "nice", "wow", "really", "wonderful", "great", "amazing", "awesome"
]

class InterruptionPolicy:
    def __init__(self, backchannels: Optional[list[str]] = None):
        source = backchannels if backchannels is not None else (DEFAULT_BACKCHANNELS + EXTENDED_BACKCHANNELS)
        normalized = self.normalize_all(source)
        self._bc_phrases: set[str] = set(normalized)
        # Tokens are single-word backchannels
        self._bc_tokens: set[str] = {p for p in self._bc_phrases if " " not in p}

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text using regex:
        - Lowercase & strip
        - Remove non-alphanumeric chars (except spaces)
        - Collapse repeated characters (e.g., "yeeaah" -> "yeah")
        - Collapse whitespace
        """
        t = text.lower().strip()
        t = re.sub(r"[-_]+", " ", t)       # Substitute Hyphens
        t = re.sub(r"[^a-z0-9\s]", "", t)  # Drop other punctuation
        t = re.sub(r"(.)\1{2,}", r"\1\1", t) # Keep max 2 repetitions (e.g. "mmmmm" -> "mm")
        t = re.sub(r"\s+", " ", t).strip() # Change multiple whitespace to single whitespace
        return t

    @classmethod
    def normalize_all(cls, items: Iterable[str]) -> list[str]:
        out: list[str] = []
        for x in items:
            nx = cls.normalize_text(x)
            if nx:
                out.append(nx)
        return out

    def _is_backchannel_only(self, transcript: str) -> bool:
        normalized = self.normalize_text(transcript)
        if not normalized:
            return False # Empty transcript is usually not an interruption context, but here return False prevents bypass? 
            # Wait, if it's empty, STT usually doesn't fire. If it does, and we return False (Not Backchannel), we Interrupt?
            # Actually if empty, we probably shouldn't interrupt at all, but `should_interrupt` logic handles that.
            # Let's say Empty -> Not Backchannel -> Interrupt? No.
            # If normalized is empty, it implies noise. 
            # Let's return True (treat as backchannel/noise) to IGNORER interruption for empty/noise text.
            return True

        if normalized in self._bc_phrases:
            return True

        words = normalized.split()
        if not words:
            return True

        # Check if EVERY word is a backchannel token
        return all(word in self._bc_tokens for word in words)

    def should_interrupt(self, transcript: str) -> bool:
        """
        Public API called by AgentActivity.
        Returns True if the transcript should trigger an interruption.
        Returns False if it should be ignored (backchannel).
        """
        if not transcript:
            return False
            
        is_backchannel = self._is_backchannel_only(transcript)
        # If it IS a backchannel, DO NOT interrupt (False).
        # If it is NOT a backchannel, DO interrupt (True).
        return not is_backchannel
