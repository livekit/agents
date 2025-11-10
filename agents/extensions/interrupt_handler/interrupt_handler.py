"""
InterruptHandler Module
-----------------------
Enhances LiveKit agents to intelligently handle user voice interruptions.
It ignores filler words ("uh", "umm", "hmm", "haan") while the agent speaks,
but recognizes real commands ("wait", "stop", etc.) instantly.
"""

import asyncio
import re
import time
from enum import Enum
from typing import List, Dict, Optional

# Define decision types for clarity
class Decision(Enum):
    IGNORED = "ignored"
    INTERRUPT = "interrupt"
    FORWARDED = "forwarded"

# Helper: simple word tokenizer
def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())

# Default configuration
DEFAULT_IGNORED = ["uh", "umm", "hmm", "haan"]
DEFAULT_STOP_WORDS = ["stop", "wait", "hold on", "pause", "no not that one"]

class AgentState:
    """Keeps track of whether the agent is currently speaking."""
    def __init__(self):
        self._lock = asyncio.Lock()
        self._speaking = False

    async def set_speaking(self, value: bool):
        async with self._lock:
            self._speaking = value

    async def is_speaking(self) -> bool:
        async with self._lock:
            return self._speaking


class InterruptHandler:
    """
    Handles real-time voice transcripts from ASR (Automatic Speech Recognition)
    to decide whether the agent should ignore, interrupt, or forward the speech.
    """

    def __init__(
        self,
        ignored_words: Optional[List[str]] = None,
        stop_words: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        low_confidence_threshold: float = 0.35,
    ):
        self.ignored_words = set((ignored_words or DEFAULT_IGNORED))
        self.stop_words = set((stop_words or DEFAULT_STOP_WORDS))
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.agent_state = AgentState()
        self._logs: List[Dict] = []

    async def handle_transcript(
        self, text: str, confidence: float, is_final: bool = True, metadata: Optional[Dict] = None
    ) -> Decision:
        """Main logic to decide whether to IGNORE, INTERRUPT, or FORWARD."""
        metadata = metadata or {}
        tokens = tokenize(text)
        tokens = [t for t in tokens if len(t) > 1]  # new: ignore 1-letter tokens

        speaking = await self.agent_state.is_speaking()

        # Case 1: agent is speaking and user sound is low-confidence (background murmur)
        if speaking and confidence < self.low_confidence_threshold:
            self._log("IGNORED_LOW_CONF", text, confidence, metadata)
            return Decision.IGNORED
        
        def is_filler(word, ignored):
            for ig in ignored:
                if word.startswith(ig):  # e.g. "ummm" starts with "umm"
                    return True
            return False


        # Case 2: agent is speaking — check if it's filler or real interruption
        if speaking:
            # Remove filler tokens
            meaningful = [t for t in tokens if not is_filler(t, self.ignored_words)]
            if not meaningful:
                self._log("IGNORED_FILLER_ONLY", text, confidence, metadata)
                return Decision.IGNORED

            # If any stop/command word is present, interrupt immediately
            full_text = text.lower()
            for sw in self.stop_words:
                if sw in full_text:
                    self._log("INTERRUPT_STOP_WORD", text, confidence, metadata, {"matched": sw})
                    return Decision.INTERRUPT

            # Otherwise, treat as real interruption
            self._log("INTERRUPT_MEANINGFUL", text, confidence, metadata, {"tokens": meaningful})
            return Decision.INTERRUPT

        # Case 3: agent is NOT speaking — normal user speech
        if confidence < (self.confidence_threshold * 0.5):
            self._log("FORWARDED_LOW_CONF", text, confidence, metadata)
            return Decision.FORWARDED

        self._log("FORWARDED", text, confidence, metadata)
        return Decision.FORWARDED

    def _log(self, tag: str, text: str, conf: float, meta: Dict, extra: Dict = None):
        """Internal structured logging."""
        entry = {
            "event": tag,
            "text": text,
            "confidence": conf,
            "metadata": meta,
            "extra": extra or {},
            "timestamp": time.time(),
        }
        self._logs.append(entry)
        print(f"[{tag}] text='{text}' conf={conf:.2f} extra={extra or {}}")

    def get_logs(self) -> List[Dict]:
        """Returns logs for debugging."""
        return self._logs

