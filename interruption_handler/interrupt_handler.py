import os
import re
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .logger import get_logger

logger = get_logger()

@dataclass
class TranscriptEvent:
    text: str
    confidence: Optional[float] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class InterruptHandler:
    def __init__(self,
        ignored_fillers=None,
        valid_interruptions=None,
        confidence_threshold=0.4
    ):
        # read env vars if available
        env_fillers = os.getenv("IGNORED_FILLERS")
        if env_fillers:
            ignored_fillers = [t.strip().lower() for t in env_fillers.split(",")]

        env_interruptions = os.getenv("VALID_INTERRUPTION_KEYWORDS")
        if env_interruptions:
            valid_interruptions = [t.strip().lower() for t in env_interruptions.split(",")]

        self.ignored_fillers = ignored_fillers or ["uh", "umm", "hmm", "haan"]
        self.valid_interruptions = valid_interruptions or ["stop", "wait", "hold on", "no not that one"]
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", confidence_threshold))

        # compile filler tokens
        self.filler_set = set([f.lower() for f in self.ignored_fillers])
        self.interrupt_set = [i.lower() for i in self.valid_interruptions]
        self._lock = asyncio.Lock()

        logger.info(f"InterruptHandler loaded with fillers={self.filler_set}")

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _contains_only_fillers(self, text: str) -> bool:
        tokens = self._normalize(text).split()
        return all(t in self.filler_set for t in tokens)

    def _contains_interruption(self, text: str) -> bool:
        norm = self._normalize(text)
        return any(k in norm for k in self.interrupt_set)

    async def on_transcript_event(self, agent_is_speaking: bool, event: TranscriptEvent):
        text = event.text
        conf = event.confidence

        # always interrupt if keyword present
        if self._contains_interruption(text):
            return ("interrupt", "keyword_detected", text)

        if agent_is_speaking:
            # low confidence → background murmur
            if conf is not None and conf < self.confidence_threshold:
                return ("ignore", "low_confidence", text)

            # filler-only → ignore
            if self._contains_only_fillers(text):
                return ("ignore", "filler_only", text)

            # non-filler or mixed → real interruption
            return ("interrupt", "mixed_content", text)

        # agent is quiet → treat as normal user input
        return ("register_speech", "agent_quiet", text)
