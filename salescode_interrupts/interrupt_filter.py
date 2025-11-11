# salescode_interrupts/interrupt_filter.py
# ---------------------------------------------------------------------------
# Core interruption logic used by the voice agent
# ---------------------------------------------------------------------------

from __future__ import annotations
import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, List, Optional

log = logging.getLogger("salescode.interrupts")

# ---------------------------------------------------------------------------
# ✅ Public keyword lists (easy to extend to ANY language)
# ---------------------------------------------------------------------------
IGNORED_FILLERS = [
    "uh", "umm", "hmm", "haan"
]

INTERRUPT_COMMANDS = [
    "stop", "wait", "pause", "hold on", "one second",
    "ruko", "thamba", "bas", "ek second",
    "no", "go back"
]


# ---------------------------------------------------------------------------
# ✅ Data classes
# ---------------------------------------------------------------------------
@dataclass
class ASRSegment:
    text: str
    is_final: bool
    confidence: Optional[float]
    language: Optional[str] = None


@dataclass
class Decision:
    allow_interrupt: bool
    reason: str
    cleaned_text: str = ""
    matched_command: Optional[str] = None


# ---------------------------------------------------------------------------
# ✅ Helpers
# ---------------------------------------------------------------------------
def _normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip().lower()


def _tokenize(s: str):
    return [t for t in re.split(r"[^\w\u0900-\u097F]+", s) if t]


# ---------------------------------------------------------------------------
# ✅ Interrupt Filter
# ---------------------------------------------------------------------------
@dataclass
class InterruptFilter:
    # Runtime callbacks
    is_agent_speaking: Callable[[], bool]
    on_valid_interrupt: Callable[[ASRSegment, Decision], None]
    on_ignored_filler: Callable[[ASRSegment, Decision], None]
    on_speech_when_quiet: Callable[[ASRSegment], None]
    stop_agent_speaking: Optional[Callable[[], None]] = None

    # Dynamic keyword lists
    ignored_words: List[str] = field(default_factory=list)
    interrupt_commands: List[str] = field(default_factory=list)

    # Parameters
    min_confidence: float = 0.6

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        # Normalize words
        self._ignored_set = {_normalize(w) for w in self.ignored_words}
        self._cmd_set = {_normalize(w) for w in self.interrupt_commands}

    # -------------------------------------------------------------------
    # ✅ Main ASR handler
    # -------------------------------------------------------------------
    async def handle_asr_segment(self, seg: ASRSegment):
        async with self._lock:
            speaking = self.is_agent_speaking()

            # ✅ If agent is silent → ANY user speech counts
            if not speaking:
                self.on_speech_when_quiet(seg)
                return

            # ✅ Agent is speaking → decide
            decision = self._decide(seg)

            if decision.allow_interrupt:
                if self.stop_agent_speaking:
                    self.stop_agent_speaking()
                self.on_valid_interrupt(seg, decision)
            else:
                self.on_ignored_filler(seg, decision)

    # -------------------------------------------------------------------
    # ✅ Decision logic
    # -------------------------------------------------------------------
    def _decide(self, seg: ASRSegment) -> Decision:
        txt_raw = _normalize(seg.text)
        tokens = [_normalize(t) for t in _tokenize(txt_raw)]
        conf = seg.confidence or 1.0

        # ✅ Command detection (highest priority)
        for cmd in self._cmd_set:
            if cmd in txt_raw:
                return Decision(True, "matched_command", txt_raw, matched_command=cmd)

        # ✅ Ignore low-confidence murmurs
        if conf < self.min_confidence:
            return Decision(False, f"low_conf({conf:.2f})", txt_raw)

        # ✅ Remove fillers
        meaningful = [t for t in tokens if t not in self._ignored_set]

        if not meaningful:
            return Decision(False, "only_filler", "")

        # ✅ Genuine non-filler speech
        return Decision(True, "non_filler", " ".join(meaningful))
