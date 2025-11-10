"""
Filler-aware interrupt handler extension.

Behavior:
- When agent is speaking:
    - If ASR segment contains only filler tokens (configurable), suppress it (do NOT stop TTS).
    - If ASR contains non-filler tokens or explicit stop/wait phrases, forward immediately (stop TTS).
    - Low-confidence ASR with filler tokens -> suppress.
- When agent is quiet:
    - Register all user speech (even fillers).
"""

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Optional, Callable, List, Set, Dict

@dataclass
class ASRResult:
    text: str
    confidence: Optional[float] = None
    is_final: bool = False
    language: Optional[str] = None
    timestamp: Optional[float] = None

class FillerFilter:
    def __init__(
        self,
        ignored_words: Optional[List[str]] = None,
        confidence_threshold: float = 0.6,
        partial_idle_ms: int = 250,
        log_fn: Optional[Callable[[str, Dict], None]] = None
    ):
        self._ignored_set: Set[str] = set(self._normalize(w) for w in (ignored_words or ['uh','umm','hmm','haan']))
        self._confidence_threshold = confidence_threshold
        self._partial_idle_ms = partial_idle_ms
        self._log = log_fn or (lambda lvl, meta: print(f"[{lvl}] {meta}"))
        self._agent_speaking = False
        self._buffer = ""             # accumulate partials while agent speaks
        self._lock = asyncio.Lock()
        self._idle_task = None

        # Callbacks user attaches:
        self.on_suppress: Optional[Callable[[ASRResult], None]] = None
        self.on_forward: Optional[Callable[[ASRResult], None]] = None
        self.on_register: Optional[Callable[[ASRResult], None]] = None

    @staticmethod
    def _normalize(token: str) -> str:
        s = token.lower().strip()
        s = re.sub(r"[^\w\s\u00C0-\u024F'-]", "", s)  # basic punctuation removal
        return s

    def _tokenize(self, text: str):
        text = text or ""
        text = text.strip()
        if not text:
            return []
        text = self._normalize(text)
        return [t for t in re.split(r"\s+", text) if t]

    async def update_ignored_words(self, words: List[str]):
        async with self._lock:
            self._ignored_set = set(self._normalize(w) for w in words)
            self._log("info", {"msg":"ignored words updated", "words": list(self._ignored_set)})

    def set_agent_speaking(self, is_speaking: bool):
        """Call when TTS starts/stops."""
        self._agent_speaking = bool(is_speaking)
        self._log("debug", {"agent_speaking": self._agent_speaking})
        if not self._agent_speaking:
            if self._idle_task and not self._idle_task.done():
                self._idle_task.cancel()
            self._buffer = ""

    async def handle_asr(self, asr: ASRResult):
        text = (asr.text or "").strip()
        if not text:
            return

        tokens = self._tokenize(text)
        conf = asr.confidence if asr.confidence is not None else 1.0

        # quick explicit-stop detection
        stop_phrases = {"stop","wait","no","hold","pause","stopnow","stopthat","waitone","one","one second"}
        if any(tok in stop_phrases for tok in tokens):
            self._log("info", {"reason":"explicit_stop", "text": text})
            if self.on_forward:
                self.on_forward(asr)
            return

        async with self._lock:
            if self._agent_speaking:
                # accumulate
                self._buffer = f"{self._buffer} {text}".strip() if self._buffer else text

                # reset idle flush timer
                if self._idle_task and not self._idle_task.done():
                    self._idle_task.cancel()
                self._idle_task = asyncio.create_task(self._idle_flush_task(asr))

                non_ignored = any(tok not in self._ignored_set for tok in tokens)
                if conf < self._confidence_threshold and not non_ignored:
                    self._log("debug", {"action":"suppress_low_confidence", "text":text, "conf":conf})
                    if self.on_suppress:
                        self.on_suppress(asr)
                    return
                if not non_ignored:
                    self._log("debug", {"action":"suppress_filler_partial", "text":text})
                    if self.on_suppress:
                        self.on_suppress(asr)
                    return
                # contains non-filler -> forward
                self._log("info", {"action":"forward_from_partial", "text": text})
                if self.on_forward:
                    self.on_forward(asr)
                self._buffer = ""
                if self._idle_task and not self._idle_task.done():
                    self._idle_task.cancel()
                return
            else:
                # agent quiet -> register everything
                self._log("debug", {"action":"register_agent_quiet", "text":text})
                if self.on_register:
                    self.on_register(asr)
                return

    async def _idle_flush_task(self, asr_snapshot: ASRResult):
        try:
            await asyncio.sleep(self._partial_idle_ms / 1000.0)
            async with self._lock:
                buf = self._buffer.strip()
                if not buf:
                    return
                tokens = self._tokenize(buf)
                conf = asr_snapshot.confidence if asr_snapshot.confidence is not None else 1.0
                non_ignored = any(tok not in self._ignored_set for tok in tokens)

                if self._agent_speaking:
                    if (not non_ignored) or (conf < self._confidence_threshold):
                        self._log("debug", {"action":"suppress_buffered", "buffer": buf, "conf": conf})
                        if self.on_suppress:
                            self.on_suppress(ASRResult(text=buf, confidence=conf, is_final=True, language=asr_snapshot.language, timestamp=time.time()))
                    else:
                        self._log("info", {"action":"forward_buffered", "buffer": buf})
                        if self.on_forward:
                            self.on_forward(ASRResult(text=buf, confidence=conf, is_final=True, language=asr_snapshot.language, timestamp=time.time()))
                else:
                    self._log("debug", {"action":"register_buffered_agent_quiet","buffer":buf})
                    if self.on_register:
                        self.on_register(ASRResult(text=buf, confidence=conf, is_final=True, language=asr_snapshot.language, timestamp=time.time()))
                self._buffer = ""
        except asyncio.CancelledError:
            return
