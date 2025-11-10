"""
livekit_interrupt_filter.py
Async, thread-safe filter to distinguish filler-only speech vs real interruptions.

API:
  InterruptFilter(...) - create filter
  await filter.handle_transcription_event(transcript, confidence, is_final, agent_speaking)
      -> dict: { action: 'ignore'|'stop'|'register', reason: str, cleaned_text: str }

  filter.update_ignored_words(new_list)  # dynamic update (thread-safe)
  filter.reload_from_env()               # reloads IGNORED_WORDS from env
"""

import asyncio
import os
import re
import signal
import logging
from typing import List, Optional


DEFAULT_COMMAND_WORDS = [
    "stop", "wait", "hold", "pause", "no", "don't", "dont", "not", "hang on", "one second", "just a second",
    "stop that", "that's enough", "enough", "actually", "correct"
]

_logger = logging.getLogger("InterruptFilter")
if not _logger.handlers:
  
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    
    text = re.sub(r"[^\w\s']+", " ", text, flags=re.UNICODE)
   
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.findall(r"[\w']+", text, flags=re.UNICODE) if t]


class InterruptFilter:
    def __init__(
        self,
        ignored_words: Optional[List[str]] = None,
        filler_confidence_threshold: float = 0.8,
        ignore_when_confidence_less_than: float = 0.5,
        extra_command_words: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        ignored_words: list of filler tokens (e.g. ['uh','umm','hmm','haan'])
        filler_confidence_threshold: if ASR confidence >= this and transcription only contains filler words -> it's filler
        ignore_when_confidence_less_than: if confidence < this and agent is speaking -> ignore (optional safety)
        extra_command_words: extra words you wish to treat as stop commands
        """
        self.logger = logger or _logger
        self._lock = asyncio.Lock()
        
        self.ignored_words = set(self._norm_list(ignored_words or ["uh", "umm", "hmm", "haan"]))
        self.filler_confidence_threshold = float(filler_confidence_threshold)
        self.ignore_when_confidence_less_than = float(ignore_when_confidence_less_than)
        self.command_words = set(self._norm_list(DEFAULT_COMMAND_WORDS + (extra_command_words or [])))
        self._stats = {"ignored": 0, "stops": 0, "registered": 0}
        
        try:
            signal.signal(signal.SIGHUP, lambda *_: asyncio.get_event_loop().create_task(self.reload_from_env()))
        except Exception:
            
            pass

    def _norm_list(self, items: List[str]) -> List[str]:
        return [ _normalize_text(x) for x in items if x and x.strip() ]

    async def update_ignored_words(self, new_list: List[str]):
        async with self._lock:
            self.ignored_words = set(self._norm_list(new_list))
            self.logger.info("Ignored words updated: %s", sorted(self.ignored_words))

    async def reload_from_env(self):
        """Read IGNORED_WORDS env var (comma separated) and update list."""
        raw = os.getenv("IGNORED_WORDS", "")
        if not raw:
            self.logger.info("IGNORED_WORDS empty or not set; leaving unchanged.")
            return
        new = [x.strip() for x in raw.split(",") if x.strip()]
        await self.update_ignored_words(new)

    async def handle_transcription_event(
        self,
        transcript: str,
        confidence: float,
        is_final: bool,
        agent_speaking: bool,
    ) -> dict:
        """
        Main decision function.

        Returns:
          { action: 'ignore' | 'stop' | 'register',
            reason: explanation string,
            cleaned_text: normalized transcript (may be empty) }
        """
        norm = _normalize_text(transcript)
        tokens = _tokenize(norm)
        conf = float(confidence) if confidence is not None else 1.0

        
        if not tokens:
            reason = "empty_transcript"
           
            action = "ignore" if agent_speaking else "register"
            self._record(action)
            self.logger.debug("Decision %s for empty transcript. agent_speaking=%s", action, agent_speaking)
            return {"action": action, "reason": reason, "cleaned_text": norm, "confidence": conf}

        async with self._lock:
            
            contains_command = any(tok in self.command_words for tok in tokens)
           
            non_filler_tokens = [tok for tok in tokens if tok not in self.ignored_words]
            only_fillers = len(non_filler_tokens) == 0

            
            if agent_speaking and conf < self.ignore_when_confidence_less_than and not contains_command:
                reason = f"low_confidence_ignored (conf={conf})"
                action = "ignore"
                self._record(action)
                self.logger.info("Ignored low-confidence audio while agent speaking: '%s' conf=%.2f", transcript, conf)
                return {"action": action, "reason": reason, "cleaned_text": norm, "confidence": conf}

            
            if contains_command:
                reason = "contains_command_word"
                action = "stop"
                self._record(action)
                self.logger.info("Stop: detected command word in transcript='%s' conf=%.2f", transcript, conf)
                return {"action": action, "reason": reason, "cleaned_text": norm, "confidence": conf}

           
            if only_fillers:
              
                if agent_speaking:
                    if conf >= self.filler_confidence_threshold:
                        reason = f"filler_ignored (conf={conf})"
                        action = "ignore"
                        self._record(action)
                        self.logger.info("Ignored filler while agent speaking: '%s' conf=%.2f", transcript, conf)
                        return {"action": action, "reason": reason, "cleaned_text": norm, "confidence": conf}
                    else:
                       
                        reason = f"filler_low_confidence_ignored (conf={conf})"
                        action = "ignore"
                        self._record(action)
                        self.logger.info("Ignored low-conf filler while agent speaking: '%s' conf=%.2f", transcript, conf)
                        return {"action": action, "reason": reason, "cleaned_text": norm, "confidence": conf}
                else:
                   
                    reason = "filler_registered_agent_quiet"
                    action = "register"
                    self._record(action)
                    self.logger.info("Registered filler while agent quiet: '%s' conf=%.2f", transcript, conf)
                    return {"action": action, "reason": reason, "cleaned_text": norm, "confidence": conf}

            
            if agent_speaking:
                reason = "non_filler_while_agent_speaking"
                action = "stop"
                self._record(action)
                self.logger.info("Stop: Non-filler tokens while agent speaking: '%s' conf=%.2f", transcript, conf)
                return {"action": action, "reason": reason, "cleaned_text": norm, "confidence": conf}
            else:
                reason = "user_speech_registered_agent_quiet"
                action = "register"
                self._record(action)
                self.logger.info("Register: User speech while agent quiet: '%s' conf=%.2f", transcript, conf)
                return {"action": action, "reason": reason, "cleaned_text": norm, "confidence": conf}

    def _record(self, action: str):
        if action not in self._stats:
            self._stats[action] = 0
        self._stats[action] += 1

    def stats(self) -> dict:
        return dict(self._stats)

    
    def decide_sync(self, transcript: str, confidence: float, is_final: bool, agent_speaking: bool) -> dict:
        """Synchronous wrapper (for quick unit testing or blocking code)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.handle_transcription_event(transcript, confidence, is_final, agent_speaking))
