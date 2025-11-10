import os
import asyncio
import re
import json
from typing import List, Optional, Callable, Dict, Any

import logging
logger = logging.getLogger("interrupt_handler")
ignored_logger = logging.getLogger("interrupt_handler.ignored")
valid_logger = logging.getLogger("interrupt_handler.valid")
if not logging.getLogger().handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    h.setFormatter(fmt)
    logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.INFO)

def _load_ignored_words_from_env() -> List[str]:
    raw = os.getenv("IGNORED_WORDS", "uh,umm,hmm,haan")
    return [w.strip().lower() for w in raw.split(",") if w.strip()]

class InterruptFilter:
    def __init__(
        self,
        ignored_words: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        on_interrupt: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._ignored_words = set(ignored_words or _load_ignored_words_from_env())
        self._confidence_threshold = float(confidence_threshold)
        self._lock = asyncio.Lock()
        self.on_interrupt = on_interrupt

    async def on_asr_event(self, text: str, confidence: float, agent_speaking: bool) -> Dict[str, Any]:
        if text is None:
            text = ""
        normalized = text.strip().lower()
        tokens = [t for t in re.split(r"\W+", normalized) if t]

        async with self._lock:
            if not tokens:
                reason = "empty_or_unintelligible"
                ignored_logger.info(json.dumps({"text": text, "confidence": confidence, "reason": reason}))
                return {"should_stop_agent": False, "handled_as_ignored": True, "reason": reason}

            non_ignored_tokens = [t for t in tokens if t not in self._ignored_words]

            if agent_speaking:
                if len(non_ignored_tokens) == 0:
                    reason = "all_filler_tokens"
                    ignored_logger.info(json.dumps({"text": text, "confidence": confidence, "reason": reason}))
                    return {"should_stop_agent": False, "handled_as_ignored": True, "reason": reason}

                if confidence >= self._confidence_threshold:
                    reason = "contains_non_filler_with_confidence"
                    valid_logger.info(json.dumps({"text": text, "confidence": confidence, "reason": reason}))
                    if callable(self.on_interrupt):
                        try:
                            self.on_interrupt({"text": text, "confidence": confidence, "reason": reason})
                        except Exception as e:
                            logger.exception("on_interrupt callback failed: %s", e)
                    return {"should_stop_agent": True, "handled_as_ignored": False, "reason": reason}
                else:
                    reason = "non_filler_low_confidence_ignored"
                    ignored_logger.info(json.dumps({"text": text, "confidence": confidence, "reason": reason}))
                    return {"should_stop_agent": False, "handled_as_ignored": True, "reason": reason}
            else:
                reason = "agent_quiet_registered"
                valid_logger.info(json.dumps({"text": text, "confidence": confidence, "reason": reason}))
                if callable(self.on_interrupt):
                    try:
                        self.on_interrupt({"text": text, "confidence": confidence, "reason": reason})
                    except Exception as e:
                        logger.exception("on_interrupt callback failed: %s", e)
                return {"should_stop_agent": False, "handled_as_ignored": False, "reason": reason}

    async def update_ignored_words(self, new_words: List[str]):
        async with self._lock:
            self._ignored_words = set(w.strip().lower() for w in new_words if w.strip())
            logger.info("Ignored words updated: %s", list(self._ignored_words))

    def set_confidence_threshold(self, threshold: float):
        self._confidence_threshold = float(threshold)
        logger.info("Confidence threshold set to %s", self._confidence_threshold)

def livekit_asr_adapter(event: Dict[str, Any]) -> Dict[str, Any]:
    text = event.get("text") or ""
    conf = event.get("confidence")
    if conf is None:
        conf = 1.0
    agent_speaking = bool(event.get("agent_speaking", False))
    return {"text": text, "confidence": float(conf), "agent_speaking": agent_speaking}
