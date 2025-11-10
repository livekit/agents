import asyncio
import logging
import os
from typing import Iterable, Dict, Any

log = logging.getLogger("InterruptFilter")
log.setLevel(logging.INFO)

DEFAULT_IGNORED = [w.strip().lower() for w in os.environ.get("IGNORED_WORDS", "uh,umm,hmm,haan").split(",") if w.strip()]
DEFAULT_CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.6))

class InterruptFilter:
    def __init__(self, ignored_words=None, conf_threshold=0.6):
        self._ignored_words = set(ignored_words or [])
        self._conf_threshold = conf_threshold
        self._lock = asyncio.Lock()

    async def update_ignored_words(self, new_list):
        async with self._lock:
            self._ignored_words = set(new_list)

    def __init__(self, ignored_words: Iterable[str] = None, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
        self.ignored_words = set(w.lower().strip() for w in (ignored_words or DEFAULT_IGNORED))
        self.conf_threshold = conf_threshold
        self._lock = asyncio.Lock()
        self.stats = {"ignored": 0, "accepted": 0}

    async def set_ignored_words(self, new_words: list[str]):
        async with self._lock:
            self._ignored_words = set(new_words)
            logger.info("InterruptFilter: ignored words updated -> %s", new_words)

    async def on_asr_event(self, text: str, *, confidence: float = 1.0, agent_speaking: bool = False) -> Dict[str, Any]:
        """
        Evaluate ASR event text and return {"should_stop_agent": bool, "reason": str}
        """
        async with self._lock:
            text = (text or "").strip()
            if not text:
                return {"should_stop_agent": False, "reason": "empty"}

            words = [w.lower().strip() for w in text.split() if w.strip()]
            non_filler_words = [w for w in words if w not in self.ignored_words]

            # Agent currently speaking
            if agent_speaking:
                if confidence < self.conf_threshold and not non_filler_words:
                    self.stats["ignored"] += 1
                    log.debug("Ignored low-confidence filler while agent speaking: %s (conf=%s)", text, confidence)
                    return {"should_stop_agent": False, "reason": "low_confidence_filler"}

                if not non_filler_words:
                    self.stats["ignored"] += 1
                    log.info("Ignored filler while agent speaking: '%s'", text)
                    return {"should_stop_agent": False, "reason": "filler_only"}

                # Mixed or real interruption
                self.stats["accepted"] += 1
                log.info("Accepted interruption while agent speaking: '%s'", text)
                return {"should_stop_agent": True, "reason": "non_filler_present"}

            # Agent quiet -> register as user speech
            self.stats["accepted"] += 1
            log.info("Agent quiet — treat as user speech: '%s'", text)
            return {"should_stop_agent": False, "reason": "agent_quiet"}

    def update_ignored_words(self, new_words: Iterable[str]) -> None:
        """Dynamically update the ignored words list."""
        self.ignored_words = set(w.lower().strip() for w in new_words if w.strip())
        log.info("Updated ignored words: %s", self.ignored_words)


def livekit_asr_adapter(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize typical LiveKit/ASR event payloads into:
      {"text": str, "confidence": float, "agent_speaking": bool}

    Accepts multiple shapes: {'text':..., 'confidence':...}, or nested,
    or legacy fields. Returns defaults when missing.
    """
    if not isinstance(data, dict):
        return {"text": str(data or ""), "confidence": 1.0, "agent_speaking": False}

    # Try common keys
    text = data.get("text") or data.get("transcript") or data.get("alternatives_text") or ""
    # Some ASR providers use a list of alternatives
    if not text:
        alt = data.get("alternatives") or data.get("results") or None
        if isinstance(alt, list) and len(alt) > 0:
            first = alt[0]
            # try to extract text from alt structure
            if isinstance(first, dict):
                text = first.get("transcript") or first.get("text") or ""
            elif isinstance(first, str):
                text = first

    # Confidence heuristics
    conf = data.get("confidence")
    if conf is None:
        # sometimes inside alternatives
        if isinstance(alt, list) and len(alt) > 0 and isinstance(alt[0], dict):
            conf = alt[0].get("confidence", 1.0)
    try:
        conf = float(conf) if conf is not None else 1.0
    except Exception:
        conf = 1.0

    agent_speaking = bool(data.get("agent_speaking")) or bool(data.get("agentIsSpeaking")) or False

    return {"text": str(text or ""), "confidence": conf, "agent_speaking": agent_speaking}
