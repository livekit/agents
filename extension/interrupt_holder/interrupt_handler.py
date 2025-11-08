import os
import asyncio
import logging
from typing import List, Callable, Optional

logger = logging.getLogger("interrupt_handler")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)

def _default_ignored_words() -> List[str]:
    raw = os.getenv("IGNORED_WORDS", "uh,umm,hmm,haan")
    return [w.strip().lower() for w in raw.split(",") if w.strip()]

class InterruptHandler:
    def __init__(
        self,
        is_agent_speaking_cb: Callable[[], bool],
        stop_agent_cb: Callable[[], None],
        accepted_callback: Optional[Callable[[str], None]] = None,
        ignored_words: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
    ):
        self._is_agent_speaking_cb = is_agent_speaking_cb
        self._stop_agent_cb = stop_agent_cb
        self._accepted_cb = accepted_callback
        self._ignored_words = ignored_words or _default_ignored_words()
        self._conf_threshold = confidence_threshold
        self._lock = asyncio.Lock()

    async def on_transcription(self, text: str, confidence: float = 1.0):
        text_norm = (text or "").strip().lower()
        if not text_norm:
            return

        async with self._lock:
            agent_speaking = self._is_agent_speaking_cb()
            if agent_speaking and confidence < self._conf_threshold:
                logger.info(f"Ignored low-confidence: '{text_norm}'")
                return

            tokens = [tok for tok in text_norm.split() if tok]
            all_ignored = all(tok in self._ignored_words for tok in tokens)

            if agent_speaking and all_ignored:
                logger.info(f"Ignored filler: '{text_norm}'")
                return

            logger.info(f"Accepted speech: '{text_norm}'")
            if self._accepted_cb:
                self._accepted_cb(text_norm)
            self._stop_agent_cb()
