import asyncio
import re
from datetime import datetime
from .config import get_config

cfg = get_config()

class InterruptHandler:
    def __init__(self, is_agent_speaking, stop_agent, logger=None):
        self.is_agent_speaking = is_agent_speaking
        self.stop_agent = stop_agent
        self.ignored_words = set(cfg["ignored_words"])
        self.command_words = set(cfg["command_words"])
        self.conf_threshold = cfg["confidence_threshold"]
        self.logger = logger or print
        self._lock = asyncio.Lock()

    async def handle_asr(self, text: str, confidence: float):
        async with self._lock:
            agent_speaking = self.is_agent_speaking()
            normalized = self._normalize(text)
            words = normalized.split()
            ts = datetime.utcnow().isoformat()

            if confidence < self.conf_threshold and agent_speaking:
                return self._log("IGNORED_LOW_CONF", text, confidence, agent_speaking, ts)

            if any(word in self.command_words for word in words):
                self._log("VALID_INTERRUPT", text, confidence, agent_speaking, ts)
                self.stop_agent()
                return

            if agent_speaking:
                if all(word in self.ignored_words for word in words):
                    return self._log("IGNORED_FILLER", text, confidence, agent_speaking, ts)

                self._log("VALID_MIXED", text, confidence, agent_speaking, ts)
                self.stop_agent()
                return

            self._log("USER_SPEECH", text, confidence, agent_speaking, ts)

    def _normalize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _log(self, tag, text, conf, speaking, ts):
        self.logger(f"{ts} | {tag} | speaking={speaking} | conf={conf} | text='{text}'")
