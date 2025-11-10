import re
from dataclasses import dataclass
from typing import List, Optional
from .config import IHConfig

_WORD = re.compile(r"\w+(?:['-]\w+)?", re.UNICODE)

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s'-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@dataclass
class Judgement:
    label: str              # "LOW_CONF", "FILLER_ONLY", "HARD_INTENT", "PENDING", "CONTENT"
    reason: str
    content_tokens: int = 0

class UtteranceClassifier:
    def __init__(self, cfg: IHConfig):
        self.cfg = cfg

    def decide(self, text: str, confidence: Optional[float], duration_ms: Optional[int]) -> Judgement:
        t = _normalize(text)
        if not t:
            return Judgement("LOW_CONF", "empty")

        if confidence is not None and confidence < self.cfg.min_confidence:
            return Judgement("LOW_CONF", f"conf<{self.cfg.min_confidence}")

        # multi-language command detection
        for phrase in self.cfg.all_commands():
            if phrase in t:
                return Judgement("HARD_INTENT", f"hard:{phrase}")

        tokens = _WORD.findall(t)
        all_fillers = self.cfg.all_fillers()
        content = [w for w in tokens if w not in all_fillers]

        if not content:
            return Judgement("FILLER_ONLY", "all_fillers", 0)

        if len(content) < self.cfg.min_content_tokens and (duration_ms or 0) < self.cfg.min_duration_ms:
            return Judgement("PENDING", "short_partial", len(content))

        return Judgement("CONTENT", "meets_threshold", len(content))
