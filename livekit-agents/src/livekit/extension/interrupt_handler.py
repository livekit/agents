from dataclasses import dataclass
from typing import Optional, Tuple

from .config import InterruptConfig
from .filters import tokenize, contains_any_phrase, normalize
import logging

log = logging.getLogger("interrupt_handler")
log.setLevel(logging.INFO)

@dataclass
class ASRChunk:
    text: str                 # raw transcript text
    avg_confidence: float     # average token conf (or 0..1 chunk conf)
    is_final: bool            # final segment flag if available

class InterruptDecision:
    IGNORE = "IGNORE"         # ignore this chunk (agent continues)
    ACCEPT = "ACCEPT"         # accept as user speech (may stop speaking)
    FORCE_STOP = "FORCE_STOP" # immediate stop (priority)

class InterruptHandler:
    def __init__(self, cfg: Optional[InterruptConfig] = None):
        self.cfg = cfg or InterruptConfig()
        self.agent_speaking = False

    # Integrations should call these when TTS starts/ends
    def on_tts_start(self) -> None:
        self.agent_speaking = True
        if self.cfg.debug_logging:
            log.info("[state] agent_speaking=True")

    def on_tts_end(self) -> None:
        self.agent_speaking = False
        if self.cfg.debug_logging:
            log.info("[state] agent_speaking=False")

    def classify(self, chunk: ASRChunk) -> Tuple[str, dict]:
        """
        Returns: (decision, meta)
        decision in {IGNORE, ACCEPT, FORCE_STOP}
        meta includes reasons for logs
        """
        text = chunk.text or ""
        tokens = tokenize(text)
        meta = {
            "text": text,
            "tokens": tokens,
            "avg_conf": chunk.avg_confidence,
            "agent_speaking": self.agent_speaking
        }

        # Priority words ALWAYS win
        if contains_any_phrase(text, self.cfg.priority_words):
            meta["reason"] = "priority_word"
            return InterruptDecision.FORCE_STOP, meta

        # If agent not speaking -> treat any detected speech as valid
        if not self.agent_speaking:
            meta["reason"] = "agent_quiet_accept"
            return InterruptDecision.ACCEPT, meta

        # Agent is speaking: apply confidence + filler ratio gates
        if chunk.avg_confidence < self.cfg.asr_conf_min:
            meta["reason"] = "low_conf_suppress_while_speaking"
            return InterruptDecision.IGNORE, meta

        if not tokens:
            meta["reason"] = "no_tokens"
            return InterruptDecision.IGNORE, meta

        # Compute filler ratio
        fillers = sum(1 for t in tokens if t in self.cfg.ignored_words)
        ratio = fillers / max(1, len(tokens))
        meta["filler_ratio"] = ratio
        meta["fillers"] = fillers
        meta["ntokens"] = len(tokens)

        if ratio >= self.cfg.filler_ratio_min:
            meta["reason"] = "filler_dominated_while_speaking"
            return InterruptDecision.IGNORE, meta

        # Otherwise, accept as real interruption
        meta["reason"] = "real_interruption_while_speaking"
        return InterruptDecision.ACCEPT, meta
