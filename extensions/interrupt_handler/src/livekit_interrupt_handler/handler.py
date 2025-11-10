import asyncio
import logging
from typing import Dict, Any, List
from .config import Config
from .utils import tokenize, is_filler_only, contains_interrupt_command

logger = logging.getLogger("interrupt_handler")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(handler)

class InterruptHandler:
    """
    Extension-layer handler that sits on top of LiveKit VAD/transcription events.
    It does NOT modify LiveKit internals; it exposes callbacks that the agent code calls.
    """

    def __init__(self,
                 ignored_words: List[str] = None,
                 commands: List[str] = None,
                 confidence_threshold: float = None,
                 strict_filler_match: bool = True):
        self.ignored_words = ignored_words or Config.ignored_words
        self.commands = commands or Config.always_interrupt_commands
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else Config.confidence_threshold
        self.strict_filler_match = strict_filler_match
        # state
        self.agent_is_speaking = False
        # protect state in async environment
        self._lock = asyncio.Lock()

    async def set_agent_speaking(self, speaking: bool):
        async with self._lock:
            self.agent_is_speaking = speaking
            logger.debug("Agent speaking state -> %s", speaking)

    async def update_ignored_words(self, new_list):
        async with self._lock:
            self.ignored_words = [x.lower() for x in new_list]
            logger.info("Updated ignored words: %s", self.ignored_words)

    async def on_transcription(self, asr_result: Dict[str, Any]):
        """
        Called by the agent when an ASR/transcription result arrives.

        Expected asr_result fields:
          - text: str
          - confidence: float (0..1) optional
          - metadata: any optional extras
        """
        text = (asr_result.get("text") or "").strip()
        confidence = asr_result.get("confidence", 1.0)
        tokens = tokenize(text)

        async with self._lock:
            agent_speaking = self.agent_is_speaking
            ignored_words = self.ignored_words
            commands = self.commands
            thresh = self.confidence_threshold
            strict = self.strict_filler_match

        # Quick path: empty text
        if not tokens:
            logger.debug("Empty/whitespace ASR; ignoring.")
            return {"action": "ignore", "reason": "empty"}

        # If contains a command token, always treat as interruption
        if contains_interrupt_command(tokens, commands):
            logger.info("VALID INTERRUPTION (command found): %s (tokens=%s)", text, tokens)
            return {"action": "interrupt", "reason": "command", "text": text}

        # If agent NOT speaking, anything is valid user speech
        if not agent_speaking:
            logger.info("Agent silent -> registering user speech: %s", text)
            return {"action": "register", "reason": "agent_silent", "text": text}

        # Agent is speaking
        # Low confidence => treat as filler/ignore (prevent false positives) unless it contains command (handled above)
        if confidence < thresh:
            logger.debug("ASR confidence low (%.2f < %.2f) while agent speaking; ignoring: %s", confidence, thresh, text)
            return {"action": "ignore", "reason": "low_confidence", "confidence": confidence, "text": text}

        # If tokens are filler-only -> ignore
        if is_filler_only(tokens, ignored_words, strict=strict):
            logger.info("IGNORED filler while agent speaking: %s (tokens=%s)", text, tokens)
            return {"action": "ignore", "reason": "filler_only", "text": text}

        # Mixed or non-filler tokens -> treat as interruption
        logger.info("VALID INTERRUPTION while agent speaking: %s (tokens=%s)", text, tokens)
        return {"action": "interrupt", "reason": "non_filler", "text": text}
