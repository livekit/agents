import logging
from interrupt_handler.utils import normalize_text
from interrupt_handler.constants import DEFAULT_IGNORED, DEFAULT_COMMANDS

logger = logging.getLogger("interrupt_handler")


class InterruptFilteringMiddleware:
    """
    Decides whether the agent should be interrupted based on:
    - filler-only speech
    - command words
    - ASR confidence
    """

    def __init__(
        self,
        ignored_words=None,
        command_words=None,
        conf_threshold: float = 0.6,
    ):
        self.ignored = set(ignored_words or DEFAULT_IGNORED)
        self.commands = set(command_words or DEFAULT_COMMANDS)
        self.conf_threshold = conf_threshold

    async def should_interrupt(
        self,
        text: str,
        confidence: float,
        agent_is_speaking: bool,
    ) -> bool:

        # When the agent is NOT speaking → treat user speech as valid turn
        if not agent_is_speaking:
            logger.debug("[FILTER] Agent silent → accept speech.")
            return True

        # Normalize into clean token list
        tokens = normalize_text(text)

        # Background noise filtering
        if confidence < self.conf_threshold:
            logger.info(
                f"[FILTER] ASR low confidence ({confidence:.2f}) → IGNORE"
            )
            return False

        # Pure filler
        if tokens and all(t in self.ignored for t in tokens):
            logger.info(f"[FILTER] FILLER detected: {tokens} → IGNORE")
            return False

        # Commands override everything
        if any(t in self.commands for t in tokens):
            logger.info(
                f"[FILTER] COMMAND detected {tokens} → INTERRUPT"
            )
            return True

        # Normal meaningful speech → interrupt
        logger.info(f"[FILTER] Real speech {tokens} → INTERRUPT")
        return True
