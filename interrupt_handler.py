import os, re, asyncio, logging
from typing import List, Callable, Optional, Set

logger = logging.getLogger("interrupt_handler")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

DEFAULT_IGNORED = os.getenv("IGNORED_WORDS", "uh,umm,hmm,haan").split(",")
DEFAULT_COMMANDS = os.getenv("COMMAND_WORDS", "stop,wait,pause,hold").split(",")
DEFAULT_CONFIDENCE = float(os.getenv("MIN_ASR_CONFIDENCE", "0.6"))

def normalize(text: str):
    text = re.sub(r"[^\w\s]", "", text.lower().strip())
    return re.sub(r"\s+", " ", text)

class InterruptHandler:
    def __init__(self, on_interrupt: Callable[[], None],
                 ignored_words: Optional[List[str]] = None,
                 command_words: Optional[List[str]] = None,
                 min_conf: float = DEFAULT_CONFIDENCE):
        self.on_interrupt = on_interrupt
        self.min_conf = min_conf
        self._lock = asyncio.Lock()
        self._ignored: Set[str] = set(map(normalize, ignored_words or DEFAULT_IGNORED))
        self._commands: Set[str] = set(map(normalize, command_words or DEFAULT_COMMANDS))

    async def decide(self, transcript: str, conf: float, agent_speaking: bool):
        norm = normalize(transcript)
        tokens = norm.split()

        async with self._lock:
            ignored, commands = self._ignored, self._commands

        if not agent_speaking:
            return True, "agent_silent"

        if any(tok in commands for tok in tokens):
            logger.info("Real interrupt: '%s'", transcript)
            return True, "command_found"

        if all(tok in ignored for tok in tokens):
            if conf >= self.min_conf:
                logger.info("Ignored filler: '%s'", transcript)
                return False, "filler_only"
            else:
                logger.debug("Low conf, ignoring: '%s'", transcript)
                return False, "low_conf"

        logger.info("Mixed/real speech: '%s'", transcript)
        return True, "non_filler"

    async def on_transcript(self, event: dict, agent_state_getter: Callable[[], bool]):
        transcript = event.get("transcript", "")
        conf = event.get("confidence", 1.0)
        agent_speaking = agent_state_getter()
        should_interrupt, reason = await self.decide(transcript, conf, agent_speaking)

        if should_interrupt and agent_speaking:
            result = self.on_interrupt()
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        logger.debug("Decision: %s (%s)", should_interrupt, reason)
