# livekit_interrupt_handler.py
"""
Interrupt handler extension for LiveKit Agents.

See README.md for integration & testing instructions.

Author: generated for branch feature/livekit-interrupt-handler-<yourname>
"""

import os
import asyncio
import logging
from typing import List, Set, Callable, Optional, Dict

logger = logging.getLogger("livekit_interrupt_handler")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

DEFAULT_IGNORED = ["uh", "umm", "hmm", "haan"]
DEFAULT_COMMAND_WORDS = ["stop", "wait", "hold", "pause", "no", "not", "cancel"]


class InterruptHandler:
    """
    Interrupt handler extension class.

    Integrate by registering the handler.on_transcription coroutine to your agent's
    transcription events and calling on_agent_speaking / on_agent_speaking_stop when
    your TTS starts/stops.

    Example integration (adapt to your agent API):
        handler = InterruptHandler(agent)
        agent.on('start_speaking', lambda *_: asyncio.create_task(handler.on_agent_speaking()))
        agent.on('stop_speaking',  lambda *_: asyncio.create_task(handler.on_agent_speaking_stop()))
        agent.on('transcription', lambda event: asyncio.create_task(handler.on_transcription(event)))
    """

    def __init__(
        self,
        agent=None,
        ignored_words: Optional[List[str]] = None,
        command_words: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        on_stop_callback: Optional[Callable[[Dict], None]] = None,
    ):
        env_list = os.getenv("IGNORED_WORDS")
        if ignored_words is None:
            ignored_words = [w.strip() for w in env_list.split(",")] if env_list else DEFAULT_IGNORED

        self._ignored_words: Set[str] = set(w.lower() for w in ignored_words if w)
        self._command_words: Set[str] = set(w.lower() for w in (command_words or DEFAULT_COMMAND_WORDS))

        self.confidence_threshold = float(confidence_threshold)
        self.on_stop_callback = on_stop_callback
        self.agent = agent

        self._agent_speaking = False
        self._lock = asyncio.Lock()

        self._ignored_count = 0
        self._valid_count = 0

        logger.info("InterruptHandler initialized; ignored_words=%s", sorted(self._ignored_words))

    async def on_agent_speaking(self, *args, **kwargs):
        async with self._lock:
            self._agent_speaking = True
            logger.debug("Agent speaking state -> True")

    async def on_agent_speaking_stop(self, *args, **kwargs):
        async with self._lock:
            self._agent_speaking = False
            logger.debug("Agent speaking state -> False")

    def set_ignored_words(self, words: List[str]):
        """
        Synchronous setter (safe to call from sync code).
        For safe async updates, call via loop.run_in_executor or similar.
        """
        self._ignored_words = set(w.lower() for w in words if w)
        logger.info("Ignored words updated -> %s", sorted(self._ignored_words))

    def get_ignored_words(self) -> List[str]:
        return sorted(list(self._ignored_words))

    def _parse_event(self, event: dict) -> dict:
        """
        Normalize event to {text, tokens, confidence}.
        Expected event structure may vary depending on your ASR; adapt as needed.
        """
        text = event.get("text", "") if isinstance(event, dict) else str(event)
        tokens = event.get("tokens") if isinstance(event, dict) else None
        if not tokens:
            tokens = [t for t in (text or "").split() if t]
        # lowercase
        tokens = [t.lower() for t in tokens]
        confidence = event.get("confidence") if isinstance(event, dict) else None
        try:
            confidence = float(confidence) if confidence is not None else 1.0
        except Exception:
            confidence = 1.0
        return {"text": text, "tokens": tokens, "confidence": confidence}

    async def _should_ignore(self, parsed: dict) -> bool:
        """
        Decision logic.
        - If agent not speaking -> never ignore.
        - If agent speaking:
            * command words present -> do not ignore
            * confidence < threshold -> ignore (unless command present)
            * tokens all in ignored_words -> ignore
            * else -> do not ignore
        """
        async with self._lock:
            agent_speaking = self._agent_speaking

        if not agent_speaking:
            logger.debug("Agent not speaking: will not ignore")
            return False

        tokens = parsed.get("tokens", [])
        confidence = parsed.get("confidence", 1.0)

        # Command words override
        for tok in tokens:
            if tok in self._command_words:
                logger.debug("Found command word '%s' -> NOT ignoring", tok)
                return False

        # Confidence low => likely background murmur: ignore
        if confidence < self.confidence_threshold:
            logger.debug("Confidence %.2f < threshold %.2f -> ignoring", confidence, self.confidence_threshold)
            return True

        if not tokens:
            logger.debug("No tokens -> ignoring")
            return True

        # strip simple punctuation and test if all tokens are in ignored list
        normalized = [t.strip(".,!?;:\\\"'()[]{}") for t in tokens]
        if all((t in self._ignored_words) for t in normalized):
            logger.debug("All tokens are filler -> ignoring")
            return True

        # Mixed / content tokens
        logger.debug("Tokens contain content -> NOT ignoring")
        return False

    async def on_transcription(self, event: dict):
        """
        Main handler for transcription events (async).
        If interruption is valid, calls on_stop_callback (if provided) or attempts agent.stop_speaking.
        """
        parsed = self._parse_event(event)
        try:
            ignore = await self._should_ignore(parsed)
        except Exception:
            logger.exception("Error computing ignore decision; treating as valid interruption")
            ignore = False

        if ignore:
            self._ignored_count += 1
            logger.info("Ignored filler while agent speaking: '%s' (confidence=%.2f)", parsed["text"], parsed["confidence"])
            return

        # valid interruption
        self._valid_count += 1
        logger.info("Valid interruption detected: '%s' (confidence=%.2f)", parsed["text"], parsed["confidence"])

        # Attempt to stop/pause agent
        if self.on_stop_callback:
            try:
                if asyncio.iscoroutinefunction(self.on_stop_callback):
                    await self.on_stop_callback(parsed)
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.on_stop_callback, parsed)
                return
            except Exception:
                logger.exception("on_stop_callback raised, falling back to agent API")

        # Fallback: try common agent API names (non-exhaustive)
        for cand in ("stop_speaking", "pause_tts", "interrupt", "stop_tts"):
            fn = getattr(self.agent, cand, None)
            if fn:
                try:
                    if asyncio.iscoroutinefunction(fn):
                        await fn(parsed)
                    else:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, fn, parsed)
                    logger.debug("Called agent.%s()", cand)
                    return
                except Exception:
                    logger.exception("agent.%s() raised an error", cand)

        logger.warning("Valid interruption detected but no stop handler available on agent")

    def stats(self) -> dict:
        return {"ignored": self._ignored_count, "valid": self._valid_count}
