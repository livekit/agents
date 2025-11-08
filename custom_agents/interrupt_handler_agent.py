import logging
import os
import re
from typing import Iterable, List, Optional, Set

from livekit.agents import Agent, AgentSession

logger = logging.getLogger("filler_filter")
logger.setLevel(logging.INFO)

DEFAULT_IGNORED = ["uh", "umm", "um", "hmm", "haan"]
DEFAULT_COMMANDS = ["wait", "stop", "hold", "pause"]

def _csv_env(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    return [w.strip().lower() for w in raw.split(",") if w.strip()]

def normalize_words(text: str) -> List[str]:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)  # strip simple punctuation
    return [p for p in text.split() if p]

class InterruptHandlerMixin:
    """
    Event-driven filter that:
      - ignores filler-only utterances while the agent is speaking
      - accepts fillers when the agent is silent
      - interrupts immediately if command keywords appear (even mixed)
    Works without importing event classes; uses string event names and getattr.
    """

    def __init__(
        self,
        ignored_words: Optional[Iterable[str]] = None,
        command_keywords: Optional[Iterable[str]] = None,
        confidence_threshold: float = 0.55,   # optional: treat very low-conf murmurs as ignorable
    ):
        self.ignored_words: Set[str] = set((ignored_words or _csv_env("FILLER_WORDS", DEFAULT_IGNORED)))
        self.command_keywords: Set[str] = set((command_keywords or _csv_env("COMMAND_KEYWORDS", DEFAULT_COMMANDS)))
        self.confidence_threshold = confidence_threshold
        self._agent_speaking = False  # tracked via events or session.current_speech fallback

        logger.info("Ignored fillers: %s", sorted(self.ignored_words))
        logger.info("Command keywords: %s", sorted(self.command_keywords))

    def _is_agent_speaking(self, session: AgentSession) -> bool:
        """
        Primary: use our own flag (set by speech start/end events) if available.
        Fallback: check session.current_speech (present in many 1.x builds).
        """
        if self._agent_speaking:
            return True
        # Fallback check; safe if attribute is missing.
        return bool(getattr(session, "current_speech", None))

    def attach(self, session: AgentSession):
        """Register callbacks using string event names (no event classes needed)."""

        @session.on("agent_speech_started")
        def _on_speech_started(ev):
            self._agent_speaking = True
            logger.debug("Agent speech started")

        @session.on("agent_speech_ended")
        def _on_speech_ended(ev):
            self._agent_speaking = False
            logger.debug("Agent speech ended")

        @session.on("user_input_transcribed")
        async def _on_transcribed(ev):
            """
            Expected ev fields (provider-dependent): transcript, is_final, language, confidence
            We use getattr(..., default) to be tolerant across versions.
            """
            is_final = getattr(ev, "is_final", True)
            if not is_final:
                return

            transcript = (getattr(ev, "transcript", None) or "").strip()
            if not transcript:
                return

            words = normalize_words(transcript)
            if not words:
                return

            confidence = getattr(ev, "confidence", None)  # may be None
            speaking = self._is_agent_speaking(session)

            # If any command keyword appears, always interrupt immediately.
            if any(w in self.command_keywords for w in words):
                if speaking:
                    logger.info("VALID INTERRUPTION (command while speaking): %r", transcript)
                    try:
                        session.interrupt()
                    except Exception as e:
                        logger.warning("interrupt() failed: %s", e)
                else:
                    logger.info("USER COMMAND (agent silent): %r", transcript)
                return

            if speaking:
                # Filler-only while speaking? Ignore (optionally check confidence)
                if all(w in self.ignored_words for w in words):
                    if confidence is not None:
                        logger.info("IGNORED FILLER (speaking, conf=%.2f): %r", confidence, transcript)
                        if confidence < self.confidence_threshold:
                            logger.debug("Low-confidence murmur ignored.")
                    else:
                        logger.info("IGNORED FILLER (speaking): %r", transcript)

                    # Gentle cleanup so TTS resumes smoothly
                    try:
                        # Stop listening momentarily (not available in all builds)
                        await session.input.set_audio_enabled(False)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        session.clear_user_turn()
                    except Exception:
                        pass
                    return

                # Non-filler content while speaking → treat as real interruption
                logger.info("VALID INTERRUPTION (non-filler while speaking): %r", transcript)
                try:
                    session.interrupt()
                except Exception as e:
                    logger.warning("interrupt() failed: %s", e)
                return

            # Agent silent: everything (even filler) counts as speech
            if all(w in self.ignored_words for w in words):
                logger.info("FILLER WHILE SILENT (registered as speech): %r", transcript)
            else:
                logger.info("USER SPEECH WHILE SILENT: %r", transcript)


class InterruptHandlerAgent(Agent, InterruptHandlerMixin):
    """Agent that attaches the filler filter when the session starts."""

    def __init__(self, **kwargs):
        Agent.__init__(self, **kwargs)
        InterruptHandlerMixin.__init__(self)

    async def on_session_started(self, session: AgentSession):
        self.attach(session)
        logger.info("Filler-word interruption filter attached.")
