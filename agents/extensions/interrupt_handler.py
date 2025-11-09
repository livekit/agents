# interrupt_handler.py
"""
InterruptFilter extension for LiveKit Agents.

Usage:
- Instantiate InterruptFilter(session, stop_callback=callable_to_stop_agent)
- Call start() to register listeners on the AgentSession (or add into agent initialization).
- Call stop() to unregister.

Design:
- Listens for ASR/transcription events (user_input_transcribed)
- Keeps track of agent-speaking state via TTS playout events (segment_playout_started/ended)
- If agent is speaking: treat filler-only transcripts (from ignored list) as ignorable
- If agent not speaking: treat same words as valid user speech
- Mixed utterances are parsed: if any token outside ignored list exists, treat as valid interruption
- Thread-safe updates to ignored list via update_ignored_words()
"""
from __future__ import annotations
import asyncio
import os
import re
import logging
from typing import Callable, Iterable, Optional, Set

logger = logging.getLogger("interrupt_filter")
logger.setLevel(os.getenv("INTERRUPT_LOG_LEVEL", "INFO"))

# default fillers; can be overridden via env var: IGNORED_WORDS="uh,umm,hmm,haan"
DEFAULT_IGNORED = {"uh", "umm", "hmm", "haan"}

# If ASR provides per-segment confidence, ignore low-confidence speech while agent speaks.
DEFAULT_CONF_THRESHOLD = float(os.getenv("INTERRUPT_CONF_THRESHOLD", "0.5"))


class InterruptFilter:
    def __init__(
        self,
        session,
        stop_callback: Callable[[], None],
        ignored_words: Optional[Iterable[str]] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    ):
        """
        session: AgentSession or object emitting user_input_transcribed and TTS playout events.
        stop_callback: callable invoked when we detect a real interruption and must stop the agent speaking.
        ignored_words: initial list of filler words to ignore while agent speaks.
        """
        self._session = session
        self._stop_cb = stop_callback
        self._agent_speaking = False
        self._lock = asyncio.Lock()
        self._ignored: Set[str] = set(
            w.strip().lower() for w in (ignored_words or os.getenv("IGNORED_WORDS", ",".join(DEFAULT_IGNORED)).split(",") if ignored_words is None else ignored_words)
        )
        self._conf_threshold = conf_threshold

        # compiled regex to split tokens (unicode word boundaries)
        self._word_re = re.compile(r"\b\w+\b", re.UNICODE)

        # registration placeholders
        self._registered = False
        self._handlers = []

        logger.info("InterruptFilter initialized with ignored_words=%s conf_threshold=%s", sorted(self._ignored), self._conf_threshold)

    async def start(self):
        """Register handlers on session events (async safe)."""
        if self._registered:
            return
        # expected events: 'user_input_transcribed', and TTS playout notifications
        # Use session.on(event_name, callback) if available, otherwise try attribute hooks.
        # Wrap callbacks to ensure asyncio compatibility.
        def _on_transcribed(event):
            # run in event loop
            asyncio.create_task(self._handle_transcription(event))

        def _on_tts_playout_started(*_args, **_kwargs):
            asyncio.create_task(self._set_agent_speaking(True))

        def _on_tts_playout_ended(*_args, **_kwargs):
            asyncio.create_task(self._set_agent_speaking(False))

        # register using event-emitter API if present:
        # AgentSession typically emits 'user_input_transcribed' (see docs).
        try:
            self._session.on("user_input_transcribed", _on_transcribed)
            logger.debug("Registered user_input_transcribed listener via session.on")
            # TTS playout events: try forwarder methods if exposed by session/transcription
            # We listen for 'tts_segment_playout_started'/'tts_segment_playout_ended' if they exist
            self._session.on("tts_segment_playout_started", _on_tts_playout_started)
            self._session.on("tts_segment_playout_ended", _on_tts_playout_ended)
            self._registered = True
            self._handlers.extend(
                [
                    ("user_input_transcribed", _on_transcribed),
                    ("tts_segment_playout_started", _on_tts_playout_started),
                    ("tts_segment_playout_ended", _on_tts_playout_ended),
                ]
            )
        except Exception:
            # Fallback: try adding attributes if direct on() not available
            logger.debug("session.on unavailable; trying to attach attributes/callbacks directly")
            if hasattr(self._session, "add_event_handler"):
                self._session.add_event_handler("user_input_transcribed", _on_transcribed)
                self._session.add_event_handler("tts_segment_playout_started", _on_tts_playout_started)
                self._session.add_event_handler("tts_segment_playout_ended", _on_tts_playout_ended)
                self._registered = True
            else:
                # Last resort: monkey patch known forwarders (if available)
                # The integrator must call set_agent_speaking() hooks if no event interface exists.
                logger.warning("Could not register event handlers automatically. Call set_agent_speaking() and feed_transcription() from your integration.")
        logger.info("InterruptFilter started; registered=%s", self._registered)

    async def stop(self):
        """Unregister handlers if possible."""
        if not self._registered:
            return
        try:
            for name, handler in self._handlers:
                self._session.off(name, handler)
            logger.info("InterruptFilter handlers unregistered via off()")
        except Exception:
            logger.debug("session.off unavailable; handlers may remain attached")
        self._registered = False

    async def _set_agent_speaking(self, speaking: bool):
        async with self._lock:
            self._agent_speaking = bool(speaking)
            logger.debug("Agent speaking set to %s", self._agent_speaking)

    # Public integration helpers (if your env can't call start/stop or events)
    async def set_agent_speaking(self, speaking: bool):
        """Call from external code when TTS playout starts/stops."""
        await self._set_agent_speaking(speaking)

    async def feed_transcription(self, transcript: str, is_final: bool = True, confidence: Optional[float] = None, metadata: Optional[dict] = None):
        """Call from external code to feed a transcription event."""
        class DummyEvent:
            pass
        ev = DummyEvent()
        ev.transcript = transcript
        ev.is_final = is_final
        ev.confidence = confidence
        ev.metadata = metadata or {}
        await self._handle_transcription(ev)

    async def _handle_transcription(self, event):
        """
        Core logic:
        - If event has 'confidence' and it's below threshold and agent is speaking -> ignore
        - Tokenize transcript into words; if all tokens are in ignored list and agent is speaking -> ignore
        - Otherwise treat as valid: call stop callback.
        """
        transcript = getattr(event, "transcript", "") or ""
        is_final = getattr(event, "is_final", True)
        conf = getattr(event, "confidence", None)

        # normalize tokens
        tokens = [t.lower() for t in self._word_re.findall(transcript)]
        if not tokens:
            logger.debug("Empty/no tokens in transcription: '%s'", transcript)
            return

        # check confidence low-case
        async with self._lock:
            agent_speaking = self._agent_speaking

        debug_ctx = {"transcript": transcript, "tokens": tokens, "agent_speaking": agent_speaking, "conf": conf}
        logger.debug("Handling transcription event: %s", debug_ctx)

        # If confidence provided and below threshold while agent speaks => ignore
        if agent_speaking and conf is not None and conf < self._conf_threshold:
            logger.info("Ignored low-confidence user speech while agent speaks: '%s' (conf=%s)", transcript, conf)
            return

        # Determine whether all tokens are filler-only
        non_ignored_tokens = [t for t in tokens if t not in self._ignored]

        if agent_speaking:
            if len(non_ignored_tokens) == 0:
                # filler-only utterance while agent speaking => ignore
                logger.info("Ignored filler while agent speaking: '%s' tokens=%s", transcript, tokens)
                return
            else:
                # there is at least one non-filler word => treat as interruption
                logger.info("Detected valid interruption while agent speaking: '%s' non_ignored=%s", transcript, non_ignored_tokens)
                try:
                    # synchronous callback may be OK; allow it to run in event loop
                    res = self._stop_cb()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as e:
                    logger.exception("Exception while calling stop_callback: %s", e)
                return
        else:
            # agent not speaking -> always register user speech (do not suppress)
            logger.info("User speech while agent quiet: registered: '%s'", transcript)
            # If you want to treat specific short fillers as noise while agent is quiet you can,
            # but assignment says they should be registered when agent is quiet.
            return

    # runtime update API
    async def update_ignored_words(self, new_words: Iterable[str]):
        async with self._lock:
            self._ignored = set(w.strip().lower() for w in new_words if w)
            logger.info("Updated ignored_words=%s", sorted(self._ignored))



    def get_ignored_words(self):
        return sorted(self._ignored)
      
    def is_filler_only(self, text: str, confidence: float = 1.0) -> bool:
        """
        Returns True if the text contains only filler words (e.g., uh, umm, hmm)
        based on the ignored words configured in the filter.
        """
        if not text:
            return False


        words = [w.strip().lower() for w in text.split() if w.strip()]
        if not words:
            return False

        # Use the getter if ignored_words is managed internally
        ignored = self.get_ignored_words() if hasattr(self, "get_ignored_words") else getattr(self, "ignored_words", [])

        # If every word in the text is in the ignored list, it's filler-only
        return all(word in ignored for word in words)

        """   
        Returns True if the text contains only filler words (e.g., uh, umm, hmm)
        based on the ignored_words list. Otherwise, returns False.
        """
        if not text:
            return False

        words = [w.strip().lower() for w in text.split() if w.strip()]
        if not words:
            return False

        # If every word in the text is a filler, return True
        return all(word in self.ignored_words for word in words)

