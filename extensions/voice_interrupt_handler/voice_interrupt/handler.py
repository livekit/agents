# handler.py
import os
import asyncio
import logging
from typing import Set, Optional

logger = logging.getLogger("voice_interrupt_handler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    logger.addHandler(handler)


class InterruptHandler:
    """
    Extension-layer interrupt handler. Attach to an AgentSession instance
    (no SDK changes). Listens to session events and filters ASR transcripts.

    Usage:
        handler = InterruptHandler(session, ignored_words={"uh","umm"}, stop_words={"stop","wait"})
        handler.start()  # attaches callbacks
        # handler.stop() to detach if needed
    """

    def __init__(
        self,
        session,
        ignored_words: Optional[Set[str]] = None,
        stop_words: Optional[Set[str]] = None,
        min_confidence: float = 0.0,
    ):
        self.session = session
        self._lock = asyncio.Lock()
        # defaults (can be overridden via env or constructor)
        default_ignored = {"uh", "umm", "hmm", "haan"}
        self.ignored_words = set(
            (os.getenv("IGNORED_WORDS") or "").split(",")
        ) if os.getenv("IGNORED_WORDS") else set(default_ignored)

        default_stop = {"stop", "wait"}
        self.stop_words = set(stop_words or default_stop)

        # transcripts below min_confidence are candidates to ignore during speaking
        self.min_confidence = float(os.getenv("MIN_ASR_CONFIDENCE") or min_confidence)

        # boolean flag: True when the agent is currently speaking
        self._agent_speaking = False
        self._attached = False

    # -------------------------
    # Public lifecycle
    # -------------------------
    def start(self):
        if self._attached:
            return
        try:
            # The LiveKit SDK's .on(...) requires synchronous callbacks.
            # Wrap our async handlers in small sync functions that schedule them.
            def _sync_on_agent_state_changed(event):
                try:
                    asyncio.get_running_loop().create_task(self._on_agent_state_changed(event))
                except RuntimeError:
                    # No running loop (shouldn't happen during normal runtime), fallback:
                    asyncio.create_task(self._on_agent_state_changed(event))

            def _sync_on_user_input_transcribed(event):
                try:
                    asyncio.get_running_loop().create_task(self._on_user_input_transcribed(event))
                except RuntimeError:
                    asyncio.create_task(self._on_user_input_transcribed(event))

            # Register the sync wrappers with the session event bus
            # Keep references so we can (optionally) detach later
            self._subs = {}
            self.session.on("agent_state_changed", _sync_on_agent_state_changed)
            self._subs["agent_state_changed"] = _sync_on_agent_state_changed

            self.session.on("user_input_transcribed", _sync_on_user_input_transcribed)
            self._subs["user_input_transcribed"] = _sync_on_user_input_transcribed

            self._attached = True
            logger.info("InterruptHandler attached to session")
        except Exception as e:
            logger.error("Failed to attach InterruptHandler: %s", e)
            raise
        
    def stop(self):
        """Detach callbacks if the session API supports it. Otherwise no-op."""
        # Try to remove registered handlers if the session provides an `off` or `remove_listener` API.
        try:
            if hasattr(self, "_subs") and self._subs:
                # common SDK names: off(event, callback) or remove_listener
                off_fn = getattr(self.session, "off", None) or getattr(self.session, "remove_listener", None)
                if callable(off_fn):
                    for ev, cb in list(self._subs.items()):
                        try:
                            off_fn(ev, cb)
                        except Exception:
                            # some SDKs don't support specific off signatures; ignore
                            logger.debug("Could not unsubscribe %s", ev)
        except Exception:
            logger.exception("Error while detaching callbacks")
        finally:
            self._attached = False
            logger.info("InterruptHandler detached (if unsubscribed)")
            
    # -------------------------
    # Config/runtime updates (bonus)
    # -------------------------
    async def update_ignored_words(self, new_list: Set[str]):
        async with self._lock:
            self.ignored_words = set(new_list)
        logger.info("Updated ignored words: %s", sorted(self.ignored_words))

    # -------------------------
    # Event handlers (async)
    # -------------------------
    async def _on_agent_state_changed(self, event):
        """
        Expected 'event' to have a field 'new_state' like 'speaking' or 'idle'.
        This method is intentionally small and locks only the flag.
        """
        new_state = getattr(event, "new_state", None) or event.get("new_state", None)
        async with self._lock:
            self._agent_speaking = (new_state == "speaking")
        logger.debug("Agent state changed -> speaking=%s", self._agent_speaking)

    async def _on_user_input_transcribed(self, event):
        """
        Expected 'event' to contain:
            - transcript (str)
            - is_final (bool)
            - confidence (optional float)
        Behavior:
            - If not final → ignore
            - If agent speaking:
                - If transcript contains any stop word => call session.interrupt()
                - Else if all tokens are in ignored_words (and confidence >= threshold) => ignore (log)
                - Else => session.interrupt() (meaningful speech)
            - If agent NOT speaking:
                - Do nothing (let session handle user input normally)
        """
        # only act on final transcripts
        is_final = getattr(event, "is_final", None)
        if is_final is False:
            return

        transcript = (getattr(event, "transcript", "") or event.get("transcript", "")).strip().lower()
        if not transcript:
            return

        confidence = getattr(event, "confidence", None)
        if confidence is None:
            try:
                confidence = float(event.get("confidence", 1.0))
            except Exception:
                confidence = 1.0

        async with self._lock:
            agent_speaking = self._agent_speaking
            ignored_words = set(self.ignored_words)
            stop_words = set(self.stop_words)
            min_conf = float(self.min_confidence)

        words = [w.strip() for w in transcript.split() if w.strip()]

        # If agent is speaking -> we filter/decide
        if agent_speaking:
            # 1) immediate stop if stop-word included
            if any(w in stop_words for w in words):
                logger.info('Meaningful interruption detected: "%s" -> interrupting agent', transcript)
                # call interrupt on session API (assumes session.interrupt exists)
                try:
                    self.session.interrupt()
                except Exception:
                    # fallback: try calling a generic stop method, or log if not available
                    logger.exception("session.interrupt() failed or not available")
                return

            # 2) If transcript appears to be only filler words AND meets confidence threshold (if provided)
            if words and all(w in ignored_words for w in words):
                if confidence >= min_conf:
                    logger.info('Ignored filler while agent speaking: "%s" (conf=%s)', transcript, confidence)
                    # intentionally swallow the input (do not forward into conversation)
                    return
                else:
                    # low confidence -> safer to ignore (avoid false interrupt). still log at debug
                    logger.debug('Low-confidence filler during agent: "%s" conf=%s -> ignoring', transcript, confidence)
                    return

            # 3) mixed or other words -> treat as meaningful interruption
            logger.info('Non-filler speech detected while agent speaks: "%s" -> interrupting', transcript)
            try:
                self.session.interrupt()
            except Exception:
                logger.exception("session.interrupt() failed or not available")
            return

        # Agent not speaking -> do nothing; the session should handle user input normally.
        logger.debug('User spoke while agent quiet: "%s" -> pass through', transcript)
        return
