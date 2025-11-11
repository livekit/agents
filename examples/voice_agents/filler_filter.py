# examples/voice_agents/filler_filter.py
import os
import asyncio
import logging
import re
from typing import Callable, Iterable

logger = logging.getLogger("filler-filter")

def _load_ignored_words():
    raw = os.getenv("IGNORED_WORDS", "uh,umm,hmm,haan,stop,uh, um, umm, hmm, ah, oh, huh, hm")
    return set(w.strip().lower() for w in raw.split(",") if w.strip())

IGNORED_WORDS = _load_ignored_words()
FILLER_CONFIDENCE_THRESHOLD = float(os.getenv("FILLER_CONFIDENCE_THRESHOLD", "0.6"))
# e.g., ignore filler-only segments only if confidence < threshold

def _tokenize(text: str):
    # simple tokenization — adapt if you have token list from ASR
    return [t for t in re.findall(r"\w+['-]?\w*|\w+", text.lower())]

class FillerInterruptionHandler:
    def __init__(self, ignored_words: Iterable[str] = None, conf_threshold: float = None):
        self.ignored_words = set(ignored_words) if ignored_words is not None else IGNORED_WORDS
        if conf_threshold is None:
            conf_threshold = FILLER_CONFIDENCE_THRESHOLD
        self.conf_threshold = conf_threshold
        self.speaking = False
        self._attached = False
        self._orig_handlers = {}

    def _is_filler_only(self, text, confidence=None):
        tokens = _tokenize(text)
        if not tokens:
            return True
        for t in tokens:
            if t not in self.ignored_words:
                return False
        # all tokens are in ignored list
        if confidence is None:
            return True
        return float(confidence) < self.conf_threshold

    async def handle_transcript(self, transcript_text: str, confidence: float = None, is_final: bool = True, raw_event=None):
        """
        Decision: if filler-only and agent is speaking -> ignore.
        Otherwise forward by calling original handler (if present) or logging as forwarded.
        """
        try:
            if self.speaking and self._is_filler_only(transcript_text, confidence):
                logger.info("IGNORED filler transcript while speaking: %r (conf=%s)", transcript_text, confidence)
                # do not propagate
                return {"ignored": True}
            # not filler-only OR agent not speaking -> forward
            logger.info("FORWARDING transcript: %r (conf=%s)", transcript_text, confidence)
            # call original handler if present
            if "transcript" in self._orig_handlers and self._orig_handlers["transcript"]:
                try:
                    # original may be sync or async
                    orig = self._orig_handlers["transcript"]
                    if asyncio.iscoroutinefunction(orig):
                        return await orig(transcript_text, confidence=confidence, is_final=is_final, raw_event=raw_event)
                    else:
                        return orig(transcript_text, confidence=confidence, is_final=is_final, raw_event=raw_event)
                except Exception as e:
                    logger.exception("Original transcript handler failed: %s", e)
            return {"ignored": False}
        except Exception:
            logger.exception("Error in handle_transcript")
            return {"ignored": False}

    def attach(self, session):
        """
        Attach to an AgentSession instance. Tries:
         1) session.register_transcript_handler(handler)
         2) session.on("transcript", handler)
         3) monkeypatch session._handle_transcript or similar internal
        Also tries to register start/stop speaking handlers.
        """
        if self._attached:
            return
        # store originals if present
        # 1) register_transcript_handler
        if hasattr(session, "register_transcript_handler"):
            try:
                self._orig_handlers["transcript"] = getattr(session, "_transcript_handler", None)
                session.register_transcript_handler(self.handle_transcript)
                logger.info("Attached via register_transcript_handler")
                self._attached = True
                return
            except Exception:
                logger.exception("register_transcript_handler failed")
        # 2) on('transcript')
        if hasattr(session, "on"):
            try:
                def sync_wrapper(ev):
                    text = getattr(ev, "text", "")
                    is_final = getattr(ev, "is_final", True)
                    confidence = getattr(ev, "confidence", None)

                    logger.info(
                        f"[FILLER-FILTER-RAW] text={text!r} conf={confidence} final={is_final} speaking={self.speaking}"
                    )

                    asyncio.create_task(
                        self.handle_transcript(
                            transcript_text=text,
                            confidence=confidence,
                            is_final=is_final,
                            raw_event=ev
                        )
                    )

                session.on("user_input_transcribed", sync_wrapper)
                logger.info("✅ Attached via session.on('user_input_transcribed')")
                self._attached = True
            except Exception as e:
                logger.exception("Failed to attach user_input_transcribed hook")

        # 3) monkey-patch likely internal method names (best effort)
        for name in ("_on_transcript", "_handle_transcript", "handle_transcript_event", "process_transcript"):
            if hasattr(session, name):
                orig = getattr(session, name)
                self._orig_handlers["transcript"] = orig
                async def _wrapped(*args, **kwargs):
                    # try to extract text/confidence/is_final from args/kwargs
                    text = None
                    confidence = None
                    is_final = kwargs.get("is_final", True)
                    # common patterns: (text,), (event,), (raw_event,)
                    if args:
                        # If first arg is a dict like event
                        first = args[0]
                        if isinstance(first, dict) and "transcript" in first:
                            text = first.get("transcript") or first.get("text")
                            confidence = first.get("confidence")
                            raw = first
                        else:
                            # assume the first arg is text
                            text = str(first)
                            raw = None
                    else:
                        text = kwargs.get("text") or kwargs.get("transcript") or ""
                        raw = kwargs
                    # call filter
                    res = await self.handle_transcript(text, confidence=confidence, is_final=is_final, raw_event=raw)
                    if res and res.get("ignored"):
                        return  # swallow
                    # forward to original
                    if asyncio.iscoroutinefunction(orig):
                        return await orig(*args, **kwargs)
                    else:
                        return orig(*args, **kwargs)
                setattr(session, name, _wrapped)
                logger.info("Attached by monkeypatching %s", name)
                self._attached = True
                break

        # 4) speaking state detection: try to register tts events or monkeypatch generate_reply
        # record original generate_reply if exists
        if hasattr(session, "generate_reply"):
            gen = session.generate_reply
            self._orig_handlers["generate_reply"] = gen
            async def _gen_and_track(*args, **kwargs):
                # set speaking True before generating reply
                self.speaking = True
                try:
                    res = await gen(*args, **kwargs)
                    return res
                finally:
                    # we can't know exactly when TTS stops, but many SDKs fire tts_end events;
                    # set a small delay fallback (adjust if TTS event exists)
                    await asyncio.sleep(0.1)
                    # do not set False immediately; the SDK might provide an event better
                    self.speaking = False
            session.generate_reply = _gen_and_track

        # Try to attach tts start/stop if SDK exposes them
        if hasattr(session, "on_tts_start"):
            try:
                session.on_tts_start(lambda *_: setattr(self, "speaking", True))
            except Exception:
                pass
        if hasattr(session, "on_tts_end"):
            try:
                session.on_tts_end(lambda *_: setattr(self, "speaking", False))
            except Exception:
                pass

        logger.info("FillerInterruptionHandler attached (best-effort). speaking initial=%s", self.speaking)
