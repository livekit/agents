import os
import logging
from typing import Any, Dict, AsyncIterator

logger = logging.getLogger("livekit.interruption_filter")
if not logger.handlers:
    # basic console handler if none configured
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Read environment variables
IGNORED_WORDS = os.getenv("IGNORED_WORDS", "uh,umm,erm,like").split(",")
try:
    FILLER_CONFIDENCE_THRESHOLD = float(os.getenv("FILLER_CONFIDENCE_THRESHOLD", "0.8"))
except ValueError:
    FILLER_CONFIDENCE_THRESHOLD = 0.8
try:
    IGNORE_WHEN_CONFIDENCE_LESS_THAN = float(os.getenv("IGNORE_WHEN_CONFIDENCE_LESS_THAN", "0.5"))
except ValueError:
    IGNORE_WHEN_CONFIDENCE_LESS_THAN = 0.5


class FillerInterruptionFilter:
    """
    Filter that prevents the LiveKit agent from interrupting
    when a user says filler words like 'uh', 'umm', etc.
    """

    def __init__(self):
        self.ignored_words = [w.strip().lower() for w in IGNORED_WORDS if w.strip()]
        self.filler_conf_threshold = FILLER_CONFIDENCE_THRESHOLD
        self.ignore_conf_threshold = IGNORE_WHEN_CONFIDENCE_LESS_THAN

    def should_ignore(self, transcript: Dict[str, Any]) -> bool:
        """
        Decide whether to ignore a voice activity event.
        transcript: dict containing keys like "text" and "confidence".
        Returns True to IGNORE the event (i.e., treat as filler / low confidence).
        """
        text = (transcript.get("text") or "").strip().lower()
        confidence = transcript.get("confidence")
        # default confidence if none provided
        if confidence is None:
            confidence = 1.0

        # If confidence is very low, ignore (avoid reacting to garbage)
        if confidence < self.ignore_conf_threshold:
            logger.info(f"Ignored event: low confidence ({confidence:.2f}) text='{text}'")
            return True

        # If the text is exactly a filler and its confidence is high enough, ignore
        if text in self.ignored_words and confidence >= self.filler_conf_threshold:
            logger.info(f"Ignored filler word: '{text}' (conf={confidence:.2f})")
            return True

        # Otherwise accept
        logger.info(f"Accepted speech: '{text}' (conf={confidence:.2f})")
        return False


# Async wrapper node (for LiveKit graph pipelines)
async def filler_filter_node(event_stream: AsyncIterator[Dict[str, Any]]):
    """
    Async generator that filters out filler interruption events
    before passing them downstream.

    Usage:
      async for event in filler_filter_node(my_event_stream):
          ... handle accepted events ...
    """
    filter_instance = FillerInterruptionFilter()
    async for event in event_stream:
        try:
            if not filter_instance.should_ignore(event):
                yield event
        except Exception as exc:
            logger.exception("Error in filler_filter_node, passing event through: %s", exc)
            yield event
