import logging
from .stages.filler_filter import FillerFilter
from .stages.confidence_filter import ConfidenceFilter
from .stages.mixed_intent_detector import MixedIntentDetector
from .stages.state_tracker import StateTracker

logger = logging.getLogger("InterruptionPipeline")


class InterruptionPipeline:
    """
    Multi-stage interruption detection pipeline for LiveKit agents.
    Determines whether a transcript should:
        - interrupt TTS
        - be ignored
        - be treated as normal speech
    With detailed debug logs for each stage.
    """

    def __init__(
        self,
        ignored_words=None,
        min_confidence=0.6,
        smooth_window=3,
        command_keywords=None,
    ):
        self.state = StateTracker()

        self.conf_filter = ConfidenceFilter(
            min_confidence=min_confidence, window=smooth_window
        )
        self.filler_filter = FillerFilter(
            ignored_words or ["uh", "umm", "hmm", "haan"]
        )
        self.intent_detector = MixedIntentDetector(
            command_keywords or ["stop", "wait", "no", "hold"]
        )

    def set_agent_speaking(self, is_speaking: bool):
        """
        Called when TTS starts or ends.
        """
        logger.debug(f"[STATE] Agent speaking = {is_speaking}")
        self.state.set_speaking(is_speaking)

    def process(self, transcript: str, confidence: float):
        transcript = transcript.strip()

        # --- EMPTY TRANSCRIPT ---
        if not transcript:
            logger.debug("[IGNORE][EMPTY] Transcript was empty.")
            return "ignore"

        # --- AGENT IS QUIET ---
        if not self.state.is_speaking():
            logger.debug(f"[SPEECH][AGENT QUIET] '{transcript}'")
            return "speech"

        # --- CONFIDENCE FILTER ---
        if not self.conf_filter.is_confident(transcript, confidence):
            logger.debug(
                f"[IGNORE][CONFIDENCE] Raw={confidence:.2f} | Text='{transcript}'"
            )
            return "ignore"

        # --- FILLER FILTER ---
        if self.filler_filter.is_filler_only(transcript):
            logger.debug(f"[IGNORE][FILLER] Filler-only detected → '{transcript}'")
            return "ignore"

        # --- COMMAND DETECTION (INTERRUPT USER INTENTION) ---
        if self.intent_detector.contains_command(transcript):
            logger.debug(
                f"[INTERRUPT][COMMAND] Keyword detected in → '{transcript}'"
            )
            return "interrupt"

        # --- VALID USER SPEECH (FALLBACK) ---
        logger.debug(f"[SPEECH][VALID] '{transcript}'")
        return "speech"
print("🔥")