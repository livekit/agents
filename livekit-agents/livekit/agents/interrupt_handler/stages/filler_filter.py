import logging
import re

logger = logging.getLogger("FillerFilter")


class FillerFilter:
    """
    Detects filler content EVEN IF the STT removes 'uh', 'umm', etc.
    Uses transcript length + pattern rules.
    """

    def __init__(self, ignored_words=None):
        # Words Deepgram sometimes outputs for filler-like sounds
        self.ignored_words = set(
            (ignored_words or ["uh", "umm", "hmm", "haan", "eh", "ah", "mm"])
        )

        # Pattern for repeated short noise-like transcripts: "a", "ah", "uhh", "mm"
        self.noise_pattern = re.compile(r"^[aehmu]{1,4}$", re.IGNORECASE)

    def is_filler_only(self, transcript: str) -> bool:
        text = transcript.strip().lower()

        # Case 1: very short noise-like transcripts (e.g., "a", "mm", "uhh")
        if self.noise_pattern.match(text):
            logger.debug(f"[FILLER-DETECT] Noise pattern match → '{text}'")
            return True

        # Case 2: extremely short random outputs (< 3 characters)
        if len(text) <= 2:
            logger.debug(f"[FILLER-DETECT] Very short text → '{text}'")
            return True

        # Case 3: STT replaced filler with weird partial words (e.g., "i-", "uhm")
        if any(word in text for word in self.ignored_words):
            logger.debug(f"[FILLER-DETECT] Explicit filler word detected → '{text}'")
            return True

        return False