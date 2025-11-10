import re
import logging

class FillerHandler:
    def __init__(self, ignored_words=None, confidence_threshold=0.6):
        self.ignored_words = ignored_words or ["uh", "umm", "hmm", "haan"]
        self.confidence_threshold = confidence_threshold
        self.agent_speaking = False
        logging.basicConfig(level=logging.INFO)

    async def handle_transcription(self, text, confidence):
        text = (text or "").strip().lower()
        if confidence is None:
            confidence = 1.0
        if confidence < self.confidence_threshold:
            logging.info(f"Ignored low-confidence input: {text}")
            return "ignore"

        filler_pattern = r"^(?:\b" + r"\b|\b".join(map(re.escape, self.ignored_words)) + r"\b)+$"
        is_filler_only = bool(re.match(filler_pattern, text))

        if self.agent_speaking:
            if is_filler_only:
                logging.info(f"Ignored filler while speaking: {text}")
                return "ignore"
            else:
                logging.info(f"Valid interruption detected: {text}")
                return "interrupt"
        else:
            logging.info(f"Registered normal speech: {text}")
            return "register"
