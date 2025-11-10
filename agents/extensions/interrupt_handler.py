import re
import asyncio

class InterruptHandler:
    """
    A class to intelligently distinguish between filler words and
    genuine user interruptions during a LiveKit voice conversation.
    """

    def __init__(self, ignored_words=None, confidence_threshold=0.6):
        # Default fillers that should be ignored while the agent speaks
        self.ignored_words = ignored_words or ['uh', 'umm', 'hmm', 'haan']
        self.confidence_threshold = confidence_threshold
        self.agent_speaking = False  # Keeps track of whether agent is currently speaking
        self.logs = []  # Store event logs (useful for debugging)

    def set_agent_state(self, speaking: bool):
        """
        Called whenever the agent starts or stops speaking.
        """
        self.agent_speaking = speaking
        print(f"[STATE] Agent speaking: {self.agent_speaking}")

    async def handle_transcript(self, text: str, confidence: float):
        """
        Handle each transcript event (ASR result).
        Returns 'valid' if it's a real interruption,
        or None if it should be ignored.
        """
        text = text.strip().lower()

        # 1. Ignore empty or low-confidence transcriptions
        if not text or confidence < self.confidence_threshold:
            self.logs.append(("ignored_low_conf", text))
            print(f"[IGNORED: low_conf] '{text}'")
            return None

        # 2. If agent is speaking, filter out filler-only speech
        if self.agent_speaking:
            words = re.findall(r'\b\w+\b', text)
            if all(word in self.ignored_words for word in words):
                self.logs.append(("ignored_filler", text))
                print(f"[IGNORED: filler] '{text}'")
                return None

        # 3. If not filler, treat as valid interruption
        self.logs.append(("valid_interrupt", text))
        print(f"[INTERRUPT] '{text}'")
        return text

    def add_ignored_word(self, word: str):
        """Dynamically add a filler word at runtime."""
        if word not in self.ignored_words:
            self.ignored_words.append(word)
            print(f"[CONFIG] Added ignored word: '{word}'")


