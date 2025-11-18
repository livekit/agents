class IntelligentInterruptionHandler:
    def __init__(self, ignored_words: list, speech_threshold: float = 0.6):
        self._ignored_words = {w.lower() for w in ignored_words}
        self._agent_is_speaking = False
        self._speech_threshold = speech_threshold

    def update_agent_speaking_status(self, is_speaking: bool):
        # Must be thread-safe in a real app, but for a simple test, this works
        self._agent_is_speaking = is_speaking

    def should_interrupt(self, text: str, confidence: float) -> bool:
        """
        Returns True if a real interrupt should happen, False if it should be ignored.
        """
        text = text.lower().strip()
        if not text:
            return False # Ignore empty transcription

        # Rule 1: Ignore low confidence (Background murmur)
        if confidence < self._speech_threshold:
            # We don't log here, as we only log when we make a decision
            return False

        # Rule 2: Agent is Quiet (Any speech registers as valid)
        if not self._agent_is_speaking:
            return True

        # Rule 3: Agent is Speaking (Check for fillers vs. commands)
        
        # Check if ALL words in the transcription are defined as filler words
        words = set(text.split())
        
        # issubset checks if 'words' is entirely contained in '_ignored_words'
        is_filler_only = words.issubset(self._ignored_words)
        
        # The transcription must contain at least one non-filler word to be a real interrupt
        return not is_filler_only