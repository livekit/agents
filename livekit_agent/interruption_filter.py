import logging


class InterruptionFilter:
    def __init__(self, ignored_words=None, confidence_threshold=0.6):
        # Default English + Hindi fillers
        default_fillers = ["uh", "umm", "hmm", "haan", "accha", "theek", "bas", "chalo"]
        self.ignored_words = set(ignored_words or default_fillers)
        self.confidence_threshold = confidence_threshold
        logging.info(f"Ignored words list initialized: {sorted(self.ignored_words)}")

    async def handle_speech_event(self, text, confidence, agent_speaking, agent):
        logging.info(f"\n🧩 TEST: '{text}' (conf={confidence}, speaking={agent_speaking})")
        logging.info(f"🎙️ Agent speaking = {agent_speaking}")

        if confidence < self.confidence_threshold:
            logging.info(f"[LOW CONF] Ignored: '{text}' (conf={confidence:.2f})")
            return "IGNORE"

        tokens = {word.lower() for word in text.split()}

        if agent_speaking:
            if tokens.issubset(self.ignored_words):
                logging.info(f"[IGNORED FILLER] {text}")
                return "IGNORE"
            else:
                logging.info(f"[INTERRUPT] {text}")
                await agent.stop_tts()
                return "INTERRUPT"
        else:
            logging.info(f"[USER SPEECH] {text}")
            await agent.handle_user_input(text)
            return "USER_INPUT"

    # Optional: dynamically add more fillers at runtime
    def add_ignored_word(self, word):
        self.ignored_words.add(word.lower())
        logging.info(f"Added to ignored words: {word}")

    def remove_ignored_word(self, word):
        self.ignored_words.discard(word.lower())
        logging.info(f"Removed from ignored words: {word}")

