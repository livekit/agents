import asyncio
import logging
from interruption_filter import InterruptionFilter

class VoiceAgent:
    def __init__(self):
        self.is_speaking = False
        logging.info("VoiceAgent initialized.")

    async def start_speaking(self):
        self.is_speaking = True
        logging.info("Agent started speaking.")

    async def stop_speaking(self):
        self.is_speaking = False
        logging.info("Agent finished speaking.")

    async def stop_tts(self):
        logging.info("stop_tts() called – interrupt detected.")

    async def handle_user_input(self, text: str):
        logging.info(f"Processing user input: {text}")


async def main():
    # True fillers only (English + Hindi)
    ignored_words = [
        "uh", "umm", "hmm", "haan",         
        "accha", "theek",                                             
    ]
    filter_layer = InterruptionFilter(ignored_words=ignored_words)

    # Initialize agent
    agent = VoiceAgent()
    await agent.start_speaking()

    # --- Simulated speech events ---
    # English fillers
    await filter_layer.handle_speech_event("uh", 0.95, agent.is_speaking, agent)
    await filter_layer.handle_speech_event("umm okay stop", 0.9, agent.is_speaking, agent)
    await filter_layer.handle_speech_event("umm", 0.9, agent.is_speaking, agent)

    # Hindi fillers
    await filter_layer.handle_speech_event("accha", 0.95, agent.is_speaking, agent)
    await filter_layer.handle_speech_event("theek", 0.95, agent.is_speaking, agent)

    # Dynamic runtime update example (safe filler)
    filter_layer.add_ignored_word("bas thik hai")  # Example of adding new filler at runtime
    await filter_layer.handle_speech_event("bas thik hai", 0.95, agent.is_speaking, agent)

    # Agent stops speaking
    await agent.stop_speaking()

    # User speech when agent is quiet
    await filter_layer.handle_speech_event("hello there", 0.95, agent.is_speaking, agent)
    await filter_layer.handle_speech_event("namaste", 0.95, agent.is_speaking, agent)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    asyncio.run(main())
