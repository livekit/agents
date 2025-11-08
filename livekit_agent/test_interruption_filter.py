import asyncio
import logging
from interruption_filter import InterruptionFilter

class MockAgent:
    def __init__(self):
        self.is_speaking = False
        logging.info("🎧 MockAgent initialized.")

    async def start_speaking(self):
        self.is_speaking = True
        logging.info("🗣️ Agent started speaking.")

    async def stop_speaking(self):
        self.is_speaking = False
        logging.info("🤫 Agent finished speaking.")

    async def stop_tts(self):
        logging.info("🔴 [MockAgent] stop_tts() called – Agent stopped speaking.")

    async def handle_user_input(self, text: str):
        logging.info(f"🟢 [MockAgent] processing user input: '{text}'")


async def test():
    global filter_layer
    filter_layer = InterruptionFilter()
    filter_layer.add_ignored_word("hmm sahi hai")  # Hindi filler runtime

    agent = MockAgent()
    await agent.start_speaking()

    # Test English fillers
    await filter_layer.handle_speech_event("uh", 0.95, True, agent)
    await filter_layer.handle_speech_event("umm okay stop", 0.9, True, agent)
    await filter_layer.handle_speech_event("umm", 0.9, True, agent)

    # Test Hindi fillers
    await filter_layer.handle_speech_event("accha", 0.95, True, agent)
    await filter_layer.handle_speech_event("theek", 0.95, True, agent)
    await filter_layer.handle_speech_event("hmm sahi hai", 0.9, True, agent)

    await agent.stop_speaking()

    # User speaking when agent quiet
    await filter_layer.handle_speech_event("hello there", 0.95, False, agent)
    await filter_layer.handle_speech_event("namaste", 0.95, False, agent)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    asyncio.run(test())
