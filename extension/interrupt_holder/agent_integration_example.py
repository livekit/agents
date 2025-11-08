import asyncio
from interrupt_handler import InterruptHandler

class MockAgent:
    def __init__(self):
        self._speaking = False

    async def start_speaking(self, text):
        self._speaking = True
        print("Agent speaking:", text)
        await asyncio.sleep(2)
        self._speaking = False

    def stop(self):
        if self._speaking:
            print("Agent STOP called")
            self._speaking = False

    def is_speaking(self):
        return self._speaking

async def main():
    agent = MockAgent()
    handler = InterruptHandler(
        is_agent_speaking_cb=agent.is_speaking,
        stop_agent_cb=agent.stop,
        accepted_callback=lambda t: print("User said:", t),
        ignored_words=["uh", "umm", "hmm", "haan"],
    )

    speak_task = asyncio.create_task(agent.start_speaking("Hello, how can I help?"))
    await asyncio.sleep(0.5)
    await handler.on_transcription("uh", confidence=0.9)
    await asyncio.sleep(0.2)
    await handler.on_transcription("stop", confidence=0.9)
    await speak_task

if __name__ == "__main__":
    asyncio.run(main())
