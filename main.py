import asyncio
from interrupt_handler import InterruptHandler

# Simulated LiveKit agent class
class DummyAgent:
    def __init__(self):
        self.speaking = True  # Start as speaking

    def is_speaking(self):
        return self.speaking

    async def start_tts(self, text):
        print(f"Agent speaking: {text}")
        self.speaking = True

    async def stop_tts(self):
        print("Agent TTS STOPPED (interrupted by user)")
        self.speaking = False



agent = DummyAgent()

def stop_callback():
    return asyncio.create_task(agent.stop_tts())

handler = InterruptHandler(on_interrupt=stop_callback)


async def simulate_user_speech():
    # Example events that mimic user speech
    events = [
        {"transcript": "uh", "confidence": 0.95},
        {"transcript": "umm", "confidence": 0.9},
        {"transcript": "umm okay stop", "confidence": 0.95},
        {"transcript": "stop", "confidence": 0.98},
    ]

    for ev in events:
        print(f"\n Detected user speech: {ev['transcript']}")
        await handler.on_transcript(ev, agent_state_getter=agent.is_speaking)
        await asyncio.sleep(1)

asyncio.run(simulate_user_speech())
