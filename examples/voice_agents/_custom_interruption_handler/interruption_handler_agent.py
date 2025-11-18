import os
import asyncio
import logging
from typing import List

from livekit.agents import VoiceAgent
from livekit.agents.utils import TranscriptionEvent

# Optional logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
IGNORED_FILLERS = os.getenv("IGNORED_FILLERS", "uh,umm,hmm,haan").split(",")
INTERRUPT_KEYWORDS = os.getenv("INTERRUPT_KEYWORDS", "stop,wait,cancel").split(",")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# Normalize words
IGNORED_FILLERS = [w.strip().lower() for w in IGNORED_FILLERS if w.strip()]
INTERRUPT_KEYWORDS = [w.strip().lower() for w in INTERRUPT_KEYWORDS if w.strip()]


class InterruptionHandlerVoiceAgent(VoiceAgent):
    """
    Custom LiveKit Agent with smart interruption handler:
      - Ignores filler words while agent is speaking
      - Allows real interruption anytime (e.g., "wait", "stop")
      - Processes filler normally when agent is not speaking
    """

    async def start(self):
        await super().start()
        logging.info("🔊 Interruption Handler Agent started")

    async def on_transcription(self, evt: TranscriptionEvent):
        """Called on every speech detection."""
        text = evt.transcribed_text.strip().lower()
        confidence = evt.confidence or 1.0
        agent_speaking = self.is_speaking()

        logging.info(f"🎙 Detected Speech: '{text}' (confidence={confidence:.2f}, speaking={agent_speaking})")

        # Ignore if confidence is too low
        if confidence < CONFIDENCE_THRESHOLD:
            logging.debug(f"🤫 Ignored due to low confidence: '{text}'")
            return

        # Stop if interrupt keyword detected
        if any(keyword in text for keyword in INTERRUPT_KEYWORDS):
            logging.info(f"🛑 REAL INTERRUPTION DETECTED: '{text}' → Stopping agent!")
            await self.stop_speaking()
            return

        # Ignore filler only **while speaking**
        if agent_speaking and all(word in IGNORED_FILLERS for word in text.split()):
            logging.info(f"🙉 Ignoring filler while agent speaks: '{text}'")
            return

        logging.info(f"📢 Forwarding speech to agent: '{text}'")
        await super().on_transcription(evt)

    async def stop_speaking(self):
        """Forcefully stop TTS output."""
        if self.is_speaking():
            await self.audio_output.stop()
            logging.info("🔇 Agent speech stopped")


async def main():
    agent = InterruptionHandlerVoiceAgent()
    await agent.start()
    await agent.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
