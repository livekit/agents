import asyncio
import os
from livekit.agents import VoiceAgent
from livekit.agents.voice import AgentSpeakingState
from livekit.agents.asr import TranscriptionEvent

IGNORED_FILLERS = os.getenv("IGNORED_FILLERS", "uh,umm,hmm,haan").split(",")
INTERRUPT_KEYWORDS = os.getenv("INTERRUPT_KEYWORDS", "stop,wait,cancel").split(",")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))

class InterruptionAwareAgent(VoiceAgent):
    async def on_transcription(self, event: TranscriptionEvent):
        text = event.text.lower()
        confidence = event.confidence
        agent_speaking = self.speaking_state == AgentSpeakingState.SPEAKING

        self.logger.info(f"ASR: '{text}' | conf={confidence} | agent_speaking={agent_speaking}")

        if confidence < CONFIDENCE_THRESHOLD and agent_speaking:
            self.logger.debug("Ignored low confidence speech while speaking.")
            return

        words = text.split()
        if agent_speaking and all(w in IGNORED_FILLERS for w in words):
            self.logger.debug(f"Ignored filler while agent speaking: '{text}'")
            return

        if agent_speaking and any(word in text for word in INTERRUPT_KEYWORDS):
            self.logger.warning(f"Interrupt detected: '{text}' — stopping")
            await self.stop_speaking()
            return await super().on_transcription(event)

        return await super().on_transcription(event)

if __name__ == "__main__":
    agent = InterruptionAwareAgent()
    agent.run()
