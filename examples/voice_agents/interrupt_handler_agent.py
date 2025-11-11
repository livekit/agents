import os
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli
)
from livekit.plugins import silero, deepgram, groq




IGNORED_FILLERS = {
    "uh", "um", "umm", "hmm", "mmm", "er", "haan",
    "you know", "i mean", "like", "basically", "literally",
    "kinda", "sorta",
    "accha", "yaani", "toh", "matlab", "arey",
    "ano", "eto", "nage", "ehh", "keu-rae", "geu",
    "yani", "bas", "shufu",
    "pues", "donc", "allora", "vale",
    "seri", "apdi", "athu", "enna",
    "ante", "andi", "em",
    "andre", "matte",
    "appo", "alle", "entha"
}

CONFIDENCE_THRESHOLD = 0.60


def is_filler(transcript, confidence):
    words = transcript.lower().strip().split()
    if confidence < CONFIDENCE_THRESHOLD:
        return True
    return all(word in IGNORED_FILLERS for word in words)


class InterruptAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a polite assistant. Speak clearly and conversationally.",
        )

    async def on_transcription(self, event):
        # If agent is currently speaking -> evaluate
        if self.session.agent.is_speaking:
            transcript = event.text
            confidence = event.confidence or 1.0

            if is_filler(transcript, confidence):
                print(f"[IGNORED FILLER] {transcript}")
                return

            print(f"[VALID INTERRUPTION] {transcript}")
            self.session.stop_speaking()  # Stop TTS immediately


async def entrypoint(ctx: JobContext):
    await ctx.connect()


    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova"),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts=deepgram.TTS(model="aura-asteria-en"),  # ✅ NEW TTS
    )


    agent=InterruptAgent()

    await session.start(
        agent=agent,
        room=ctx.room
    )

    session.on("transcription", agent.on_transcription)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
