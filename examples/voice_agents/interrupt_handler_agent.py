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
     "hmm", "hmmm", "hmm", "hmmhmm", "mmm", "mmmm", "mmh", "mm-hmm",
    "mm-mm", "mhm", "mhmm", "hm", "hm?", "hmh", "hmhmm", "uh", "uhh",
    "uhhh", "um", "umm", "ummm", "uh-huh", "uh-uh", "huh", "huh",
    "hnnn", "hunh", "humm", "hummm", "um-hmm", "mm", "mm", "mmhmm",
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
    "appo", "alle", "entha",
    "continue", "go on", "carry on", "keep going", "keep it up", "proceed",
    "move forward", "press on", "push on", "keep at it", "hold on",
    "stay on it", "keep moving", "resume", "persist", "maintain",
    "keep working", "follow through", "advance", "keep progressing",
    "further", "stick with it", "march on", "keep doing", "remain at it",
    "don't stop", "continue onward", "onward", "carry forward", "keep pushing",
    "keep rolling", "go ahead", "take it forward", "move ahead", "back to it",
    "keep steady", "keep going as is", "hold course", "stay the course",
    "keep pace", "progress", "push forward", "maintain momentum",
    "drive ahead", "keep advancing", "keep flowing", "continue the same",
    "keep following", "keep the momentum", "move along",
}

CONFIDENCE_THRESHOLD = 0.60


def is_filler(transcript, confidence):

    words = transcript.lower().strip().split()

    # Ignore low-confidence audio (breathing, background room noise)
    if confidence < CONFIDENCE_THRESHOLD:
        return True

    # Ignore very short hesitation sounds like "mhmm", "mm", "huh"
    if len(words) == 1 and len(words[0]) <= 3:
        return True

    # Ignore utterances made entirely of filler tokens
    if all(w in IGNORED_FILLERS for w in words):
        return True

    # Otherwise → contains meaningful speech → treat as interruption
    return False


class InterruptAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a polite assistant. Speak clearly and conversationally.",
        )
        self.interrupted = False  # NEW

    async def on_transcription(self, event):
        # Only consider interruptions while the agent is speaking
        if self.session.agent.is_speaking:
            transcript = event.text
            confidence = event.confidence or 1.0

            if is_filler(transcript, confidence):
                print(f"[IGNORED FILLER] {transcript}")
                return

            # Meaningful user speech → interrupt
            print(f"[VALID INTERRUPTION] {transcript}")
            self.session.stop_speaking()
            self.interrupted = True  # NEW

    async def on_vad(self, event):
        # event.speech == False → silence detected
        if self.interrupted and not event.speech:
            print("[USER FINISHED SPEAKING] → Resuming agent response...")
            self.interrupted = False
            await self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    await ctx.connect()


    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova"),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts=deepgram.TTS(model="aura-asteria-en", stream = True),  # ✅ NEW TTS
    )


    agent=InterruptAgent()

    await session.start(
        agent=agent,
        room=ctx.room
    )

    session.on("transcription", agent.on_transcription)
    session.on("vad", agent.on_vad)   # Auto resume on silence


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
