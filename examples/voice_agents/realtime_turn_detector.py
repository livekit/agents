import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("realtime-turn-detector")
logger.setLevel(logging.INFO)

load_dotenv()

## This example demonstrates how to use LiveKit's turn detection model with a realtime LLM.
## Since the current turn detection model runs in text space, it will need to be combined
## with a STT model, even though the audio is going directly to the Realtime API.
## In this example, speech is being processed in parallel by both the STT and the realtime API


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        allow_interruptions=True,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.realtime.RealtimeModel(
            voice="alloy",
            # it's necessary to turn off turn detection in the OpenAI Realtime API in order to use
            # LiveKit's turn detection model
            turn_detection=None,
            input_audio_transcription=None,  # we use Deepgram STT instead
        ),
    )
    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
