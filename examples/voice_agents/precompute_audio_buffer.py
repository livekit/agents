import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.voice.room_io import PreConnectAudioHandler, RoomInputOptions
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# uncomment to enable Krisp background voice/noise cancellation
# currently supported on Linux and MacOS
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "You are curious and friendly, and have a sense of humor.",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # register a pre-connect audio handler before connecting to the room so that
    # it won't miss the audio buffer (TODO (long): does this makes sense?)
    pre_connect_audio = PreConnectAudioHandler(ctx.room).register()

    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(voice="ash"),
        # use LiveKit's turn detection model
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            pre_connect_audio=pre_connect_audio,
            pre_connect_audio_timeout=5.0,
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
