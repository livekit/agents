import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.voice.room_io import RoomInputOptions, RoomIO
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("pre-connect-audio-agent")

load_dotenv()


# This example demonstrates the pre-connect audio buffer for instant connect feature.
# It captures what users say during connection time so they don't need to wait for the connection.
# The process works in three steps:
# 1. RoomIO is set up with pre_connect_audio=True
# 2. When connecting to the room, the client sends any audio spoken before connection
# 3. This pre-connection audio is combined with new audio after connection is established


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
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(voice="ash"),
        # use LiveKit's turn detection model
        turn_detection=MultilingualModel(),
    )

    # create and start room_io with pre-connect audio enabled to register the byte stream handler
    room_io = RoomIO(
        agent_session=session,
        room=ctx.room,
        input_options=RoomInputOptions(pre_connect_audio=True, pre_connect_audio_timeout=5.0),
    )
    await room_io.start()

    # connect to room to notify the client to send pre-connect audio buffer,
    await ctx.connect()

    # put the time consuming model/knowledge loading here
    # user audio buffering starts after the room_io is started

    await session.start(agent=MyAgent())


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
