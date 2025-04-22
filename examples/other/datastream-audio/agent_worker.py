import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.plugins import openai

logger = logging.getLogger("basic-agent")

load_dotenv()

## This is the audio sender. It sends audio to another participant in the room through a datastream.

RECEIVER_IDENTITY = "agent-receiver"


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(),
    )

    # stream audio to another participant in the room through a datastream
    session.output.audio = DataStreamAudioOutput(
        room=ctx.room, destination_identity=RECEIVER_IDENTITY, sample_rate=24000
    )
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
        room_output_options=RoomOutputOptions(audio_enabled=False),
    )

    @session.output.audio.on("playback_finished")
    def _on_playback_finished(ev: PlaybackFinishedEvent):
        logger.info(f"playback finished: {ev.playback_position} interrupted: {ev.interrupted}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
