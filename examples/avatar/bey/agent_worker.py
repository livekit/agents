import argparse
import logging
import sys
from functools import partial

from dotenv import load_dotenv

from livekit.agents import (
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.plugins import openai
from livekit.plugins.bey import start_bey_avatar_session

logger = logging.getLogger("bey-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext, avatar_id: str | None):
    await ctx.connect()

    # initialize the local agent
    local_agent_session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        # stt=deepgram.STT(),
        # llm=openai.LLM(model="gpt-4o-mini"),
        # tts=cartesia.TTS(),
    )

    # start a Beyond Presence avatar session
    if avatar_id is not None:
        bey_avatar_session = await start_bey_avatar_session(ctx, avatar_id)
    else:
        bey_avatar_session = await start_bey_avatar_session(ctx)  # will use a default stock avatar

    # send all audio generated by the local agent to the avatar agent
    local_agent_session.output.audio = bey_avatar_session.local_agent_audio_output

    # make sure the avatar agent from Beyond Presence has joined the call
    await bey_avatar_session.wait_for_avatar_agent()

    # start local agent with room input but only text room output
    # (the avatar agent will handle your video and audio output)
    await local_agent_session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
        room_output_options=bey_avatar_session.local_agent_room_output_options,
    )

    @local_agent_session.output.audio.on("playback_finished")
    def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
        logger.info(
            "playback_finished",
            extra={"playback_position": ev.playback_position, "interrupted": ev.interrupted},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default avatar is Ege's Stock Avatar from https://docs.bey.dev/avatars/default
    parser.add_argument("--avatar-id", type=str)

    args, remaining_args = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_args  # pass remaining args to livekit cli

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=partial(entrypoint, avatar_id=args.avatar_id),
            worker_type=WorkerType.ROOM,
        )
    )
