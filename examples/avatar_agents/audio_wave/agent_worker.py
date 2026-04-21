import logging
import os
from dataclasses import asdict, dataclass

import httpx
from dotenv import load_dotenv

from livekit import api, rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.io import PlaybackFinishedEvent, PlaybackStartedEvent
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.plugins import silero

load_dotenv()

logger = logging.getLogger("avatar-example")

AVATAR_IDENTITY = "avatar_worker"
AVATAR_DISPATCHER_URL = os.getenv("AVATAR_DISPATCHER_URL", "http://localhost:8089/launch")


@dataclass
class AvatarConnectionInfo:
    room_name: str
    url: str
    """LiveKit server URL"""
    token: str
    """Token for avatar worker to join"""


async def launch_avatar(ctx: JobContext, avatar_dispatcher_url: str, avatar_identity: str) -> None:
    """
    Send a request to the avatar service for it to join the room

    This function should be wrapped in a avatar plugin.
    """

    # create a token for the avatar to join the room
    token = (
        api.AccessToken()
        .with_identity(avatar_identity)
        .with_name("Avatar Runner")
        .with_grants(api.VideoGrants(room_join=True, room=ctx.room.name))
        .with_kind("agent")
        .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: ctx.local_participant_identity})
        .to_jwt()
    )

    logger.info(f"Sending connection info to avatar dispatcher {avatar_dispatcher_url}")
    connection_info = AvatarConnectionInfo(room_name=ctx.room.name, url=ctx._info.url, token=token)
    async with httpx.AsyncClient() as client:
        response = await client.post(avatar_dispatcher_url, json=asdict(connection_info))
        response.raise_for_status()
    logger.info("Avatar handshake completed")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = Agent(instructions="Talk to me!")
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemini-2.5-flash"),
        tts=inference.TTS("cartesia/sonic-3"),
        vad=silero.VAD.load(),
        resume_false_interruption=False,
    )

    await launch_avatar(ctx, AVATAR_DISPATCHER_URL, AVATAR_IDENTITY)
    session.output.audio = DataStreamAudioOutput(
        ctx.room,
        destination_identity=AVATAR_IDENTITY,
        # (optional) wait for the avatar to publish video track before generating a reply
        wait_remote_track=rtc.TrackKind.KIND_VIDEO,
        # the example avatar_runner uses AvatarRunner which sends lk.playback_started
        wait_playback_start=True,
    )

    # start agent with room input and room text output
    await session.start(agent=agent, room=ctx.room)

    @session.output.audio.on("playback_finished")
    def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
        # the avatar should notify when the audio playback is finished
        logger.info(
            "playback_finished",
            extra={
                "playback_position": ev.playback_position,
                "interrupted": ev.interrupted,
            },
        )

    @session.output.audio.on("playback_started")
    def on_playback_started(ev: PlaybackStartedEvent) -> None:
        # the avatar should notify when the audio playback is started
        logger.info(
            "playback_started",
            extra={"created_at": ev.created_at},
        )

    await session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
