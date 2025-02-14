import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from functools import partial

import httpx
from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.pipeline import AgentTask, PipelineAgent
from livekit.agents.pipeline.datastream_io import DataStreamOutput
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from livekit.agents.pipeline.room_io import (
    LK_PUBLISH_FOR_ATTR,
    RoomInput,
    RoomInputOptions,
    RoomTranscriptEventSink,
)
from livekit.agents.pipeline.transcription import TextSynchronizer
from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


AVATAR_IDENTITY = "avatar_worker"


@dataclass
class AvatarConnectionInfo:
    room_name: str
    url: str
    """LiveKit server URL"""
    token: str
    """Token for avatar worker to join"""


async def launch_avatar_worker(
    ctx: JobContext, avatar_dispatcher_url: str, avatar_identity: str
) -> None:
    """Wait for worker participant to join and start streaming"""
    # create a token for the avatar worker
    agent_identity = ctx.room.local_participant.identity
    token = (
        api.AccessToken()
        .with_identity(avatar_identity)
        .with_name("Avatar Runner")
        .with_grants(api.VideoGrants(room_join=True, room=ctx.room.name))
        .with_kind("agent")
        .with_attributes({LK_PUBLISH_FOR_ATTR: agent_identity})
        .to_jwt()
    )

    logger.info(f"Sending connection info to avatar dispatcher {avatar_dispatcher_url}")
    connection_info = AvatarConnectionInfo(
        room_name=ctx.room.name, url=ctx._info.url, token=token
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            avatar_dispatcher_url, json=asdict(connection_info)
        )
        response.raise_for_status()
    logger.info("Avatar handshake completed")

    # wait for the remote participant to join
    await ctx.wait_for_participant(
        identity=avatar_identity, kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    )
    logger.info("Avatar runner joined")


async def entrypoint(ctx: JobContext, avatar_dispatcher_url: str):
    await ctx.connect()

    agent = PipelineAgent(
        task=AgentTask(
            instructions="Talk to me!",
            llm=openai.realtime.RealtimeModel(),
            # stt=deepgram.STT(),
            # llm=openai.LLM(model="gpt-4o-mini"),
            # tts=cartesia.TTS(),
        )
    )

    room_input = RoomInput(ctx.room, options=RoomInputOptions(audio_sample_rate=24000))
    ds_output = DataStreamOutput(ctx.room, destination_identity=AVATAR_IDENTITY)

    # wait for the participant to join the room and the avatar worker to connect
    await room_input.start(agent)
    await launch_avatar_worker(ctx, avatar_dispatcher_url, AVATAR_IDENTITY)

    # connect the output audio to the agent
    # agent.output.audio = ds_output.audio

    # or connect the output audio with transcription sync
    text_sink = RoomTranscriptEventSink(ctx.room, participant=AVATAR_IDENTITY)
    text_sync = TextSynchronizer(ds_output.audio, text_sink)
    agent.output.text = text_sync.text_sink
    agent.output.audio = text_sync.audio_sink

    await agent.start()

    @agent.output.audio.on("playback_finished")
    def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
        logger.info(
            "playback_finished",
            extra={
                "playback_position": ev.playback_position,
                "interrupted": ev.interrupted,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avatar-url", type=str, default="http://localhost:8089/launch"
    )
    args, remaining_args = parser.parse_known_args()
    print(sys.argv, remaining_args)
    sys.argv = sys.argv[:1] + remaining_args

    # WorkerType.ROOM is the default worker type which will create an agent for every room.
    # You can also use WorkerType.PUBLISHER to create a single agent for all participants that publish a track.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=partial(entrypoint, avatar_dispatcher_url=args.avatar_url),
            worker_type=WorkerType.ROOM,
        )
    )
