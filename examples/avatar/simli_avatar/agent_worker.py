import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from functools import partial

import httpx
from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import AgentState, JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.pipeline import AgentTask, PipelineAgent
from livekit.agents.pipeline.datastream_io import DataStreamOutput
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from livekit.agents.pipeline.room_io import (
    ATTRIBUTE_PUBLISH_FOR,
    RoomInput,
    RoomInputOptions,
    RoomTranscriptEventSink,
)
from livekit.agents.pipeline.transcription import TextSynchronizer
from livekit.plugins import openai

logger = logging.getLogger("avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


AVATAR_IDENTITY = "avatar_worker"


@dataclass
class AvatarConnectionInfo:
    """Contains connection parameters required to launch an avatar worker"""
    room_name: str  # Target LiveKit room name
    url: str        # LiveKit server URL
    token: str      # Authentication token for joining the room


async def launch_avatar_worker(
    ctx: JobContext, avatar_dispatcher_url: str, avatar_identity: str
) -> None:
    """Orchestrate avatar worker launch through the dispatcher service.
    
    Handles:
    - Generating worker join token
    - Sending launch request to dispatcher
    - Waiting for worker to join the room
    """
    # Create limited-access token for the avatar worker
    agent_identity = ctx.room.local_participant.identity
    token = (
        api.AccessToken()
        .with_identity(avatar_identity)
        .with_name("Simli Avatar Runner")
        .with_grants(api.VideoGrants(room_join=True, room=ctx.room.name))
        .with_kind("agent")
        .with_attributes({ATTRIBUTE_PUBLISH_FOR: agent_identity})  # Special attribute
        .to_jwt()
    )

    # Send launch request to dispatcher service
    logger.info(f"Requesting avatar worker from {avatar_dispatcher_url}")
    connection_info = AvatarConnectionInfo(
        room_name=ctx.room.name, url=ctx._info.url, token=token
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(avatar_dispatcher_url, json=asdict(connection_info))
        response.raise_for_status()

    # Wait for avatar worker to join the room
    await ctx.wait_for_participant(
        identity=avatar_identity,
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    )
    logger.info("Avatar worker connection established")


async def entrypoint(ctx: JobContext, avatar_dispatcher_url: str):
    """Main entrypoint for the agent worker.
    
    Configures and manages:
    - LiveKit room connection
    - AI pipeline components
    - Avatar worker coordination
    - Audio/text synchronization
    """
    await ctx.connect()  # Connect to LiveKit room

    # Initialize AI pipeline components
    agent = PipelineAgent(
        task=AgentTask(
            instructions="Talk to me!",  # System prompt for LLM
            llm=openai.realtime.RealtimeModel(),  # Real-time LLM for streaming
            # stt=deepgram.STT(),  # Uncomment to add speech-to-text
            # tts=cartesia.TTS(),  # Uncomment to add text-to-speech
        )
    )
    
    # Track agent state changes
    @agent.on("agent_state_changed")
    def on_agent_state_changed(state: AgentState):
        logger.info("Agent state changed", extra={"state": state})

    # Configure room input/output
    room_input = RoomInput(ctx.room, options=RoomInputOptions(audio_sample_rate=24000))
    ds_output = DataStreamOutput(ctx.room, destination_identity=AVATAR_IDENTITY)

    # Start processing room input and launch avatar worker
    await room_input.start(agent)
    await launch_avatar_worker(ctx, avatar_dispatcher_url, AVATAR_IDENTITY)

    # Set up audio/text synchronization
    text_sink = RoomTranscriptEventSink(ctx.room, participant=AVATAR_IDENTITY)
    text_sync = TextSynchronizer(ds_output.audio, text_sink)
    
    # Connect pipeline outputs
    agent.output.text = text_sync.text_sink  # Send synchronized text
    agent.output.audio = text_sync.audio_sink  # Send synchronized audio

    # Start agent processing
    await agent.start()

    # Track audio playback completion
    @agent.output.audio.on("playback_finished")
    def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
        logger.info(
            "Audio playback completed",
            extra={
                "position": ev.playback_position,
                "interrupted": ev.interrupted
            }
        )


if __name__ == "__main__":
    # Command line interface configuration
    parser = argparse.ArgumentParser(
        description="LiveKit Avatar Agent Worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--avatar-url",
        type=str,
        default="http://localhost:8089/launch",
        help="Dispatcher service endpoint URL"
    )
    
    # Parse arguments and run LiveKit worker
    args, remaining_args = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_args  # Clean args for LiveKit CLI

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=partial(entrypoint, avatar_dispatcher_url=args.avatar_url),
            worker_type=WorkerType.ROOM,  # Create agent per room
        )
    )
