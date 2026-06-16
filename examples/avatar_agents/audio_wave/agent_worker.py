import asyncio
import logging
import os
from dataclasses import asdict, dataclass

import httpx
from dotenv import load_dotenv

from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    cli,
    get_job_context,
    inference,
)
from livekit.agents.voice.avatar import AvatarSession, DataStreamAudioOutput
from livekit.agents.voice.io import PlaybackFinishedEvent, PlaybackStartedEvent
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

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


class CustomAvatarSession(AvatarSession):
    """Minimal avatar plugin backed by the example avatar dispatcher.

    Subclasses the base :class:`AvatarSession` so we get the join-wait, metrics
    and teardown for free — :meth:`AvatarSession.aclose` already removes the
    avatar participant from the room. We only add the dispatcher handshake and
    route the agent audio with :meth:`replace_audio_tail`.
    """

    def __init__(self, *, avatar_dispatcher_url: str, avatar_identity: str) -> None:
        super().__init__()
        self._avatar_dispatcher_url = avatar_dispatcher_url
        self._avatar_identity = avatar_identity

    @property
    def avatar_identity(self) -> str:
        return self._avatar_identity

    @property
    def provider(self) -> str:
        return "example-datastream-avatar"

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        await super().start(agent_session, room)

        # create a token for the avatar to join the room under our identity
        token = (
            api.AccessToken()
            .with_identity(self._avatar_identity)
            .with_name("Avatar Runner")
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            .with_kind("agent")
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: room.local_participant.identity})
            .to_jwt()
        )

        logger.info(
            f"sending connection info to avatar dispatcher {self._avatar_dispatcher_url}",
            extra={"identity": self._avatar_identity},
        )
        connection_info = AvatarConnectionInfo(
            room_name=room.name, url=get_job_context()._info.url, token=token
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(self._avatar_dispatcher_url, json=asdict(connection_info))
            response.raise_for_status()
        logger.info("avatar handshake completed", extra={"identity": self._avatar_identity})

        # route the agent audio to this avatar. replace_audio_tail swaps only the
        # bottom sink, keeping the TranscriptSynchronizer / recorder wrappers (and any
        # event listeners attached to session.output.audio) intact across hot swaps
        agent_session.output.replace_audio_tail(
            DataStreamAudioOutput(
                room,
                destination_identity=self._avatar_identity,
                # (optional) wait for the avatar to publish video track before generating a reply
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                # the example avatar_runner uses AvatarRunner which sends lk.playback_started
                wait_playback_start=True,
            )
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = Agent(instructions="Talk to me!")
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemini-2.5-flash"),
        tts=inference.TTS("cartesia/sonic-3"),
        resume_false_interruption=False,
    )

    avatar = CustomAvatarSession(
        avatar_dispatcher_url=AVATAR_DISPATCHER_URL,
        avatar_identity=AVATAR_IDENTITY,
    )
    await avatar.start(session, ctx.room)

    # start agent with room input and room text output
    await session.start(agent=agent, room=ctx.room)

    swap_lock = asyncio.Lock()

    @ctx.room.local_participant.register_rpc_method("swap_avatar")
    async def swap_avatar(data: rtc.RpcInvocationData) -> str:
        """RPC handler: tear down the current avatar and launch a fresh one.

        Trigger from a client with:
            room.local_participant.perform_rpc(
                destination_identity=<agent_identity>, method="swap_avatar", payload=""
            )
        """
        nonlocal avatar
        async with swap_lock:
            logger.info("swapping avatar")
            # remove the current avatar first; we reuse the same identity, so the room
            # can't hold both at once
            await avatar.aclose()

            avatar = CustomAvatarSession(
                avatar_dispatcher_url=AVATAR_DISPATCHER_URL,
                avatar_identity=AVATAR_IDENTITY,
            )
            # start() routes audio to the new avatar via replace_audio_tail; frames are
            # buffered until it publishes its video track, so playback resumes seamlessly
            await avatar.start(session, ctx.room)
            await avatar.wait_for_join()

            logger.info("avatar swapped")
            return "ok"

    # these listeners are attached to the top of the audio chain, which replace_audio_tail
    # leaves untouched, so they keep firing across avatar swaps
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

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message" and ev.item.role == "assistant":
            logger.info(
                "agent response metrics",
                extra={"metrics": ev.item.metrics},
            )

    await session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
