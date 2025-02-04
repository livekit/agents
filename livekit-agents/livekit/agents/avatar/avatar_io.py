import logging

import httpx
from livekit import api
from pydantic import BaseModel

from .. import JobContext
from ..pipeline.datastream_io import DataStreamOutput

logger = logging.getLogger(__name__)

DEFAULT_AVATAR_IDENTITY = "lk.avatar_worker"
RPC_INTERRUPT_PLAYBACK = "lk.interrupt_playback"
RPC_PLAYBACK_FINISHED = "lk.playback_finished"
AUDIO_STREAM_TOPIC = "lk.audio_stream"


class AvatarConnectionInfo(BaseModel):
    room_name: str
    url: str  # LiveKit server URL
    token: str  # Token for avatar worker to join


class AvatarOutput(DataStreamOutput):
    def __init__(
        self,
        ctx: JobContext,
        *,
        destination_identity: str = DEFAULT_AVATAR_IDENTITY,
        avatar_dispatcher_url: str = "http://localhost:8089/launch",
    ):
        super().__init__(ctx.room, destination_identity=destination_identity)
        self._ctx = ctx
        self._avatar_dispatcher_url = avatar_dispatcher_url

    async def start(self) -> None:
        """Wait for worker participant to join and start streaming"""
        # create a token for the avatar worker
        # TODO(long): do we need to set agent=True here? in playground if not the video track is not automatically displayed
        token = (
            api.AccessToken()
            .with_identity(self._destination_identity)
            .with_name("Avatar Worker")
            .with_grants(
                api.VideoGrants(room_join=True, room=self._room.name, agent=True)
            )
            .with_metadata("avatar_worker")
            .to_jwt()
        )

        logger.info(
            f"Sending connection info to avatar dispatcher {self._avatar_dispatcher_url}"
        )
        await self._launch_worker(
            AvatarConnectionInfo(
                room_name=self._room.name, url=self._ctx._info.url, token=token
            )
        )
        logger.info("Avatar worker connected")

        # wait for the remote participant to join
        await self._ctx.wait_for_participant(identity=self._destination_identity)

    async def _launch_worker(self, info: AvatarConnectionInfo) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._avatar_dispatcher_url, json=info.model_dump()
            )
            response.raise_for_status()
