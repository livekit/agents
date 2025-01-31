import asyncio
import ctypes
import json
import logging
from typing import AsyncIterator, Literal, Optional, Union

import httpx
from livekit import api, rtc
from livekit.agents import JobContext, utils
from livekit.agents.pipeline import io as agent_io
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_AVATAR_IDENTITY = "lk.avatar_worker"
RPC_INTERRUPT_PLAYBACK = "lk.interrupt_playback"
RPC_PLAYBACK_FINISHED = "lk.playback_finished"
AUDIO_STREAM_TOPIC = "lk.audio_stream"


class AvatarConnectionInfo(BaseModel):
    room_name: str
    url: str  # LiveKit server URL
    token: str  # Token for avatar worker to join


class AudioSink(agent_io.AudioSink):
    """
    AudioSink implementation that streams audio to a remote avatar worker using LiveKit DataStream.
    """

    def __init__(
        self,
        ctx: JobContext,
        avatar_identity: str = DEFAULT_AVATAR_IDENTITY,
        avatar_dispatcher_url: str = "http://localhost:8890/launch",
    ):
        super().__init__()
        self._ctx = ctx
        self._room = self._ctx.room
        self._avatar_identity = avatar_identity
        self._avatar_dispatcher_url = avatar_dispatcher_url
        self._remote_participant: Optional[rtc.RemoteParticipant] = None

        self._stream_writer: Optional[rtc.ByteStreamWriter] = None

    async def start(self) -> None:
        """Wait for worker participant to join and start streaming"""
        # create a token for the avatar worker
        # TODO(long): do we need to set agent=True here?
        # in playground if not the video track is not automatically displayed
        token = (
            api.AccessToken()
            .with_identity(self._avatar_identity)
            .with_name("Avatar Worker")
            .with_grants(api.VideoGrants(room_join=True, room=self._room.name, agent=True))
            .to_jwt()
        )

        logger.info(
            f"Sending connection info to avatar dispatcher {self._avatar_dispatcher_url}"
        )
        await self._handshake(
            AvatarConnectionInfo(
                room_name=self._room.name, url=self._ctx._info.url, token=token
            )
        )
        logger.info("Avatar worker connected")

        # playback finished handler
        def _handle_playback_finished(data: rtc.RpcInvocationData) -> None:
            event = PlaybackFinishedEvent(**json.loads(data.payload))
            self.on_playback_finished(
                playback_position=event.playback_position,
                interrupted=event.interrupted,
            )
            return "ok"

        self._room.local_participant.register_rpc_method(
            RPC_PLAYBACK_FINISHED, _handle_playback_finished
        )

        # wait for the remote participant to join
        self._remote_participant = await self._ctx.wait_for_participant(
            identity=self._avatar_identity
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture and stream audio frame to remote worker"""
        await super().capture_frame(frame)

        if not self._remote_participant:
            raise RuntimeError("Worker participant not found")

        if not self._stream_writer:
            self._stream_writer = await self._room.local_participant.stream_bytes(
                name=utils.shortuuid("AUDIO_"),
                topic=AUDIO_STREAM_TOPIC,
                destination_identities=[self._remote_participant.identity],
                extensions={  # TODO: use attributes instead
                    "sample_rate": str(frame.sample_rate),
                    "num_channels": str(frame.num_channels),
                },
            )
        await self._stream_writer.write(bytes(frame.data))

    def flush(self) -> None:
        """Mark end of current audio segment"""
        super().flush()
        if self._stream_writer is None:
            return

        # close the stream marking the end of the segment
        asyncio.create_task(self._stream_writer.aclose())
        self._stream_writer = None

    def clear_buffer(self) -> None:
        """Stop current stream immediately"""
        assert self._remote_participant is not None

        asyncio.create_task(
            self._room.local_participant.perform_rpc(
                destination_identity=self._remote_participant.identity,
                method=RPC_INTERRUPT_PLAYBACK,
                payload="",
            )
        )

    async def _handshake(self, connection_info: AvatarConnectionInfo) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._avatar_dispatcher_url,
                json=connection_info.model_dump(),
            )
            response.raise_for_status()


class AudioFlushSentinel:
    pass


class AudioReceiver(rtc.EventEmitter[Literal["interrupt_playback"]]):
    """
    Audio receiver that receives streamed audio from a sender participant using LiveKit DataStream.
    Used by the worker to receive audio frames from a sender identified by the DATASTREAM_AUDIO_SENDER attribute.
    """

    def __init__(self, room: rtc.Room):
        super().__init__()
        self._room = room
        self._remote_participant: Optional[rtc.RemoteParticipant] = None

        self._stream_readers: list[rtc.ByteStreamReader] = []
        self._stream_reader_changed: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        """
        Wait for sender participant to join and start receiving.
        Usage:
            audio_receiver = AudioReceiver(room)
            await audio_receiver.start()
            async for frame in audio_receiver.stream():
                if isinstance(frame, AudioFlushSentinel):
                    # segment completed
                    pass
                else:
                    # process frame
                    pass
        """

        self._remote_participant = await self._wait_for_agent()

        def _handle_interrupt(data: rtc.RpcInvocationData) -> None:
            self.emit("interrupt_playback")
            return "ok"

        self._room.local_participant.register_rpc_method(
            RPC_INTERRUPT_PLAYBACK, _handle_interrupt
        )

        def _handle_stream_received(
            reader: rtc.ByteStreamReader, remote_participant_id: str
        ) -> None:
            if remote_participant_id != self._remote_participant.identity:
                return

            self._stream_readers.append(reader)
            self._stream_reader_changed.set()

        self._room.set_byte_stream_handler(_handle_stream_received, AUDIO_STREAM_TOPIC)

    async def notify_playback_finished(
        self, playback_position: int, interrupted: bool
    ) -> None:
        """Notify worker that playback has finished"""
        assert self._remote_participant is not None

        await self._room.local_participant.perform_rpc(
            destination_identity=self._remote_participant.identity,
            method=RPC_PLAYBACK_FINISHED,
            payload=json.dumps(
                {"playback_position": playback_position, "interrupted": interrupted}
            ),
        )

    @utils.log_exceptions(logger=logger)
    async def stream(self) -> AsyncIterator[Union[rtc.AudioFrame, AudioFlushSentinel]]:
        while True:
            await self._stream_reader_changed.wait()

            while self._stream_readers:
                reader = self._stream_readers.pop(0)
                sample_rate = int(
                    reader.info["attributes"]["sample_rate"]
                )  # TODO: use dataclass
                num_channels = int(reader.info["attributes"]["num_channels"])
                async for data in reader:
                    # TODO: make sure the data is a multiple of num_channels * samples_per_channel
                    samples_per_channel = len(data) // num_channels // ctypes.sizeof(ctypes.c_int16)
                    frame = rtc.AudioFrame(
                        data=data,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                        samples_per_channel=samples_per_channel,
                    )
                    yield frame
                yield AudioFlushSentinel()

            self._stream_reader_changed.clear()

    async def _wait_for_agent(self) -> rtc.RemoteParticipant:
        for participant in self._room.remote_participants.values():
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                return participant

        fut = asyncio.Future[rtc.RemoteParticipant]()

        def _handle_participant_joined(participant: rtc.RemoteParticipant) -> None:
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                fut.set_result(participant)

        self._room.on("participant_joined", _handle_participant_joined)
        try:
            return await fut
        finally:
            self._room.off("participant_joined", _handle_participant_joined)
