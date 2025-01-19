import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Optional, Union

from livekit import rtc
from livekit.agents.pipeline import io as agent_io
from livekit.agents.pipeline.io import PlaybackFinishedEvent

logger = logging.getLogger(__name__)

AUDIO_SENDER_ATTR = "__livekit_avatar_audio_sender"
AUDIO_RECEIVER_ATTR = "__livekit_avatar_audio_receiver"
RPC_INTERRUPT_PLAYBACK = "__livekit_avatar_interrupt_playback"
RPC_PLAYBACK_FINISHED = "__livekit_avatar_playback_finished"


class AudioSink(agent_io.AudioSink):
    """
    AudioSink implementation that streams audio to a remote avatar worker using LiveKit DataStream.
    Sends audio frames to a worker participant identified by the AUDIO_RECEIVER_ATTR attribute.
    """

    def __init__(self, room: rtc.Room):
        super().__init__()
        self._room = room
        self._remote_participant: Optional[rtc.RemoteParticipant] = None

        self._stream_writer: Optional[rtc.FileStreamWriter] = None

    async def start(self) -> None:
        """Wait for worker participant to join and start streaming"""
        # mark self as sender
        await self._room.local_participant.set_attributes({AUDIO_SENDER_ATTR: "true"})
        self._remote_participant = await wait_for_participant(
            room=self._room, attribute=AUDIO_RECEIVER_ATTR
        )

        # playback finished handler
        def _handle_playback_finished(data: rtc.RpcInvocationData) -> None:
            event = PlaybackFinishedEvent(**json.loads(data.payload))
            self.on_playback_finished(
                playback_position=event.playback_position,
                interrupted=event.interrupted,
            )

        self._room.local_participant.register_rpc_method(
            RPC_PLAYBACK_FINISHED, _handle_playback_finished
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture and stream audio frame to remote worker"""
        if not self._remote_participant:
            raise RuntimeError("Worker participant not found")

        if not self._stream_writer:
            # start new stream
            # TODO: any better option to send the metadata?
            name = f"audio_{frame.sample_rate}_{frame.num_channels}"
            self._stream_writer = await self._room.local_participant.stream_file(
                file_name=name,
                mime_type="audio/raw",
                destination_identities=[self._remote_participant.identity],
            )
        await self._stream_writer.write(frame.data)

        await super().capture_frame(frame)

    async def flush(self) -> None:
        """Mark end of current audio segment"""
        assert self._stream_writer is not None

        # close the stream marking the end of the segment
        await self._stream_writer.aclose()
        self._stream_writer = None

        # TODO: should the flush and clear_buffer async?
        super().flush()

    async def clear_buffer(self) -> None:
        """Stop current stream immediately"""
        assert self._remote_participant is not None

        await self._room.local_participant.perform_rpc(
            destination_identity=self._remote_participant.identity,
            method=RPC_INTERRUPT_PLAYBACK,
            payload="",
        )


class AudioEndSentinel:
    pass


@dataclass
class _AudioReader:
    stream_reader: rtc.FileStreamReader
    sample_rate: int
    num_channels: int


class AudioReceiver(rtc.EventEmitter[Literal["interrupt_playback"]]):
    """
    Audio receiver that receives streamed audio from a sender participant using LiveKit DataStream.
    Used by the worker to receive audio frames from a sender identified by the DATASTREAM_AUDIO_SENDER attribute.
    """

    def __init__(self, room: rtc.Room):
        self._room = room
        self._remote_participant: Optional[rtc.RemoteParticipant] = None

        self._stream_readers: list[_AudioReader] = []
        self._stream_received: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        """
        Wait for sender participant to join and start receiving.
        Also marks this participant as a worker.
        """
        # mark self as worker
        await self._room.local_participant.set_attributes({AUDIO_RECEIVER_ATTR: "true"})

        self._remote_participant = await wait_for_participant(
            room=self._room, attribute=AUDIO_SENDER_ATTR
        )

        def _handle_interrupt(data: rtc.RpcInvocationData) -> None:
            self.emit("interrupt_playback")

        self._room.local_participant.register_rpc_method(
            RPC_INTERRUPT_PLAYBACK, _handle_interrupt
        )

        def _handle_stream_received(
            reader: rtc.FileStreamReader, remote_participant_id: str
        ) -> None:
            if remote_participant_id != self._remote_participant.identity:
                logger.warning(
                    "Received stream from unexpected participant",
                    extra={
                        "remote_participant_id": remote_participant_id,
                        "expected_participant_id": self._remote_participant.identity,
                    },
                )
                return

            file_name = reader.info.file_name
            _, sample_rate, num_channels = file_name.split("_")
            self._stream_readers.append(
                _AudioReader(reader, int(sample_rate), int(num_channels))
            )
            self._stream_received.set()

        self._room.on("file_stream_received", _handle_stream_received)

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

    async def stream(self) -> AsyncIterator[Union[rtc.AudioFrame, AudioEndSentinel]]:
        while True:
            await self._stream_received.wait()

            while self._stream_readers:
                reader = self._stream_readers.pop(0)
                async for data in reader.stream_reader:
                    yield rtc.AudioFrame(
                        data=data,
                        sample_rate=reader.sample_rate,
                        num_channels=reader.num_channels,
                    )
                yield AudioEndSentinel()

            self._stream_received.clear()


async def wait_for_participant(room: rtc.Room, attribute: str) -> rtc.RemoteParticipant:
    """Wait for a participant with the given attribute to join the room"""
    # Wait for participant to join
    future = asyncio.Future[rtc.RemoteParticipant]()

    def on_participant_join(participant: rtc.RemoteParticipant):
        if attribute in participant.attributes and not future.done():
            future.set_result(participant)

    room.on("participant_joined", on_participant_join)
    try:
        # check if participant already in room
        for participant in room.participants.values():
            if attribute in participant.attributes:
                return participant

        return await future
    finally:
        room.off("participant_joined", on_participant_join)
