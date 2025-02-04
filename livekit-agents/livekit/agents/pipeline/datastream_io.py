import asyncio
import ctypes
import json
import logging
from dataclasses import asdict
from typing import AsyncIterator, Literal, Optional

from livekit import rtc

from .. import utils
from ..utils import aio
from .io import AudioSink, PlaybackFinishedEvent

logger = logging.getLogger(__name__)

RPC_CLEAR_BUFFER = "lk.clear_buffer"
RPC_PLAYBACK_FINISHED = "lk.playback_finished"
AUDIO_STREAM_TOPIC = "lk.audio_stream"


class DataStreamOutput:
    def __init__(self, room: rtc.Room, *, destination_identity: str):
        self._room = room
        self._destination_identity = destination_identity
        self._audio_sink = DataStreamAudioSink(
            room, destination_identity=destination_identity
        )

    @property
    def audio(self) -> "DataStreamAudioSink":
        return self._audio_sink


class DataStreamAudioSink(AudioSink):
    """
    AudioSink implementation that streams audio to a remote avatar worker using LiveKit DataStream.
    """

    def __init__(self, room: rtc.Room, *, destination_identity: str):
        super().__init__()
        self._room = room
        self._destination_identity = destination_identity
        self._stream_writer: Optional[rtc.ByteStreamWriter] = None
        self._pushed_duration: float = 0.0

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

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture and stream audio frame to remote worker"""
        await super().capture_frame(frame)

        if not self._stream_writer:
            self._stream_writer = await self._room.local_participant.stream_bytes(
                name=utils.shortuuid("AUDIO_"),
                topic=AUDIO_STREAM_TOPIC,
                destination_identities=[self._destination_identity],
                attributes={
                    "sample_rate": str(frame.sample_rate),
                    "num_channels": str(frame.num_channels),
                },
            )
            self._pushed_duration = 0.0
        await self._stream_writer.write(bytes(frame.data))
        self._pushed_duration += frame.duration

    def flush(self) -> None:
        """Mark end of current audio segment"""
        super().flush()
        if self._stream_writer is None:
            return

        # close the stream marking the end of the segment
        asyncio.create_task(self._stream_writer.aclose())
        self._stream_writer = None
        logger.info(f"pushed duration: {self._pushed_duration}")

    def clear_buffer(self) -> None:
        asyncio.create_task(
            self._room.local_participant.perform_rpc(
                destination_identity=self._destination_identity,
                method=RPC_CLEAR_BUFFER,
                payload="",
            )
        )


class AudioFlushSentinel:
    pass


class DataStreamAudioReceiver(rtc.EventEmitter[Literal["clear_buffer"]]):
    """
    Audio receiver that receives streamed audio from a sender participant using LiveKit DataStream.
    If the sender_identity is provided, subscribe to the specified participant. If not provided,
    subscribe to the first agent participant in the room.
    """

    def __init__(self, room: rtc.Room, *, sender_identity: Optional[str] = None):
        super().__init__()
        self._room = room
        self._sender_identity = sender_identity
        self._remote_participant: Optional[rtc.RemoteParticipant] = None

        self._stream_readers: list[rtc.ByteStreamReader] = []
        self._stream_reader_changed: asyncio.Event = asyncio.Event()

        self._data_ch = aio.Chan[rtc.AudioFrame | AudioFlushSentinel]()
        self._read_atask: Optional[asyncio.Task[None]] = None
        self._current_reader: Optional[rtc.ByteStreamReader] = None
        self._current_reader_cleared: bool = False

    async def start(self) -> None:
        # wait for the first agent participant to join
        self._remote_participant = await self._wait_for_participant(
            identity=self._sender_identity
        )

        def _handle_clear_buffer(data: rtc.RpcInvocationData) -> None:
            if self._current_reader:
                self._current_reader_cleared = True
            self.emit("clear_buffer")
            return "ok"

        self._room.local_participant.register_rpc_method(
            RPC_CLEAR_BUFFER, _handle_clear_buffer
        )

        def _handle_stream_received(
            reader: rtc.ByteStreamReader, remote_participant_id: str
        ) -> None:
            if remote_participant_id != self._remote_participant.identity:
                return

            self._stream_readers.append(reader)
            self._stream_reader_changed.set()

        self._room.register_byte_stream_handler(
            AUDIO_STREAM_TOPIC, _handle_stream_received
        )

        self._read_atask = asyncio.create_task(self._read_stream())

    async def notify_playback_finished(
        self, playback_position: int, interrupted: bool
    ) -> None:
        """Notify the sender that playback has finished"""
        assert self._remote_participant is not None
        event = PlaybackFinishedEvent(
            playback_position=playback_position, interrupted=interrupted
        )
        try:
            logger.info(
                f"notifying playback finished: {event.playback_position:.3f}s, "
                f"interrupted: {event.interrupted}"
            )
            await self._room.local_participant.perform_rpc(
                destination_identity=self._remote_participant.identity,
                method=RPC_PLAYBACK_FINISHED,
                payload=json.dumps(asdict(event)),
            )
        except Exception as e:
            logger.exception(f"error notifying playback finished: {e}")

    @property
    def data_ch(self) -> aio.Chan[rtc.AudioFrame | AudioFlushSentinel]:
        return self._data_ch

    @utils.log_exceptions(logger=logger)
    async def _read_stream(self) -> None:
        while not self._data_ch.closed:
            await self._stream_reader_changed.wait()

            while self._stream_readers and not self._data_ch.closed:
                self._current_reader = self._stream_readers.pop(0)
                sample_rate = int(self._current_reader.info.attributes["sample_rate"])
                num_channels = int(self._current_reader.info.attributes["num_channels"])
                async for data in self._current_reader:
                    if self._data_ch.closed:
                        return

                    if self._current_reader_cleared:
                        # ignore the rest data of the current reader if clear_buffer was called
                        continue

                    samples_per_channel = (
                        len(data) // num_channels // ctypes.sizeof(ctypes.c_int16)
                    )
                    frame = rtc.AudioFrame(
                        data=data,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                        samples_per_channel=samples_per_channel,
                    )
                    self._data_ch.send_nowait(frame)
                self._data_ch.send_nowait(AudioFlushSentinel())
                self._current_reader = None
                self._current_reader_cleared = False

            self._stream_reader_changed.clear()

    async def _wait_for_participant(
        self, identity: Optional[str] = None
    ) -> rtc.RemoteParticipant:
        """Wait for a participant to join the room and return it"""

        def _is_matching_participant(participant: rtc.RemoteParticipant) -> bool:
            if identity is not None and participant.identity != identity:
                return False
            return participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT

        for participant in self._room.remote_participants.values():
            if _is_matching_participant(participant):
                return participant

        fut = asyncio.Future[rtc.RemoteParticipant]()

        def _handle_participant_joined(participant: rtc.RemoteParticipant) -> None:
            if _is_matching_participant(participant):
                fut.set_result(participant)

        self._room.on("participant_joined", _handle_participant_joined)
        try:
            return await fut
        finally:
            self._room.off("participant_joined", _handle_participant_joined)

    async def __anext__(self) -> rtc.AudioFrame | AudioFlushSentinel:
        return await self._data_ch.recv()

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioFlushSentinel]:
        return self

    def close(self) -> None:
        self._data_ch.close()
