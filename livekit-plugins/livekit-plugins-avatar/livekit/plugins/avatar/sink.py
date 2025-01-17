import asyncio
import json
from typing import AsyncIterator, Literal, Optional

from livekit import rtc
from livekit.agents.pipeline.io import AudioSink, PlaybackFinishedEvent

from .control import AvatarPlaybackControl, RPCPlaybackControl

# Participant attribute keys
DATASTREAM_AUDIO_SENDER = "datastream_audio.sender"
DATASTREAM_AUDIO_WORKER = "datastream_audio.worker"


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


class DataStreamAudioSink(AudioSink):
    """
    AudioSink implementation that streams audio to a remote avatar worker using LiveKit DataStream.
    Sends audio frames to a worker participant identified by the DATASTREAM_AUDIO_WORKER attribute.
    """

    def __init__(self, room: rtc.Room):
        super().__init__()
        self._room = room
        self._remote_participant: Optional[rtc.RemoteParticipant] = None
        self._stream_writer: Optional[rtc.FileStreamWriter] = None
        self._control: Optional[AvatarPlaybackControl] = None

    async def start(self) -> None:
        """Wait for worker participant to join and start streaming"""
        # mark self as sender
        await self._room.local_participant.set_attributes(
            {DATASTREAM_AUDIO_SENDER: "true"}
        )
        self._remote_participant = await wait_for_participant(
            self._room, attribute=DATASTREAM_AUDIO_WORKER
        )

        # setup control channel
        self._control = RPCPlaybackControl(
            self._room.local_participant,
            self._remote_participant.identity,
        )

        self._room.local_participant.register_rpc_method(
            RPCPlaybackControl.RPC_PLAYBACK_FINISHED,
            self._handle_playback_finished,
        )

    async def _handle_playback_finished(self, data: rtc.RpcInvocationData) -> None:
        """Handle playback finished RPC from worker"""
        event = PlaybackFinishedEvent(**json.loads(data.payload))
        self.on_playback_finished(
            playback_position=event.playback_position, interrupted=event.interrupted
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture and stream audio frame to remote worker"""
        if not self._remote_participant:
            raise RuntimeError("Worker participant not found")

        await super().capture_frame(frame)

        if not self._stream_writer:
            # Start new stream if needed
            self._stream_writer = await self._room.local_participant.send_file(
                file_name="audio_stream",
                mime_type="audio/raw",
                destination_identities=[self._remote_participant.identity],
            )

        # Pack frame metadata and data
        metadata = {
            "sample_rate": frame.sample_rate,
            "num_channels": frame.num_channels,
            "samples_per_channel": frame.samples_per_channel,
        }
        # TODO: Pack metadata + frame.data into binary format and send
        # await self._stream_writer.write(packed_data)

    def flush(self) -> None:
        """Mark end of current audio segment"""
        super().flush()
        if self._stream_writer:
            # Send end of segment marker
            metadata = {"type": "end_of_segment"}
            # TODO: Pack metadata into binary format and send
            # await self._stream_writer.write(packed_data)

    def clear_buffer(self) -> None:
        """Stop current stream immediately"""
        if not self._control:
            raise RuntimeError("Control channel not initialized")

        # Interrupt remote playback
        asyncio.create_task(self._control.interrupt_playback())
        super().clear_buffer()

    @property
    def control(self) -> AvatarPlaybackControl:
        """Get the playback control"""
        if not self._control:
            raise RuntimeError("Control channel not initialized")
        return self._control


class DataStreamAudioReceiver(rtc.EventEmitter[Literal["playback_interrupted"]]):
    """
    Audio receiver that receives streamed audio from a sender participant using LiveKit DataStream.
    Used by the worker to receive audio frames from a sender identified by the DATASTREAM_AUDIO_SENDER attribute.
    """

    def __init__(self, room: rtc.Room):
        self._room = room
        self._sender_participant: Optional[rtc.RemoteParticipant] = None
        self._stream_reader: Optional[rtc.FileStreamReader] = None
        self._control: Optional[AvatarPlaybackControl] = None

    async def start(self) -> None:
        """
        Wait for sender participant to join and start receiving.
        Also marks this participant as a worker.
        """
        # mark self as worker
        await self._room.local_participant.set_attributes(
            {DATASTREAM_AUDIO_WORKER: "true"}
        )

        self._sender_participant = await wait_for_participant(
            self._room, attribute=DATASTREAM_AUDIO_SENDER
        )
        self._control = RPCPlaybackControl(
            self._room.local_participant,
            self._sender_participant.identity,
        )

        def handle_interrupt(data: rtc.RpcInvocationData) -> None:
            self.emit("playback_interrupted")

        self._room.local_participant.register_rpc_method(
            RPCPlaybackControl.RPC_INTERRUPT,
            handle_interrupt,
        )

        self._stream_reader = await self._room.local_participant.receive_file()

    async def receive(self) -> AsyncIterator[rtc.AudioFrame]:
        """Receive audio frames from sender"""
        if not self._stream_reader:
            raise RuntimeError("Receiver not started")

        if not self._sender_participant:
            raise RuntimeError("Sender participant not found")

        while True:
            # TODO: Read and unpack binary data
            # data = await self._stream_reader.read()
            # metadata, frame_data = unpack_data(data)

            # if metadata.get('type') == 'end_of_segment':
            #     break

            # frame = rtc.AudioFrame(
            #     data=frame_data,
            #     sample_rate=metadata['sample_rate'],
            #     num_channels=metadata['num_channels'],
            #     samples_per_channel=metadata['samples_per_channel']
            # )
            # yield frame
            pass

    async def close(self) -> None:
        """Close the audio receiver"""
        if self._stream_reader:
            await self._stream_reader.close()
            self._stream_reader = None

    @property
    def control(self) -> AvatarPlaybackControl:
        """Get the playback control"""
        if not self._control:
            raise RuntimeError("Control channel not initialized")
        return self._control
