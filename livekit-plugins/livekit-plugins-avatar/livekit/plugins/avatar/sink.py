from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from livekit import rtc
from livekit.agents import utils
from livekit.agents.pipeline.io import AudioSink
from livekit.rtc import data_stream
from pydantic import BaseModel

from .protocol import AvatarProtocol, PlaybackFinishedEvent, RPCProtocol
from .transport import AudioStreamFrame, TransportMode


@dataclass
class AvatarConfig:
    """Configuration for the avatar plugin"""

    worker_identity: str  # Identity of the remote avatar worker
    mime_type: str = "audio/raw"  # Mime type for audio stream
    chunk_size: int = 15_000  # Max size of each audio chunk


class AudioTransport(ABC):
    """Abstract interface for audio transport protocols"""

    @abstractmethod
    async def start_stream(self, stream_id: str) -> None:
        """Start a new audio stream"""
        pass

    @abstractmethod
    async def write_chunk(self, chunk: bytes) -> None:
        """Write audio chunk to current stream"""
        pass

    @abstractmethod
    async def end_stream(self) -> None:
        """End current audio stream"""
        pass


class DataStreamTransport(AudioTransport):
    """LiveKit DataStream implementation of audio transport"""

    def __init__(self, local_participant: rtc.LocalParticipant, config: AvatarConfig):
        self._local = local_participant
        self._config = config
        self._stream_writer: Optional[data_stream.FileStreamWriter] = None

    async def start_stream(self, stream_id: str) -> None:
        self._stream_writer = await self._local.send_file(
            file_name=f"{stream_id}.raw",
            mime_type=self._config.mime_type,
            destination_identities=[self._config.worker_identity],
        )

    async def write_chunk(self, chunk: bytes) -> None:
        if self._stream_writer:
            await self._stream_writer.write(chunk)

    async def end_stream(self) -> None:
        if self._stream_writer:
            await self._stream_writer.close()
            self._stream_writer = None


class AvatarAudioSink(AudioSink):
    """
    AudioSink implementation that streams audio to a remote avatar worker.
    Captures audio from the agent and streams it to a remote worker for avatar generation.
    """

    def __init__(
        self,
        transport: AudioTransport,
        protocol: AvatarProtocol,
        sample_rate: int = 48000,
        num_channels: int = 2,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        self._transport = transport
        self._protocol = protocol
        self._num_channels = num_channels
        self._current_stream_id: Optional[str] = None
        self._protocol.on_playback_finished = self._on_playback_finished

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture audio frame and stream to avatar worker"""
        await super().capture_frame(frame)

        if self._current_stream_id is None:
            # Generate new stream ID and start transport if needed
            self._current_stream_id = utils.shortuuid("audio_")
            await self._transport.start(TransportMode.SEND)

        # Send frame
        stream_frame = AudioStreamFrame(
            stream_id=self._current_stream_id,
            frame=frame,
        )
        await self._transport.send(stream_frame)

    def flush(self) -> None:
        """Mark end of current audio segment"""
        super().flush()
        if self._current_stream_id:
            await self._transport.close()
            self._current_stream_id = None

    def clear_buffer(self) -> None:
        """Stop current stream immediately"""
        if self._current_stream_id:
            await self._protocol.interrupt_playback(self._current_stream_id)
        self.flush()

    async def _on_playback_finished(self, event: PlaybackFinishedEvent) -> None:
        """Handle playback finished event from worker"""
        if event.stream_id == self._current_stream_id:
            self.on_playback_finished(
                playback_position=event.playback_position, interrupted=event.interrupted
            )


# Factory function to create DataStream-based sink
def create_datastream_sink(
    local_participant: rtc.LocalParticipant,
    config: AvatarConfig,
) -> AvatarAudioSink:
    """Create an AvatarAudioSink that uses LiveKit DataStream for transport"""
    transport = DataStreamTransport(local_participant, config)
    protocol = RPCProtocol(local_participant, config.worker_identity)
    return AvatarAudioSink(transport, protocol)
