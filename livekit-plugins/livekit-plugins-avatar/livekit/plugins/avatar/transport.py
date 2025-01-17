from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

from livekit import rtc


class TransportMode(Enum):
    SEND = "send"
    RECEIVE = "receive"


@dataclass
class AudioStreamFrame:
    """A frame of audio data with its stream ID"""

    stream_id: str
    frame: rtc.AudioFrame

    @classmethod
    def from_wav_bytes(cls, wav_data: bytes, stream_id: str) -> "AudioStreamFrame":
        # read the wav data into an AudioFrame
        frame = rtc.AudioFrame()
        return cls(stream_id=stream_id, frame=frame)


class AudioTransport(ABC):
    """Abstract interface for audio transport protocols"""

    @abstractmethod
    async def start(self, mode: TransportMode) -> None:
        """
        Start the transport in either send or receive mode.
        Must be called before sending or receiving frames.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the transport and clean up resources"""
        pass

    @abstractmethod
    async def send(self, frame: AudioStreamFrame) -> None:
        """Send an audio frame. Transport must be in SEND mode."""
        pass

    @abstractmethod
    async def receive(self) -> AsyncIterator[AudioStreamFrame]:
        """
        Receive audio frames. Transport must be in RECEIVE mode.
        Yields AudioStreamFrame objects containing stream ID and audio frame.
        """
        pass


class DataStreamTransport(AudioTransport):
    """LiveKit DataStream implementation of audio transport"""

    def __init__(self, local_participant: rtc.LocalParticipant, remote_identity: str):
        self._local = local_participant
        self._remote_identity = remote_identity
        self._mode: Optional[TransportMode] = None

    async def start(self, mode: TransportMode) -> None:
        self._mode = mode
        if self._mode == TransportMode.SEND:
            self._stream_writer = await self._local.send_file(
                file_name=f"{self._remote_identity}.wav",
                mime_type="audio/wav",
                destination_identities=[self._remote_identity],
            )
        elif self._mode == TransportMode.RECEIVE:
            self._stream_reader = await self._local.receive_file()

    async def close(self) -> None:
        if self._stream_writer:
            await self._stream_writer.close()
            self._stream_writer = None
        self._mode = None

    async def send(self, frame: AudioStreamFrame) -> None:
        pass

    async def receive(self) -> AsyncIterator[AudioStreamFrame]:
        pass
