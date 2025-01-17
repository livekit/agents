from __future__ import annotations

import asyncio
from typing import Dict, Optional

from livekit import rtc
from livekit.rtc import data_stream

from .protocol import AvatarProtocol, PlaybackFinishedEvent, RPCProtocol
from .transport import AudioTransport, AudioFrameMetadata


class AvatarWorker:
    """Worker that generates synchronized avatar video based on received audio"""

    def __init__(
        self,
        room: rtc.Room,
        transport: AudioTransport,
        protocol: AvatarProtocol,
        video_width: int = 1280,
        video_height: int = 720,
        video_fps: float = 30.0,
    ) -> None:
        self._room = room
        self._transport = transport
        self._protocol = protocol
        
        self._video_source = rtc.VideoSource(width=video_width, height=video_height)
        self._audio_source = rtc.AudioSource(sample_rate=48000, num_channels=2)
        self._av_sync = rtc.AVSynchronizer(
            audio_source=self._audio_source,
            video_source=self._video_source,
            video_fps=video_fps,
        )
        
        self._active_streams: Dict[str, asyncio.Task] = {}
        
        # Register RPC handlers
        room.local_participant.register_rpc_method(
            RPCProtocol.RPC_INTERRUPT, self._handle_interrupt
        )

    async def _handle_interrupt(self, data: rtc.RpcInvocationData) -> None:
        """Handle interrupt RPC from sink"""
        stream_id = data.payload
        if stream_id in self._active_streams:
            # Cancel stream processing
            self._active_streams[stream_id].cancel()
            # Notify sink about interrupted playback
            await self._protocol.notify_playback_finished(
                stream_id=stream_id,
                playback_position=self._audio_source.queued_duration,
                interrupted=True,
            )

    async def start(self) -> None:
        """Start processing audio streams"""
        try:
            await self._transport.start(data_stream.TransportMode.RECEIVE)
            async for stream_frame in self._transport.receive():
                if stream_frame.stream_id not in self._active_streams:
                    # Start new stream processing
                    task = asyncio.create_task(
                        self._process_stream(stream_frame.stream_id)
                    )
                    self._active_streams[stream_frame.stream_id] = task
                
                # Generate and sync frames
                video_frame = self._generate_video_frame(stream_frame.frame)
                await self._av_sync.push(stream_frame.frame)
                await self._av_sync.push(video_frame)
        except asyncio.CancelledError:
            # Cancel all active streams
            for task in self._active_streams.values():
                task.cancel()
            await self._transport.close()
            raise

    async def _process_stream(self, stream_id: str) -> None:
        """Process a single audio stream"""
        try:
            # Process until stream ends or interrupted
            await asyncio.Future()  # Wait forever until cancelled
        except asyncio.CancelledError:
            # Stream interrupted
            await self._protocol.notify_playback_finished(
                stream_id=stream_id,
                playback_position=self._audio_source.queued_duration,
                interrupted=True,
            )
        finally:
            if stream_id in self._active_streams:
                del self._active_streams[stream_id]

    def _generate_video_frame(self, audio_frame: rtc.AudioFrame) -> rtc.VideoFrame:
        """Generate avatar video frame based on audio frame (to be implemented)"""
        pass
