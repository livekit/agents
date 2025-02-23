import asyncio
import json
import time
from dataclasses import dataclass

import requests
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration, RTCRtpCodecParameters
from aiortc.mediastreams import MediaStreamTrack
from av import VideoFrame, AudioFrame
from av.audio.resampler import AudioResampler
import websockets
from livekit.agents.pipeline.avatar import MediaOptions
from livekit import rtc
import numpy as np
from typing import AsyncIterator
import warnings
import ssl

warnings.filterwarnings("ignore", message="Unverified HTTPS request")  # Temporary for testing

@dataclass
class SimliConfig:
    api_key: str
    face_id: str
    sync_audio: bool = True
    handle_silence: bool = True
    max_session_length: int = 600
    max_idle_time: int = 30

class SimliAvatar:
    """LiveKit-compatible Simli avatar integration."""
    
    def __init__(self, config: SimliConfig, media_options: MediaOptions):
        self.config = config
        self.media_options = media_options
        self.pc = None
        self.audio_receiver = None
        self.video_receiver = None
        self._running = False
        self.frame_count = 0

    async def connect(self):
        """Initialize connection to Simli servers."""
        response = requests.post(
            "https://api.simli.ai/startAudioToVideoSession",
            json={
                "apiKey": self.config.api_key,
                "faceId": self.config.face_id,
                "syncAudio": self.config.sync_audio,
                "handleSilence": self.config.handle_silence,
                "maxSessionLength": self.config.max_session_length,
                "maxIdleTime": self.config.max_idle_time
            },
            verify=False
        )
        response.raise_for_status()
        session_token = response.json()["session_token"]

        # Configure WebRTC
        self.pc = RTCPeerConnection(RTCConfiguration([
            RTCIceServer(urls=["stun:stun.l.google.com:19302"])
        ]))
        
        # Explicitly configure codecs to avoid RTX issues
        audio_transceiver = self.pc.addTransceiver("audio", direction="recvonly")
        video_transceiver = self.pc.addTransceiver("video", direction="recvonly")
        
        # Force VP8 codec and disable RTX
        video_transceiver.setCodecPreferences([
            RTCRtpCodecParameters(
                mimeType="video/VP8",
                clockRate=90000,
                payloadType=96,
                parameters={"apt": 96}  # Explicitly associate payload type
            )
        ])
        self.pc.on("track", self._handle_track)

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        async with websockets.connect(
            "wss://api.simli.ai/StartWebRTCSession",
            ssl=ssl.create_default_context() if False else None,
            server_hostname="api.simli.ai"
        ) as ws:
            await ws.send(json.dumps(self.pc.localDescription.__dict__))
            answer = json.loads(await ws.recv())
            await self.pc.setRemoteDescription(RTCSessionDescription(**answer))
            await ws.send(session_token)
            await ws.recv()  # Wait for START confirmation

        self._running = True

    def _handle_track(self, track: MediaStreamTrack):
        """Handle incoming media tracks from Simli."""
        if track.kind == "audio":
            self.audio_receiver = track
        elif track.kind == "video":
            self.video_receiver = track

    async def send_audio(self, frame: rtc.AudioFrame):
        """Send audio to Simli for processing."""
        # Convert LiveKit audio frame to Simli format
        raw_data = np.frombuffer(frame.data, dtype=np.int16)
        await self._send_audio_data(raw_data.tobytes())

    async def _send_audio_data(self, data: bytes):
        """Internal method to send audio via websocket."""
        async with websockets.connect(
            "wss://api.simli.ai/audio",
            ssl=ssl.create_default_context() if False else None,
            server_hostname="api.simli.ai"
        ) as ws:
            await ws.send(data)

    async def video_frames(self) -> AsyncIterator[rtc.VideoFrame]:
        """Yield video frames from Simli."""
        while self._running:
            frame = await self.video_receiver.recv()
            yield rtc.VideoFrame(
                width=frame.width,
                height=frame.height,
                type=rtc.VideoBufferType.I420,
                data=frame.to_ndarray().tobytes()
            )
            # log every 100 frames
            if self.frame_count % 100 == 0:
                print(f"Received frame {self.frame_count}")
            self.frame_count += 1

    async def stop(self):
        """Cleanup resources."""
        self._running = False
        if self.pc:
            await self.pc.close() 