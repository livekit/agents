"""Meeting audio ingestion and relay streaming for external video meetings."""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import Callable
from typing import Never

import numpy as np

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, APIError, utils
from livekit.agents.voice.avatar import AvatarWebSocket, WebSocketTarget
from livekit.agents.voice.io import AudioInput

logger = logging.getLogger(__name__)

_HEADER = struct.Struct("<IB")
_DEFAULT_STT_RATE = 16000
_STT_CHANNELS = 1
_INITIAL_RECONNECT_DELAY_S = 1.0
_MAX_RECONNECT_DELAY_S = 30.0


def _deserialize_frame(payload: bytes) -> rtc.AudioFrame | None:
    if len(payload) < _HEADER.size:
        return None
    sample_rate, channels = _HEADER.unpack_from(payload)
    channels = channels or 1
    pcm = payload[_HEADER.size :]
    samples_per_channel = len(pcm) // 2 // channels
    if samples_per_channel <= 0:
        return None

    frame_bytes = samples_per_channel * channels * 2
    pcm = pcm[:frame_bytes]

    return rtc.AudioFrame(
        data=pcm,
        sample_rate=sample_rate or _DEFAULT_STT_RATE,
        num_channels=channels,
        samples_per_channel=samples_per_channel,
    )


def _downmix_to_mono(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    """Downmix multi-channel PCM to mono for STT (parity with RoomIO)."""
    if frame.num_channels == 1:
        return frame

    data = np.array(frame.data, dtype=np.int16)
    mono = data.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
    return rtc.AudioFrame(
        data=mono.tobytes(),
        sample_rate=frame.sample_rate,
        num_channels=_STT_CHANNELS,
        samples_per_channel=frame.samples_per_channel,
    )


class MeetingAudioInput(AudioInput):
    """AudioInput that feeds mixed external meeting audio into VoiceSession STT."""

    def __init__(self, *, rate_out: int = _DEFAULT_STT_RATE, queue_size: int = 100) -> None:
        """Initialize the meeting audio input.

        Args:
            rate_out: Output sample rate in Hz for frames consumed by STT.
            queue_size: Maximum number of audio frames to buffer.
        """
        super().__init__(label="lemonslice-meeting-audio")
        self._loop = asyncio.get_running_loop()
        self._rate_out = rate_out
        self._queue: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue(maxsize=queue_size)
        self._resampler: rtc.AudioResampler | None = None
        self._resampler_in_rate: int | None = None

    def submit(self, payload: bytes) -> None:
        """Enqueue a serialized PCM frame from the meeting relay WebSocket."""
        try:
            self._loop.call_soon_threadsafe(self._push, payload)
        except RuntimeError:
            pass

    def _enqueue(self, frame: rtc.AudioFrame) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("meeting audio queue full; dropping frame")

    def _push(self, payload: bytes) -> None:
        frame = _deserialize_frame(payload)
        if frame is None:
            return
        frame = _downmix_to_mono(frame)
        for out in self._resample(frame):
            self._enqueue(out)

    def _resample(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        if frame.sample_rate == self._rate_out:
            return [frame]
        if self._resampler is None or self._resampler_in_rate != frame.sample_rate:
            self._resampler = rtc.AudioResampler(
                frame.sample_rate, self._rate_out, num_channels=frame.num_channels
            )
            self._resampler_in_rate = frame.sample_rate
        return self._resampler.push(frame)

    async def __anext__(self) -> rtc.AudioFrame:
        """Return the next resampled audio frame for STT consumption."""
        return await self._queue.get()


class _Relay(AvatarWebSocket[bytes | str, Never]):
    """The meeting relay's socket: PCM audio frames as binary, chat as JSON text; it is sent
    nothing."""

    ping_interval = 20.0

    def __init__(
        self,
        url: str,
        audio_sink: Callable[[bytes], None],
        chat_sink: Callable[[str], None] | None,
    ) -> None:
        super().__init__("LemonSlice meeting relay")
        self._url = url
        self._audio_sink = audio_sink
        self._chat_sink = chat_sink
        self._audio_frames = 0
        self._chat_messages = 0

    async def connect(self, conn_options: APIConnectOptions) -> WebSocketTarget:
        return WebSocketTarget(self._url)

    def on_message(self, message: bytes | str) -> None:
        if isinstance(message, bytes):
            self._audio_sink(message)
            self._audio_frames += 1
            if self._audio_frames == 1:
                logger.debug("received first pcm audio frame in meeting relay")
        elif self._chat_sink is not None:
            self._chat_sink(message)
            self._chat_messages += 1
            if self._chat_messages == 1:
                logger.debug("received first chat message in meeting relay")


async def stream_meeting_relay(
    websocket_url: str,
    audio_sink: Callable[[bytes], None],
    *,
    stop: asyncio.Event,
    chat_sink: Callable[[str], None] | None = None,
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    reconnect_delay_s: float = _INITIAL_RECONNECT_DELAY_S,
    max_reconnect_delay_s: float = _MAX_RECONNECT_DELAY_S,
) -> None:
    """Stream meeting audio and chat from the LemonSlice meeting relay.

    Reconnects with exponential backoff until ``stop`` is set.

    Args:
        websocket_url: WebSocket URL returned by join_meeting.
        audio_sink: Callback invoked with each binary PCM payload.
        stop: Event that signals the relay loop to exit.
        chat_sink: Optional callback invoked with each text chat JSON payload.
        conn_options: The connect limit of each connection.
        reconnect_delay_s: Initial reconnect delay in seconds.
        max_reconnect_delay_s: Maximum reconnect delay in seconds.
    """
    backoff_time = reconnect_delay_s

    async def relay(socket: _Relay) -> None:
        nonlocal backoff_time
        await socket.open(conn_options)
        backoff_time = reconnect_delay_s
        logger.debug("connected to meeting relay")
        await socket.run(conn_options)

    while not stop.is_set():
        socket = _Relay(websocket_url, audio_sink, chat_sink)
        try:
            async with utils.aio.TaskSet() as tasks:
                await tasks.first_completed(relay(socket), stop.wait())
        except APIError:
            logger.warning(
                "meeting relay disconnected; retrying in %ss", backoff_time, exc_info=True
            )
        finally:
            await socket.aclose()
        if not stop.is_set():
            await asyncio.sleep(backoff_time)
            backoff_time = min(backoff_time * 2, max_reconnect_delay_s)
