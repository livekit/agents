"""Meeting audio ingestion and relay streaming for external video meetings."""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import Callable

import aiohttp

from livekit import rtc
from livekit.agents.voice.io import AudioInput

logger = logging.getLogger(__name__)

_HEADER = struct.Struct("<IB")
_DEFAULT_STT_RATE = 16000
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
    return rtc.AudioFrame(
        data=pcm,
        sample_rate=sample_rate or _DEFAULT_STT_RATE,
        num_channels=channels,
        samples_per_channel=samples_per_channel,
    )


class MeetingAudioInput(AudioInput):
    """AudioInput that feeds mixed external meeting audio into AgentSession STT."""

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


async def stream_meeting_relay(
    websocket_url: str,
    audio_sink: Callable[[bytes], None],
    *,
    stop: asyncio.Event,
    chat_sink: Callable[[str], None] | None = None,
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
        reconnect_delay_s: Initial reconnect delay in seconds.
        max_reconnect_delay_s: Maximum reconnect delay in seconds.
    """
    backoff_time = reconnect_delay_s
    while not stop.is_set():
        try:
            async with (
                aiohttp.ClientSession() as http,
                http.ws_connect(websocket_url, heartbeat=20.0) as ws,
            ):
                backoff_time = reconnect_delay_s
                audio_frames = 0
                chat_messages = 0
                logger.debug("connected to meeting relay")
                async for msg in ws:
                    if stop.is_set():
                        break
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        audio_sink(msg.data)
                        audio_frames += 1
                        if audio_frames == 1:
                            logger.debug("received first pcm audio frame in meeting relay")
                    elif msg.type == aiohttp.WSMsgType.TEXT and chat_sink is not None:
                        chat_sink(msg.data)
                        chat_messages += 1
                        if chat_messages == 1:
                            logger.debug("received first chat message in meeting relay")
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "meeting relay disconnected; retrying in %ss",
                backoff_time,
                exc_info=True,
            )
        if not stop.is_set():
            await asyncio.sleep(backoff_time)
            backoff_time = min(backoff_time * 2, max_reconnect_delay_s)


async def stream_meeting_audio(
    websocket_url: str,
    sink: Callable[[bytes], None],
    *,
    stop: asyncio.Event,
    reconnect_delay_s: float = _INITIAL_RECONNECT_DELAY_S,
    max_reconnect_delay_s: float = _MAX_RECONNECT_DELAY_S,
) -> None:
    """Stream meeting audio from the LemonSlice meeting relay.

    Args:
        websocket_url: WebSocket URL returned by join_meeting.
        sink: Callback invoked with each binary PCM payload.
        stop: Event that signals the relay loop to exit.
        reconnect_delay_s: Initial reconnect delay in seconds.
        max_reconnect_delay_s: Maximum reconnect delay in seconds.
    """
    await stream_meeting_relay(
        websocket_url,
        sink,
        stop=stop,
        reconnect_delay_s=reconnect_delay_s,
        max_reconnect_delay_s=max_reconnect_delay_s,
    )
