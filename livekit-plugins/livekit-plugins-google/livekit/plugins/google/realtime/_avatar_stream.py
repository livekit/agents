from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import TypeVar

import av
import av.container

from livekit import rtc
from livekit.agents.utils.codecs.decoder import StreamBuffer

from ..log import logger

AVATAR_AUDIO_SAMPLE_RATE = 24000
AVATAR_AUDIO_CHANNELS = 1

_FrameT = TypeVar("_FrameT", rtc.VideoFrame, rtc.AudioFrame)


class AvatarStreamDecoder:
    """Decodes the fragmented MP4 stream that Gemini Live sends in avatar mode.

    One decoder covers one Live connection: the server sends a single init segment when
    the session opens, then a `moof`/`mdat` pair per video frame and per audio chunk.
    Decoding runs on a dedicated thread, and frames are handed back to the event loop.
    """

    def __init__(
        self,
        *,
        on_video: Callable[[rtc.VideoFrame], None],
        on_audio: Callable[[rtc.AudioFrame], None],
    ) -> None:
        self._on_video = on_video
        self._on_audio = on_audio
        self._loop = asyncio.get_running_loop()
        self._buffer = StreamBuffer()
        self._lock = threading.Lock()
        self._written = 0
        self._read = 0
        self._drain_waiters: list[tuple[int, Callable[[], None]]] = []
        self._closed = False
        self._thread = threading.Thread(
            target=self._decode_loop, name="gemini-avatar-decoder", daemon=True
        )
        self._thread.start()

    def push(self, data: bytes) -> None:
        if not self._closed:
            with self._lock:
                self._written += len(data)
            self._buffer.write(data)

    def call_when_drained(self, callback: Callable[[], None]) -> None:
        """Run `callback` on the event loop once every frame pushed so far is dispatched."""
        if self._closed:
            callback()
            return
        with self._lock:
            self._drain_waiters.append((self._written, callback))

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer.close()
        await asyncio.to_thread(self._thread.join)
        with self._lock:
            waiters, self._drain_waiters = self._drain_waiters, []
        for _, callback in waiters:
            callback()

    def read(self, size: int = -1) -> bytes:
        # the demuxer reads again only after it has yielded every packet it already read
        with self._lock:
            ready = [cb for target, cb in self._drain_waiters if target <= self._read]
            self._drain_waiters = [w for w in self._drain_waiters if w[0] > self._read]
        for callback in ready:
            self._schedule(callback)
        data = self._buffer.read(size)
        with self._lock:
            self._read += len(data)
        return data

    def _schedule(self, callback: Callable[[], None]) -> None:
        try:
            self._loop.call_soon_threadsafe(callback)
        except RuntimeError:
            self._closed = True

    def _dispatch(self, callback: Callable[[_FrameT], None], frame: _FrameT) -> None:
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(callback, frame)
        except RuntimeError:
            # the event loop is closed
            self._closed = True

    def _decode_loop(self) -> None:
        container: av.container.InputContainer | None = None
        try:
            # default probing holds back about the first second of the stream, which is
            # idle frames; a smaller probe size loses the first keyframe instead
            container = av.open(self, mode="r", format="mp4", buffer_size=4096)
            streams = [*container.streams.video[:1], *container.streams.audio[:1]]
            for stream in container.streams.video[:1]:
                # frame threading adds one frame of latency per thread
                stream.codec_context.thread_type = "SLICE"

            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=AVATAR_AUDIO_SAMPLE_RATE
            )
            for packet in container.demux(*streams):
                if self._closed:
                    break
                for frame in packet.decode():
                    if isinstance(frame, av.VideoFrame):
                        self._dispatch(
                            self._on_video,
                            rtc.VideoFrame(
                                frame.width,
                                frame.height,
                                rtc.VideoBufferType.I420,
                                frame.to_ndarray(format="yuv420p").tobytes(),
                            ),
                        )
                    elif isinstance(frame, av.AudioFrame):
                        for resampled in resampler.resample(frame):
                            self._dispatch(
                                self._on_audio,
                                rtc.AudioFrame(
                                    data=resampled.to_ndarray().tobytes(),
                                    sample_rate=AVATAR_AUDIO_SAMPLE_RATE,
                                    num_channels=AVATAR_AUDIO_CHANNELS,
                                    samples_per_channel=resampled.samples,
                                ),
                            )
        except Exception:
            if not self._closed:
                logger.exception("failed to decode the Gemini avatar stream")
        finally:
            if container is not None:
                container.close()
