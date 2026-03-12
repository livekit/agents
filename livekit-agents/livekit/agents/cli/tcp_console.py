from __future__ import annotations

import asyncio
import queue as stdlib_queue
import time

from livekit import rtc
from livekit.protocol.agent_pb import agent_session as agent_pb

from ..log import logger
from ..voice import io
from ..voice.remote_session import TcpSessionTransport

WIRE_SAMPLE_RATE = 48000
AGENT_SAMPLE_RATE = 24000

_SENTINEL = object()


class TcpAudioInput(io.AudioInput):
    """Audio input backed by a stdlib queue.

    We use a stdlib queue (not an asyncio Chan) because the producer
    (push_frame, called from the main event loop) and the consumer
    (__anext__, called from the job-executor thread's event loop) live
    on *different* asyncio loops when JobExecutorType.THREAD is used.
    A stdlib queue + run_in_executor is the only safe bridge.
    """

    def __init__(self) -> None:
        super().__init__(label="TCP Console")
        self._queue: stdlib_queue.Queue[rtc.AudioFrame | object] = stdlib_queue.Queue()
        self._resampler = rtc.AudioResampler(
            input_rate=WIRE_SAMPLE_RATE,
            output_rate=AGENT_SAMPLE_RATE,
            num_channels=1,
        )
        self._closed = False

    def push_frame(self, frame: agent_pb.SessionAudioFrame) -> None:
        if self._closed:
            return
        audio_frame = rtc.AudioFrame(
            data=frame.data,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )
        resampled = self._resampler.push(audio_frame)
        for rf in resampled:
            self._queue.put_nowait(rf)

    def close(self) -> None:
        """Unblock any waiting consumer and mark as closed."""
        self._closed = True
        self._queue.put_nowait(_SENTINEL)

    async def __anext__(self) -> rtc.AudioFrame:
        loop = asyncio.get_running_loop()
        try:
            item = await loop.run_in_executor(None, self._queue.get)
        except RuntimeError:
            # Executor shut down — treat as end of stream.
            raise StopAsyncIteration
        if item is _SENTINEL:
            raise StopAsyncIteration
        return item


class TcpAudioOutput(io.AudioOutput):
    def __init__(self, transport: TcpSessionTransport) -> None:
        super().__init__(
            label="TCP Console",
            next_in_chain=None,
            sample_rate=AGENT_SAMPLE_RATE,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self._transport = transport
        self._resampler = rtc.AudioResampler(
            input_rate=AGENT_SAMPLE_RATE,
            output_rate=WIRE_SAMPLE_RATE,
            num_channels=1,
        )

        self._pushed_duration: float = 0.0
        self._capture_start: float = 0.0
        self._flush_task: asyncio.Task[None] | None = None
        self._playout_done = asyncio.Event()
        self._interrupted_ev = asyncio.Event()
        self._agent_loop: asyncio.AbstractEventLoop | None = None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

        if self._agent_loop is None:
            self._agent_loop = asyncio.get_running_loop()

        if self._flush_task and not self._flush_task.done():
            logger.error("capture_frame called while previous flush is in progress")
            await self._flush_task

        if not self._pushed_duration:
            self._capture_start = time.monotonic()
            self.on_playback_started(created_at=time.time())

        self._pushed_duration += frame.duration

        resampled = self._resampler.push(frame)
        for rf in resampled:
            audio_frame = agent_pb.SessionAudioFrame(
                data=bytes(rf.data),
                sample_rate=WIRE_SAMPLE_RATE,
                num_channels=rf.num_channels,
                samples_per_channel=rf.samples_per_channel,
            )
            msg = agent_pb.AgentSessionMessage(audio_output=audio_frame)
            self._transport.send_message_threadsafe(msg)

    def flush(self) -> None:
        super().flush()
        msg = agent_pb.AgentSessionMessage(
            audio_playback_flush=agent_pb.SessionAudioPlaybackFlush()
        )
        self._transport.send_message_threadsafe(msg)

        if self._pushed_duration:
            if self._flush_task and not self._flush_task.done():
                logger.error("flush called while previous flush is in progress")
                self._flush_task.cancel()

            self._playout_done.clear()
            self._interrupted_ev.clear()
            self._flush_task = asyncio.create_task(self._wait_for_playout())

    def clear_buffer(self) -> None:
        msg = agent_pb.AgentSessionMessage(
            audio_playback_clear=agent_pb.SessionAudioPlaybackClear()
        )
        self._transport.send_message_threadsafe(msg)

        if self._pushed_duration:
            self._interrupted_ev.set()

    def notify_playout_finished(self) -> None:
        if self._agent_loop is not None:
            self._agent_loop.call_soon_threadsafe(self._playout_done.set)
        else:
            self._playout_done.set()

    async def _wait_for_playout(self) -> None:
        wait_done = asyncio.create_task(self._playout_done.wait())
        wait_interrupt = asyncio.create_task(self._interrupted_ev.wait())
        try:
            await asyncio.wait(
                [wait_done, wait_interrupt],
                return_when=asyncio.FIRST_COMPLETED,
            )
            interrupted = wait_interrupt.done() and not wait_done.done()
        finally:
            wait_done.cancel()
            wait_interrupt.cancel()

        if interrupted:
            played = time.monotonic() - self._capture_start
            played = min(max(0, played), self._pushed_duration)
        else:
            played = self._pushed_duration

        self.on_playback_finished(playback_position=played, interrupted=interrupted)

        self._pushed_duration = 0.0
        self._interrupted_ev.clear()
