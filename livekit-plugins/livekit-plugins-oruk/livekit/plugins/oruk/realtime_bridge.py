"""Explicit VAD-to-commit bridge; no transcript-text guessing for metadata."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterator

import numpy as np

from livekit import rtc
from livekit.agents import stt, utils, vad

from .realtime import RealtimeStream, RealtimeSTT

_VAD_BACKLOG_TIMEOUT = 1.0


async def _mono16k(audio: AsyncIterable[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
    sample_rate = 0
    resampler: rtc.AudioResampler | None = None

    def split(frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        data = memoryview(frame.data).cast("B")
        for i in range(0, len(data), 10_240):
            chunk = data[i : i + 10_240].tobytes()
            yield rtc.AudioFrame(chunk, 16000, 1, len(chunk) // 2)

    async for frame in audio:
        if sample_rate and sample_rate != frame.sample_rate:
            raise ValueError("Input sample rate must remain constant")
        if not sample_rate:
            sample_rate = frame.sample_rate
            if sample_rate != 16000:
                resampler = rtc.AudioResampler(sample_rate, 16000, num_channels=1)
        if frame.num_channels != 1:
            samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
            mono = np.rint(samples.astype(np.float32).mean(axis=1)).astype(np.int16)
            frame = rtc.AudioFrame(mono.tobytes(), sample_rate, 1, len(mono))
        frames = resampler.push(frame) if resampler else [frame]
        for converted in frames:
            for chunk in split(converted):
                yield chunk
    if resampler:
        for converted in resampler.flush():
            for chunk in split(converted):
                yield chunk


def _push_bytes(stream: RealtimeStream, data: bytes) -> None:
    for i in range(0, len(data), 10_240):
        chunk = data[i : i + 10_240]
        stream.push_frame(rtc.AudioFrame(chunk, 16000, 1, len(chunk) // 2))


async def vad_stream_node(
    recognizer: RealtimeSTT,
    audio: AsyncIterable[rtc.AudioFrame],
    *,
    detector: vad.VAD,
) -> AsyncIterator[stt.SpeechEvent]:
    """Stream while speaking and commit exactly at each observed VAD end.

    This bridge targets the checked-in Silero VAD event contract: each processed
    PCM window is emitted in INFERENCE_DONE before its START/END boundary. Other
    VAD implementations must satisfy that same contract; mismatches fail closed.
    END.frames contains the *whole* utterance and is never sent again. Prefix
    overlap from the preceding turn is trimmed using consumed PCM positions.

    A caller-provided detector owns its model lifecycle. This function creates
    no model, downloads nothing and closes only its VAD/recognition streams.
    """
    vad_stream = detector.stream()
    try:
        stream = recognizer.stream()
    except BaseException:
        await vad_stream.aclose()
        raise
    pending = bytearray()
    capacity_available = asyncio.Event()
    input_ended = asyncio.Event()
    processed_bytes = 0
    sent_until = 0
    last_inference_index: int | None = None
    active = False

    async def forward() -> None:
        async for frame in _mono16k(audio):
            data = bytes(frame.data)
            # AgentSession may inject two seconds of shutdown silence as a burst.
            # Wait for the real VAD to consume pending PCM without expanding the
            # one-second pending-buffer budget. A stalled consumer fails finitely.
            deadline = asyncio.get_running_loop().time() + _VAD_BACKLOG_TIMEOUT
            while len(pending) + len(data) > 32_000:
                capacity_available.clear()
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise RuntimeError("VAD input capacity wait timed out")
                try:
                    await asyncio.wait_for(capacity_available.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise RuntimeError("VAD input capacity wait timed out") from None
            pending.extend(data)
            vad_stream.push_frame(frame)
            await asyncio.sleep(0)
        vad_stream.end_input()
        input_ended.set()

    def pcm(event: vad.VADEvent) -> bytes:
        if any(frame.sample_rate != 16000 or frame.num_channels != 1 for frame in event.frames):
            raise ValueError("VAD must return its original mono 16 kHz PCM windows")
        return b"".join(bytes(frame.data) for frame in event.frames)

    async def segment() -> None:
        nonlocal active, processed_bytes, sent_until, last_inference_index
        async for event in vad_stream:
            if event.type == vad.VADEventType.INFERENCE_DONE:
                data = pcm(event)
                if not data or bytes(pending[: len(data)]) != data:
                    raise ValueError("VAD processed-window contract mismatch")
                del pending[: len(data)]
                capacity_available.set()
                processed_bytes += len(data)
                last_inference_index = event.samples_index
                if active:
                    _push_bytes(stream, data)
                    sent_until = processed_bytes
            elif event.type == vad.VADEventType.START_OF_SPEECH:
                if active or event.samples_index != last_inference_index:
                    raise ValueError("VAD must emit its processed window before START")
                data = pcm(event)
                start = processed_bytes - len(data)
                if start < 0:
                    raise ValueError("VAD prefix exceeds observed input")
                # A later START can include padding already sent in the prior turn.
                data = data[max(0, sent_until - start) :]
                if not data:
                    raise ValueError("VAD START has no new audio")
                _push_bytes(stream, data)
                sent_until = processed_bytes
                active = True
            elif event.type == vad.VADEventType.END_OF_SPEECH:
                if not active or event.samples_index != last_inference_index:
                    raise ValueError("VAD end boundary has no matching active segment")
                stream.flush()
                active = False
        # Silero's end_input flush resets speech state, without synthesizing EOS.
        # Preserve the remaining sub-window tail only when speech was active.
        if active and pending:
            _push_bytes(stream, bytes(pending))
        pending.clear()
        stream.end_input()

    async def produce() -> None:
        segment_task = asyncio.create_task(segment())

        async def drain_deadline() -> None:
            # Bound a stalled short input too: it may never fill the PCM budget.
            # Normal live input has no overall wall-clock deadline.
            await input_ended.wait()
            try:
                await asyncio.wait_for(asyncio.shield(segment_task), _VAD_BACKLOG_TIMEOUT)
            except asyncio.TimeoutError:
                raise RuntimeError("VAD input drain timed out") from None

        tasks = [
            asyncio.create_task(forward()),
            segment_task,
            asyncio.create_task(drain_deadline()),
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            await utils.aio.cancel_and_wait(*tasks)
            await stream.aclose()
            raise
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await vad_stream.aclose()

    producer = asyncio.create_task(produce())
    try:
        async for event in stream:
            yield event
        await producer
    finally:
        await utils.aio.cancel_and_wait(producer)
        await stream.aclose()
