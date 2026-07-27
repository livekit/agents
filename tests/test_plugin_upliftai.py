"""Tests for the UpliftAI TTS plugin's streaming synthesis.

These are hermetic: the WebSocket client is replaced with fakes, no UpliftAI
credentials or network access are required (the ``plugin`` marker is used because
the module imports the plugin's optional dependencies, e.g. python-socketio).
"""

from __future__ import annotations

import asyncio
import io
import math
import struct
import time

import av
import pytest

from livekit.agents import APIError
from livekit.plugins import upliftai

pytestmark = pytest.mark.plugin("upliftai")

SAMPLE_RATE = 22050


class FakeClient:
    """Duck-typed WebSocketClient serving synthetic PCM audio."""

    def __init__(self, *, fail_on_request: int | None = None, delay: float = 0.15):
        self.requests: list[tuple[str, float, str | None]] = []  # (text, submit_time, request_id)
        self.cancels: list[str] = []
        self.fail_on_request = fail_on_request
        self.delay = delay
        self._feed_tasks: list[asyncio.Task] = []

    async def drain(self) -> None:
        """Stop all feed tasks; tests call this so no tasks outlive the test."""
        for task in self._feed_tasks:
            task.cancel()
        await asyncio.gather(*self._feed_tasks, return_exceptions=True)

    async def cancel(self, request_id: str) -> None:
        self.cancels.append(request_id)

    async def synthesize(self, text: str, request_id: str | None = None) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        idx = len(self.requests)
        self.requests.append((text, time.perf_counter(), request_id))

        async def _feed() -> None:
            await asyncio.sleep(self.delay)  # simulated synthesis latency
            if self.fail_on_request == idx:
                await q.put(APIError("boom from server"))
                return
            for _ in range(3):
                await q.put(b"\x00\x01" * 2205)  # 0.1s of 22050Hz mono s16
                await asyncio.sleep(0.02)
            await q.put(None)

        self._feed_tasks.append(asyncio.create_task(_feed()))
        return q

    async def disconnect(self) -> None:
        pass


def _make_tts(client: FakeClient, **kwargs) -> upliftai.TTS:
    tts = upliftai.TTS(api_key="fake-key", output_format="PCM_22050_16", **kwargs)
    tts._client = client  # type: ignore[assignment]
    return tts


async def _run_stream(
    text_pieces: list[str],
    *,
    fail_on_request: int | None = None,
    push_delay: float = 0.01,
):
    fake = FakeClient(fail_on_request=fail_on_request)
    tts = _make_tts(fake)

    stream = tts.stream()
    start = time.perf_counter()
    input_done_at: float | None = None
    first_audio_at: float | None = None
    frames = []

    async def _push() -> None:
        nonlocal input_done_at
        for piece in text_pieces:
            stream.push_text(piece)
            await asyncio.sleep(push_delay)  # simulate LLM token cadence
        stream.end_input()
        input_done_at = time.perf_counter() - start

    push_task = asyncio.create_task(_push())
    try:
        async for ev in stream:
            if first_audio_at is None:
                first_audio_at = time.perf_counter() - start
            frames.append(ev)
        await push_task
    finally:
        push_task.cancel()
        await stream.aclose()
        await fake.drain()
    return fake, frames, first_audio_at, input_done_at


async def test_streams_before_input_ends() -> None:
    """Audio must start flowing while the LLM is still streaming text, with chunks
    split at Urdu/English sentence boundaries or the max-chunk-length cap."""
    text = (
        "یہ پہلا جملہ ہے۔ کیا آپ ٹھیک ہیں؟ "
        "This is an English sentence that follows. "
        "اب ایک لمبا جملہ جو بغیر کسی وقفے کے چلتا رہتا ہے اور چلتا رہتا ہے "
        "اور مزید الفاظ شامل ہوتے رہتے ہیں تاکہ ہم زیادہ سے زیادہ حد کی جانچ کر سکیں "
        "اور یہ دیکھ سکیں کہ بفر کب مجبوراً خالی ہوتا ہے کیونکہ کوئی جملہ ختم نہیں ہو رہا "
        "یہاں تک کہ آخر کار یہ ختم ہوتا ہے۔ Short end."
    )
    pieces = [text[i : i + 12] for i in range(0, len(text), 12)]  # ~LLM-delta sized
    fake, frames, first_audio_at, input_done_at = await _run_stream(pieces)

    assert len(fake.requests) >= 4, "expected the turn to be split into multiple chunks"
    assert all(len(t) <= 220 for t, _, _ in fake.requests), "chunk exceeded max length"
    # first chunk must end at an urdu boundary, not swallow the whole turn
    assert "۔" in fake.requests[0][0] or "؟" in fake.requests[0][0]
    assert frames[-1].is_final
    assert first_audio_at is not None and input_done_at is not None
    assert first_audio_at < input_done_at, "audio must start before the LLM finishes"


async def test_server_error_propagates() -> None:
    with pytest.raises(APIError):
        await _run_stream(
            ["Hello there. ", "Second sentence here it is. ", "Third one now."],
            fail_on_request=1,
        )


async def test_short_utterance_flushed_at_end() -> None:
    """Input below min_chunk_len with no boundary is still synthesized at end of turn."""
    fake, frames, _, _ = await _run_stream(["ok"])
    assert len(fake.requests) == 1
    assert fake.requests[0][0] == "ok"
    assert frames and frames[-1].is_final


async def test_prewarm_connects_once_and_dedups() -> None:
    connect_calls = 0

    class FakeConnClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.connected = False

        async def connect(self) -> bool:
            nonlocal connect_calls
            connect_calls += 1
            await asyncio.sleep(0.05)
            self.connected = True
            return True

    tts = _make_tts(FakeConnClient())
    tts.prewarm()
    tts.prewarm()  # dedup while first is in flight
    assert connect_calls <= 1
    await asyncio.sleep(0.1)
    assert tts._client is not None and tts._client.connected
    assert connect_calls == 1
    tts.prewarm()  # already connected -> no-op
    await asyncio.sleep(0.02)
    assert connect_calls == 1
    await tts.aclose()


async def test_pending_prewarm_cancelled_on_aclose() -> None:
    class SlowConnClient(FakeClient):
        connected = False

        async def connect(self) -> bool:
            await asyncio.sleep(10)
            return True

    tts = _make_tts(SlowConnClient())
    tts.prewarm()
    await asyncio.sleep(0.01)
    await asyncio.wait_for(tts.aclose(), timeout=2)  # must not hang on the prewarm task


async def test_interruption_cancels_inflight_requests() -> None:
    fake = FakeClient(delay=0.05)
    tts = _make_tts(fake)

    stream = tts.stream()
    stream.push_text("One short sentence here. " * 10)  # ~10 chunks
    stream.end_input()

    consumed = 0
    async for _ in stream:
        consumed += 1
        if consumed >= 2:
            break
    await stream.aclose()  # simulates a barge-in
    await fake.drain()

    submitted = {rid for _, _, rid in fake.requests}
    assert len(fake.cancels) >= 1, "in-flight requests must be cancelled server-side"
    assert set(fake.cancels) <= submitted, "cancelled an unknown request id"
    # a cancel for an already-completed request is a harmless no-op, so drained
    # requests are not asserted to be excluded (emitter flush timing is not
    # deterministic relative to drain state)


def _make_mp3(freq: float, secs: float = 0.5) -> bytes:
    """Encode a sine tone as a complete standalone MP3 file."""
    buf = io.BytesIO()
    out = av.open(buf, "w", format="mp3")
    stream = out.add_stream("libmp3lame", rate=SAMPLE_RATE)
    stream.layout = "mono"
    n = int(SAMPLE_RATE * secs)
    pcm = struct.pack(
        f"<{n}h",
        *(int(10000 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n)),
    )
    frame = av.AudioFrame(format="s16", layout="mono", samples=n)
    frame.sample_rate = SAMPLE_RATE
    frame.planes[0].update(pcm)
    for packet in stream.encode(frame):
        out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()
    return buf.getvalue()


class Mp3Client(FakeClient):
    """Serves each request a complete MP3 file, streamed in slices."""

    def __init__(self, files: list[bytes]):
        super().__init__()
        self._files = files

    async def synthesize(self, text: str, request_id: str | None = None) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        idx = len(self.requests)
        self.requests.append((text, time.perf_counter(), request_id))
        data = self._files[idx % len(self._files)]

        async def _feed() -> None:
            for i in range(0, len(data), 4096):
                await q.put(data[i : i + 4096])
                await asyncio.sleep(0.005)
            await q.put(None)

        self._feed_tasks.append(asyncio.create_task(_feed()))
        return q


async def test_mp3_chunks_decode_fully() -> None:
    """Regression test for the per-chunk decode redesign: each request returns a
    complete MP3 file, and a decoder shared across chunks truncates at the first
    file boundary. Every chunk must decode independently and completely."""
    files = [_make_mp3(f) for f in (440.0, 880.0, 660.0)]
    tts = upliftai.TTS(api_key="fake-key", output_format="MP3_22050_32")
    client = Mp3Client(files)
    tts._client = client  # type: ignore[assignment]

    stream = tts.stream()
    stream.push_text("First sentence goes here. Second sentence goes here. Third sentence here.")
    stream.end_input()
    total = 0.0
    async for ev in stream:
        total += ev.frame.duration
    await stream.aclose()
    await client.drain()

    expected = 0.5 * 3
    assert total >= expected * 0.9, f"decoded only {total:.2f}s of {expected:.2f}s"


async def test_undecodable_chunk_raises() -> None:
    """The SDK decoder is fail-open (decode errors only log); the plugin must turn
    a chunk that yields zero frames into a retryable APIError, not a silent gap."""
    garbage = b"\xde\xad\xbe\xef" * 2048

    tts = upliftai.TTS(api_key="fake-key", output_format="MP3_22050_32")
    client = Mp3Client([garbage])
    tts._client = client  # type: ignore[assignment]

    stream = tts.stream()
    stream.push_text("This will not decode.")
    stream.end_input()

    with pytest.raises(APIError):
        async for _ in stream:
            pass
    await stream.aclose()
    await client.drain()
