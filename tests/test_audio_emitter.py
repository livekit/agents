"""Tests for AudioEmitter: is_final, timed transcripts, duration tracking, segments."""

from __future__ import annotations

import asyncio
import io
import wave

import pytest

from livekit.agents import tts, utils
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT, TimedString


def _make_pcm(sample_rate: int, num_channels: int, duration_ms: int) -> bytes:
    num_samples = sample_rate * duration_ms // 1000 * num_channels
    return b"\x00\x00" * num_samples


def _make_wav(sample_rate: int, num_channels: int, duration_ms: int) -> bytes:
    num_samples = sample_rate * duration_ms // 1000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(num_channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * (num_samples * num_channels))
    return buf.getvalue()


SR = 24000
NC = 1


async def _run_emitter(
    produce_fn, *, stream: bool = False, expect_finals: int = 1, mime_type: str = "audio/pcm"
):
    """Run emitter, collect events until we've seen expect_finals is_final events."""
    dst_ch = utils.aio.Chan[tts.SynthesizedAudio]()
    emitter = tts.AudioEmitter(label="test", dst_ch=dst_ch)
    emitter.initialize(
        request_id="req1",
        sample_rate=SR,
        num_channels=NC,
        mime_type=mime_type,
        stream=stream,
    )

    events: list[tts.SynthesizedAudio] = []
    finals_seen = 0

    async def collect():
        nonlocal finals_seen
        async for ev in dst_ch:
            events.append(ev)
            if ev.is_final:
                finals_seen += 1
                if finals_seen >= expect_finals:
                    return

    ct = asyncio.create_task(collect())
    produce_fn(emitter)
    await emitter.join()
    await ct
    return emitter, events


@pytest.mark.asyncio
async def test_basic_is_final():
    def produce(e):
        for _ in range(50):
            e.push(_make_pcm(SR, NC, 10))
        e.end_input()

    _, events = await _run_emitter(produce)

    assert len(events) > 0
    assert events[-1].is_final
    assert all(not e.is_final for e in events[:-1])
    total = sum(e.frame.duration for e in events)
    assert abs(total - 0.5) < 0.02


@pytest.mark.asyncio
async def test_single_tiny_push():
    def produce(e):
        e.push(_make_pcm(SR, NC, 5))
        e.end_input()

    _, events = await _run_emitter(produce)
    assert len(events) >= 1
    assert events[-1].is_final


@pytest.mark.asyncio
async def test_timed_transcripts_all_preserved():
    words = ["hello", "world", "this", "is", "a", "test"]

    def produce(e):
        for i, word in enumerate(words):
            e.push(_make_pcm(SR, NC, 80))
            e.push_timed_transcript(
                TimedString(text=word, start_time=i * 0.08, end_time=(i + 1) * 0.08)
            )
        e.end_input()

    _, events = await _run_emitter(produce)

    recovered = []
    for ev in events:
        for t in ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, []):
            if isinstance(t, TimedString):
                recovered.append(str(t))

    assert recovered == words, f"expected {words}, got {recovered}"


@pytest.mark.asyncio
async def test_timed_transcripts_have_timestamps():
    def produce(e):
        e.push(_make_pcm(SR, NC, 200))
        e.push_timed_transcript(TimedString(text="hello", start_time=0.0, end_time=0.1))
        e.push_timed_transcript(TimedString(text="world", start_time=0.1, end_time=0.2))
        e.end_input()

    _, events = await _run_emitter(produce)

    all_ts: list[TimedString] = []
    for ev in events:
        for t in ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, []):
            if isinstance(t, TimedString):
                all_ts.append(t)

    assert len(all_ts) == 2
    assert str(all_ts[0]) == "hello"
    assert all_ts[0].start_time == 0.0
    assert str(all_ts[1]) == "world"
    assert all_ts[1].start_time == 0.1


@pytest.mark.asyncio
async def test_streaming_segments_each_have_is_final():
    def produce(e):
        for seg_idx in range(3):
            e.start_segment(segment_id=f"seg_{seg_idx}")
            for _ in range(10):
                e.push(_make_pcm(SR, NC, 10))
            e.end_segment()
        e.end_input()

    _, events = await _run_emitter(produce, stream=True, expect_finals=3)

    finals = [e for e in events if e.is_final]
    assert len(finals) == 3
    total = sum(e.frame.duration for e in events)
    assert abs(total - 0.3) < 0.02


@pytest.mark.asyncio
async def test_streaming_transcripts_across_segments():
    def produce(e):
        e.start_segment(segment_id="seg_0")
        e.push(_make_pcm(SR, NC, 200))
        e.push_timed_transcript(TimedString(text="first", start_time=0.0, end_time=0.2))
        e.end_segment()

        e.start_segment(segment_id="seg_1")
        e.push(_make_pcm(SR, NC, 200))
        e.push_timed_transcript(TimedString(text="second", start_time=0.0, end_time=0.2))
        e.end_segment()

        e.end_input()

    _, events = await _run_emitter(produce, stream=True, expect_finals=2)

    all_words = []
    for ev in events:
        for t in ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, []):
            if isinstance(t, TimedString):
                all_words.append(str(t))

    assert "first" in all_words
    assert "second" in all_words


@pytest.mark.asyncio
async def test_pushed_duration_accurate():
    def produce(e):
        e.push(_make_pcm(SR, NC, 500))
        e.end_input()

    emitter, _ = await _run_emitter(produce)
    assert abs(emitter.pushed_duration() - 0.5) < 0.02


@pytest.mark.asyncio
async def test_marker_does_not_inflate_duration():
    """flush() followed by end_input() with no new audio: marker should not inflate duration."""

    def produce(e):
        e.push(_make_pcm(SR, NC, 100))
        e.flush()
        e.end_input()

    emitter, events = await _run_emitter(produce)
    assert events[-1].is_final
    assert abs(emitter.pushed_duration() - 0.1) < 0.02


@pytest.mark.asyncio
async def test_flush_mid_stream_preserves_decoder_state():
    """Regression: flush() between two chunks of the same encoded stream must not
    tear the decoder down.

    A stateful codec (WAV/OGG/MP3) parses headers once; if the FlushSegment branch
    in the decoder path discards the decoder, the next chunk — which is mid-file
    payload (raw PCM continuation, no RIFF) — gets fed to a fresh decoder that
    expects a header and fails. The user-visible symptom is
    ``ValueError: Invalid WAV file: missing RIFF/WAVE: ...`` and silent audio loss.

    flush() must release the held-back tail without touching the decoder.
    """
    duration_ms = 500
    wav = _make_wav(SR, NC, duration_ms)
    # Split somewhere inside the data chunk so chunk2 has no RIFF header — this is
    # the exact shape the streaming decoder branch must handle.
    split = 100
    chunk1, chunk2 = wav[:split], wav[split:]
    assert chunk1[:4] == b"RIFF"
    assert chunk2[:4] != b"RIFF"

    def produce(e):
        e.push(chunk1)
        e.flush()
        e.push(chunk2)
        e.end_input()

    emitter, events = await _run_emitter(produce, mime_type="audio/wav")

    expected_samples = SR * duration_ms // 1000
    total_samples = sum(ev.frame.samples_per_channel for ev in events)
    # Accept a small tolerance for the trailing-tail handling, but anything close
    # to the full payload means the decoder survived the flush. Pre-fix this
    # would drop nearly everything (a few hundred samples of pre-flush audio).
    assert total_samples >= int(expected_samples * 0.95), (
        f"lost samples after mid-stream flush: got {total_samples}, expected ~{expected_samples}"
    )
    assert events[-1].is_final


@pytest.mark.asyncio
async def test_every_frame_has_timed_transcript_metadata():
    """Every emitted frame must have the USERDATA_TIMED_TRANSCRIPT key in userdata."""

    def produce(e):
        e.push(_make_pcm(SR, NC, 300))
        e.push_timed_transcript(TimedString(text="word", start_time=0.0, end_time=0.3))
        e.end_input()

    _, events = await _run_emitter(produce)

    for i, ev in enumerate(events):
        assert USERDATA_TIMED_TRANSCRIPT in ev.frame.userdata, (
            f"frame {i} (is_final={ev.is_final}) missing {USERDATA_TIMED_TRANSCRIPT} in userdata"
        )
