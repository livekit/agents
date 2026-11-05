import asyncio
import contextlib
import io
import logging
import os
import struct
import time

import pytest

from livekit.agents.utils.media import AudioStreamDecoder, audio_frames_from_file

# Decodes audio on background threads / executors with blocking waits; it deadlocks when forced
# to share one event loop with other tests.
pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "change-sophie.opus")


def _make_wav(sample_rate: int, num_channels: int, num_samples: int) -> bytes:
    """Generate a valid PCM16 WAV file in memory."""
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * num_channels * (bits_per_sample // 8)

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(
        struct.pack(
            "<HHIIHH", 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample
        )
    )
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    # silence PCM data
    buf.write(b"\x00" * data_size)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_wav_inline_decoder():
    """WAV inline decoder should produce the correct total number of samples."""
    sample_rate = 24000
    num_channels = 1
    num_samples = 24000  # 1 second of audio

    wav_bytes = _make_wav(sample_rate, num_channels, num_samples)

    decoder = AudioStreamDecoder(
        sample_rate=sample_rate, num_channels=num_channels, format="audio/wav"
    )

    # push in small chunks to exercise the incremental state machine
    chunk_size = 37  # deliberately odd size to split across headers and data
    for i in range(0, len(wav_bytes), chunk_size):
        decoder.push(wav_bytes[i : i + chunk_size])
    decoder.end_input()

    total_samples = 0
    async for frame in decoder:
        assert frame.sample_rate == sample_rate
        assert frame.num_channels == num_channels
        total_samples += frame.samples_per_channel

    assert total_samples == num_samples
    await decoder.aclose()


@pytest.mark.asyncio
async def test_wav_multi_chunk_each_with_headers():
    """Each push() is a complete WAV file; decoder must re-parse headers each time."""
    sample_rate = 24000
    num_channels = 1
    samples_per_chunk = 2400  # 100ms per chunk
    num_chunks = 5

    wav_chunk = _make_wav(sample_rate, num_channels, samples_per_chunk)

    decoder = AudioStreamDecoder(
        sample_rate=sample_rate, num_channels=num_channels, format="audio/wav"
    )

    for _ in range(num_chunks):
        decoder.push(wav_chunk)
    decoder.end_input()

    total_samples = 0
    async for frame in decoder:
        assert frame.sample_rate == sample_rate
        assert frame.num_channels == num_channels
        total_samples += frame.samples_per_channel

    assert total_samples == samples_per_chunk * num_chunks
    await decoder.aclose()


@pytest.mark.asyncio
async def test_wav_multi_chunk_with_resampling():
    """Multiple WAV chunks with resampling should produce correct total duration."""
    src_rate = 16000
    out_rate = 48000
    num_channels = 1
    samples_per_chunk = 1600  # 100ms at source rate
    num_chunks = 3

    wav_chunk = _make_wav(src_rate, num_channels, samples_per_chunk)

    decoder = AudioStreamDecoder(
        sample_rate=out_rate, num_channels=num_channels, format="audio/wav"
    )

    for _ in range(num_chunks):
        decoder.push(wav_chunk)
    decoder.end_input()

    total_samples = 0
    async for frame in decoder:
        assert frame.sample_rate == out_rate
        total_samples += frame.samples_per_channel

    expected = samples_per_chunk * num_chunks * out_rate // src_rate
    assert abs(total_samples - expected) <= out_rate // 50  # within 20ms tolerance
    await decoder.aclose()


@pytest.mark.asyncio
async def test_aclose_while_probing_is_not_an_error(caplog: pytest.LogCaptureFixture) -> None:
    """Closing a decoder mid-probe is intentional and must not be reported as a decode failure."""
    with open(TEST_AUDIO_FILEPATH, "rb") as f:
        head = f.read(16)  # far less than av.open needs to probe the container

    decoder = AudioStreamDecoder()
    decoder.push(head)
    await asyncio.sleep(0.05)  # the decode thread is now blocked reading inside av.open

    with caplog.at_level(logging.DEBUG, logger="livekit.agents"):
        await decoder.aclose()
        await asyncio.sleep(0.05)

    assert [r.getMessage() for r in caplog.records if "error decoding" in r.getMessage()] == []


@pytest.mark.asyncio
async def test_aborted_read_does_not_signal_end_of_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """The BackgroundAudioPlayer stop() path: an aborted read is a truncated file, not an EOF.

    Signalling the end of input hands the partial file to PyAV as a complete container, which
    it then reports as invalid data.
    """
    real_push, real_end_input = AudioStreamDecoder.push, AudioStreamDecoder.end_input
    pushes: list[None] = []
    started = asyncio.Event()
    ends: list[None] = []

    def slow_push(self: AudioStreamDecoder, chunk: bytes) -> None:
        pushes.append(None)
        started.set()
        time.sleep(0.005)  # hold the reader mid-file for the whole test
        real_push(self, chunk)

    def spy_end_input(self: AudioStreamDecoder) -> None:
        ends.append(None)
        real_end_input(self)

    monkeypatch.setattr(AudioStreamDecoder, "push", slow_push)
    monkeypatch.setattr(AudioStreamDecoder, "end_input", spy_end_input)

    gen = audio_frames_from_file(TEST_AUDIO_FILEPATH)

    async def drain() -> None:
        async for _ in gen:
            pass

    task = asyncio.create_task(drain())
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await gen.aclose()  # what PlayHandle.stop() ultimately triggers

    assert pushes, "the reader never ran, so the abort was never exercised"
    assert ends == []


@pytest.mark.asyncio
async def test_closing_partial_decoders_releases_admission_while_wrappers_remain_alive() -> None:
    from livekit.agents import _core

    with open(TEST_AUDIO_FILEPATH, "rb") as source:
        head = source.read(16)
    closed = []
    for _ in range(40):
        decoder = AudioStreamDecoder(format="audio/ogg")
        decoder.push(head)
        await decoder.aclose()
        closed.append(decoder)

    # A direct native push raises admission failures; the public stream catches them as EOF.
    replacement = _core.AudioDecoder("audio/ogg")
    replacement.push(head)
    await replacement.aclose()
