import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from livekit import rtc
from livekit.agents.utils.codecs import (
    AudioStreamDecoder,
    AudioStreamEncoder,
    EncodedAudioChunk,
)
from livekit.agents.utils.codecs.encoder import FrameBuffer

TEST_AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "change-sophie.opus")


def create_test_frame(
    duration_ms: int = 100, sample_rate: int = 48000, num_channels: int = 1
) -> rtc.AudioFrame:
    """Create a test audio frame with silence."""
    samples_per_channel = int(sample_rate * duration_ms / 1000)
    data = b"\x00\x00" * samples_per_channel * num_channels
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


def create_sine_wave_frame(
    duration_ms: int = 100,
    sample_rate: int = 48000,
    num_channels: int = 1,
    frequency: int = 440,
) -> rtc.AudioFrame:
    """Create a test audio frame with a sine wave."""
    import math

    samples_per_channel = int(sample_rate * duration_ms / 1000)
    data = bytearray()
    for i in range(samples_per_channel):
        sample = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate))
        sample_bytes = sample.to_bytes(2, byteorder="little", signed=True)
        for _ in range(num_channels):
            data.extend(sample_bytes)
    return rtc.AudioFrame(
        data=bytes(data),
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


@pytest.mark.asyncio
async def test_pcm_passthrough():
    """Test that PCM format passes through frames unchanged."""
    encoder = AudioStreamEncoder(sample_rate=48000, num_channels=1, format="pcm")

    frames = [create_test_frame(100) for _ in range(5)]
    total_duration = sum(f.samples_per_channel / f.sample_rate for f in frames)

    for frame in frames:
        encoder.push(frame)
    encoder.end_input()

    chunks: list[EncodedAudioChunk] = []
    async for chunk in encoder:
        chunks.append(chunk)

    await encoder.aclose()

    # f.data is a buffer with int16 elements, so we need bytes(f.data) for byte count
    total_input_bytes = sum(len(bytes(f.data)) for f in frames)
    total_output_bytes = sum(len(c.data) for c in chunks)
    assert total_output_bytes == total_input_bytes

    total_chunk_duration = sum(c.duration for c in chunks)
    assert abs(total_chunk_duration - total_duration) < 0.01


@pytest.mark.asyncio
async def test_opus_encoding():
    """Test Opus encoding produces valid data."""
    encoder = AudioStreamEncoder(
        sample_rate=48000,
        num_channels=1,
        format="opus",
        bitrate=64000,
    )

    frames = [create_sine_wave_frame(100) for _ in range(10)]

    for frame in frames:
        encoder.push(frame)
    encoder.end_input()

    chunks: list[EncodedAudioChunk] = []
    async for chunk in encoder:
        chunks.append(chunk)

    await encoder.aclose()

    assert len(chunks) > 0
    total_bytes = sum(len(c.data) for c in chunks)
    assert total_bytes > 0

    # OGG container magic bytes
    all_data = b"".join(c.data for c in chunks)
    assert all_data[:4] == b"OggS", "Expected OGG container format"


@pytest.mark.asyncio
async def test_mp3_encoding():
    """Test MP3 encoding produces valid data."""
    encoder = AudioStreamEncoder(
        sample_rate=48000,
        num_channels=1,
        format="mp3",
        bitrate=128000,
    )

    frames = [create_sine_wave_frame(100) for _ in range(10)]

    for frame in frames:
        encoder.push(frame)
    encoder.end_input()

    chunks: list[EncodedAudioChunk] = []
    async for chunk in encoder:
        chunks.append(chunk)

    await encoder.aclose()

    assert len(chunks) > 0
    total_bytes = sum(len(c.data) for c in chunks)
    assert total_bytes > 0


@pytest.mark.asyncio
async def test_stereo_encoding():
    """Test encoding stereo audio."""
    encoder = AudioStreamEncoder(
        sample_rate=48000,
        num_channels=2,
        format="opus",
        bitrate=96000,
    )

    frames = [create_sine_wave_frame(100, num_channels=2) for _ in range(5)]

    for frame in frames:
        encoder.push(frame)
    encoder.end_input()

    chunks: list[EncodedAudioChunk] = []
    async for chunk in encoder:
        chunks.append(chunk)

    await encoder.aclose()

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_opus_sample_rate_adjustment():
    """Test that non-Opus sample rates get adjusted correctly."""
    encoder = AudioStreamEncoder(
        sample_rate=44100,  # not directly supported by Opus
        num_channels=1,
        format="opus",
    )

    frames = [create_sine_wave_frame(100, sample_rate=44100) for _ in range(5)]

    for frame in frames:
        encoder.push(frame)
    encoder.end_input()

    chunks: list[EncodedAudioChunk] = []
    async for chunk in encoder:
        chunks.append(chunk)

    await encoder.aclose()

    assert len(chunks) > 0
    all_data = b"".join(c.data for c in chunks)
    assert all_data[:4] == b"OggS"


@pytest.mark.asyncio
async def test_roundtrip_encode_decode():
    """Test encoding and then decoding produces similar audio."""
    if not os.path.exists(TEST_AUDIO_FILEPATH):
        pytest.skip(f"Test file not found: {TEST_AUDIO_FILEPATH}")

    decoder = AudioStreamDecoder(sample_rate=48000, num_channels=1)
    with open(TEST_AUDIO_FILEPATH, "rb") as f:
        opus_data = f.read()
    decoder.push(opus_data)
    decoder.end_input()

    original_frames: list[rtc.AudioFrame] = []
    async for frame in decoder:
        original_frames.append(frame)
    await decoder.aclose()

    encoder = AudioStreamEncoder(
        sample_rate=48000,
        num_channels=1,
        format="opus",
        bitrate=64000,
    )

    for frame in original_frames:
        encoder.push(frame)
    encoder.end_input()

    encoded_chunks: list[EncodedAudioChunk] = []
    async for chunk in encoder:
        encoded_chunks.append(chunk)
    await encoder.aclose()

    encoded_data = b"".join(c.data for c in encoded_chunks)
    decoder2 = AudioStreamDecoder(sample_rate=48000, num_channels=1, format="audio/opus")
    decoder2.push(encoded_data)
    decoder2.end_input()

    decoded_frames: list[rtc.AudioFrame] = []
    async for frame in decoder2:
        decoded_frames.append(frame)
    await decoder2.aclose()

    assert len(decoded_frames) > 0

    original_duration = sum(f.samples_per_channel / f.sample_rate for f in original_frames)
    decoded_duration = sum(f.samples_per_channel / f.sample_rate for f in decoded_frames)
    assert abs(original_duration - decoded_duration) < 0.5


@pytest.mark.asyncio
async def test_encoder_properties():
    """Test encoder property getters."""
    encoder = AudioStreamEncoder(
        sample_rate=24000,
        num_channels=2,
        format="opus",
        bitrate=96000,
    )

    assert encoder.sample_rate == 24000
    assert encoder.num_channels == 2
    assert encoder.format == "opus"
    assert encoder.bitrate == 96000

    await encoder.aclose()


@pytest.mark.asyncio
async def test_default_bitrates():
    """Test that default bitrates are set correctly."""
    opus_encoder = AudioStreamEncoder(format="opus")
    assert opus_encoder.bitrate == 64000
    await opus_encoder.aclose()

    mp3_encoder = AudioStreamEncoder(format="mp3")
    assert mp3_encoder.bitrate == 128000
    await mp3_encoder.aclose()

    pcm_encoder = AudioStreamEncoder(format="pcm")
    assert pcm_encoder.bitrate is None
    await pcm_encoder.aclose()


@pytest.mark.asyncio
async def test_empty_input():
    """Test encoder handles empty input gracefully."""
    encoder = AudioStreamEncoder(format="opus")
    encoder.end_input()

    chunks = []
    async for chunk in encoder:
        chunks.append(chunk)

    await encoder.aclose()

    assert len(chunks) == 0


def test_frame_buffer():
    """Test the FrameBuffer thread-safe buffer."""
    buffer = FrameBuffer()
    frames = [create_test_frame(100) for _ in range(5)]
    received_frames: list[rtc.AudioFrame] = []
    write_completed = threading.Event()

    def writer():
        for frame in frames:
            buffer.write(frame)
        buffer.end_input()
        write_completed.set()

    def reader():
        while True:
            batch = buffer.read_all()
            if not batch:
                if buffer.is_eof:
                    break
                continue
            received_frames.extend(batch)

    with ThreadPoolExecutor(max_workers=2) as executor:
        reader_future = executor.submit(reader)
        writer_future = executor.submit(writer)

        writer_future.result()
        reader_future.result()

    assert len(received_frames) == len(frames)
