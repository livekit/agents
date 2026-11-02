"""Sample preservation across PCM chunk boundaries and mid-stream flushes."""

import pytest

from livekit.agents.utils.audio import AudioByteStream

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("num_channels", [1, 2])
@pytest.mark.parametrize("progressive", [False, True])
@pytest.mark.parametrize("split", [1, 2, 3, 960, 961, 3001, 4095, 4096, 4097])
def test_flush_preserves_samples(num_channels: int, progressive: bool, split: int) -> None:
    pcm = bytes(range(256)) * 64
    stream = AudioByteStream(
        sample_rate=24000,
        num_channels=num_channels,
        samples_per_channel=2400,
        progressive=progressive,
    )
    frames = stream.push(pcm[:split])
    frames.extend(stream.flush())

    # A flush releases complete samples for every channel, retaining only the partial sample.
    complete_bytes = split - split % (2 * num_channels)
    assert b"".join(frame.data.tobytes() for frame in frames) == pcm[:complete_bytes]
    assert stream.flush() == []

    frames.extend(stream.push(pcm[split:]))
    frames.extend(stream.flush())
    assert b"".join(frame.data.tobytes() for frame in frames) == pcm
    assert stream.buffered_duration == 0


@pytest.mark.parametrize("num_channels", [1, 2])
def test_repeated_flushes_preserve_samples(num_channels: int) -> None:
    pcm = bytes(range(256)) * 64
    stream = AudioByteStream(sample_rate=24000, num_channels=num_channels, progressive=True)
    frames = []
    for offset in range(0, len(pcm), 103):
        frames.extend(stream.push(pcm[offset : offset + 103]))
        frames.extend(stream.flush())
    assert b"".join(frame.data.tobytes() for frame in frames) == pcm


@pytest.mark.parametrize(("num_channels", "partial_bytes"), [(1, 1), (2, 1), (2, 2), (2, 3)])
def test_reset_progressive_preserves_partial_sample(num_channels: int, partial_bytes: int) -> None:
    pcm = bytes(range(256)) * 64
    stream = AudioByteStream(sample_rate=24000, num_channels=num_channels, progressive=True)
    initial_samples = 480
    initial_bytes = initial_samples * 2 * num_channels
    split = 3 * initial_bytes + partial_bytes
    frames = stream.push(pcm[:split])
    assert [frame.samples_per_channel for frame in frames] == [480, 960]
    frames.extend(stream.flush())

    stream.reset_progressive()
    next_frames = stream.push(pcm[split : 4 * initial_bytes])
    assert [frame.samples_per_channel for frame in next_frames] == [initial_samples]
    frames.extend(next_frames)
    frames.extend(stream.push(pcm[4 * initial_bytes :]))
    frames.extend(stream.flush())
    assert b"".join(frame.data.tobytes() for frame in frames) == pcm


@pytest.mark.parametrize("num_channels", [1, 2])
def test_clear_discards_partial_sample_and_resets_progressive(num_channels: int) -> None:
    stream = AudioByteStream(sample_rate=24000, num_channels=num_channels, progressive=True)
    initial_bytes = 480 * 2 * num_channels
    stream.push(b"\x11\x22" * (3 * initial_bytes // 2) + b"\x33")

    stream.clear()
    assert stream.buffered_duration == 0
    assert stream.flush() == []

    pcm = b"\x44\x55" * (initial_bytes // 2)
    frames = stream.push(pcm)
    assert [frame.samples_per_channel for frame in frames] == [480]
    assert b"".join(frame.data.tobytes() for frame in frames) == pcm
