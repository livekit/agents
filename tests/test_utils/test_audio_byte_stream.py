import pytest

from livekit.agents.utils.audio import AudioByteStream

pytestmark = pytest.mark.unit

RATE = 24000


def _pcm(ms: float, channels: int = 1) -> bytes:
    return bytes(i % 256 for i in range(int(RATE * ms / 1000) * channels * 2))


def _ms(frames: list) -> list[float]:
    return [round(f.duration * 1000, 3) for f in frames]


def _capped(channels: int = 1) -> AudioByteStream:
    return AudioByteStream(
        RATE, channels, samples_per_channel=RATE // 20, min_samples_per_channel=RATE // 100
    )


@pytest.mark.parametrize("packet_ms, expected", [(100, [50, 50]), (80, [50, 30]), (85, [50, 35])])
def test_capped_never_holds_back_a_whole_packet(packet_ms: int, expected: list[int]) -> None:
    stream = _capped()
    for _ in range(3):
        assert _ms(stream.push(_pcm(packet_ms))) == expected
    assert stream.flush() == []


def test_capped_waits_for_the_minimum_then_packs_it() -> None:
    stream = _capped()
    assert stream.push(_pcm(4)) == []
    assert stream.push(_pcm(4)) == []
    assert _ms(stream.push(_pcm(4))) == [12]
    assert _ms(stream.push(_pcm(20))) == [20]


def test_capped_stereo_emits_whole_samples_only() -> None:
    stream = _capped(channels=2)
    # a 100ms stereo packet plus two stray bytes: the half-sample stays buffered
    frames = stream.push(_pcm(100, channels=2) + b"\x00\x00")
    assert _ms(frames) == [50, 50]
    assert all(f.num_channels == 2 for f in frames)
    assert stream.flush() == []  # an incomplete sample is dropped with a warning


def test_capped_preserves_bytes_and_clears() -> None:
    stream = _capped()
    pcm = _pcm(80) + _pcm(85) + _pcm(7)
    out = b"".join(bytes(f.data) for f in stream.push(pcm))
    out += b"".join(bytes(f.data) for f in stream.flush())
    assert out == pcm

    stream.push(_pcm(7))
    stream.clear()
    assert stream.flush() == []


def test_capped_rejects_progressive() -> None:
    with pytest.raises(ValueError):
        AudioByteStream(RATE, 1, progressive=True, min_samples_per_channel=RATE // 100)
