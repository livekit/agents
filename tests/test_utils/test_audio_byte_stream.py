import pytest

from livekit import rtc
from livekit.agents.utils.audio import AudioByteStream

pytestmark = pytest.mark.unit


def _pcm(ms: int, rate: int = 24000, channels: int = 1) -> bytes:
    return bytes(i % 256 for i in range(rate * ms // 1000 * channels * 2))


def _durations(frames: list[rtc.AudioFrame]) -> list[float]:
    return [frame.duration * 1000 for frame in frames]


@pytest.mark.parametrize("rate", [16000, 24000, 44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("packet_ms", [50, 100])
def test_progressive_starts_with_available_target_frames(
    rate: int, channels: int, packet_ms: int
) -> None:
    stream = AudioByteStream(rate, channels, samples_per_channel=rate // 20, progressive=True)
    data = _pcm(packet_ms, rate, channels)

    frames = stream.push(data)

    assert _durations(frames) == pytest.approx([50] * (packet_ms // 50))
    assert b"".join(bytes(frame.data) for frame in frames) == data
    assert all(frame.sample_rate == rate and frame.num_channels == channels for frame in frames)
    assert stream.flush() == []


@pytest.mark.parametrize("progressive", [False, True])
def test_realtime_large_packets_have_no_initial_playout_gap(progressive: bool) -> None:
    stream = AudioByteStream(24000, 1, samples_per_channel=1200, progressive=progressive)
    playout_end_ms = 0.0
    for arrival_ms in range(0, 1000, 100):
        assert playout_end_ms == pytest.approx(arrival_ms)
        frames = stream.push(_pcm(100))
        playout_end_ms += sum(_durations(frames))
        assert playout_end_ms == pytest.approx(arrival_ms + 100)


@pytest.mark.parametrize("progressive", [False, True])
def test_small_packets_keep_their_startup_behavior(progressive: bool) -> None:
    stream = AudioByteStream(24000, 1, samples_per_channel=1200, progressive=progressive)
    expected = [[20], [], [40], [], [], [50]] if progressive else [[], [], [50], [], [50], []]
    frames = []
    for durations in expected:
        emitted = stream.push(_pcm(20))
        assert _durations(emitted) == pytest.approx(durations)
        frames.extend(emitted)
    frames.extend(stream.flush())
    assert b"".join(bytes(frame.data) for frame in frames) == _pcm(20) * len(expected)


@pytest.mark.parametrize("prefix_ms", [0, 10])
def test_target_frame_can_accumulate_before_first_emission(prefix_ms: int) -> None:
    stream = AudioByteStream(24000, 1, samples_per_channel=1200, progressive=True)
    prefix, suffix = _pcm(prefix_ms), _pcm(50 - prefix_ms)
    assert stream.push(prefix) == []
    frames = stream.push(memoryview(suffix))
    assert _durations(frames) == pytest.approx([50])
    assert b"".join(bytes(frame.data) for frame in frames) == prefix + suffix
    assert stream.flush() == []


@pytest.mark.parametrize(("packet_ms", "expected"), [(20, [20]), (100, [50, 50])])
@pytest.mark.parametrize("initial_ms", [20, 100])
def test_clear_resets_startup_and_discards_previous_audio(
    initial_ms: int, packet_ms: int, expected: list[int]
) -> None:
    stream = AudioByteStream(24000, 1, samples_per_channel=1200, progressive=True)
    stream.push(_pcm(initial_ms))
    assert stream.push(_pcm(10)) == []
    stream.clear()

    data = _pcm(packet_ms)
    frames = stream.push(data)
    assert _durations(frames) == pytest.approx(expected)
    assert b"".join(bytes(frame.data) for frame in frames) == data
    assert stream.flush() == []


def test_large_packet_does_not_skip_a_ramp_already_started() -> None:
    stream = AudioByteStream(24000, 1, samples_per_channel=1200, progressive=True)
    assert _durations(stream.push(_pcm(20))) == pytest.approx([20])
    data = _pcm(100)
    frames = stream.push(data)
    assert _durations(frames) == pytest.approx([40, 50])
    tail = stream.flush()
    assert _durations(tail) == pytest.approx([10])
    assert b"".join(bytes(frame.data) for frame in frames + tail) == data


@pytest.mark.parametrize(("initial_ms", "next_frame_ms"), [(20, 40), (100, 50)])
def test_flush_preserves_progression(initial_ms: int, next_frame_ms: int) -> None:
    stream = AudioByteStream(24000, 1, samples_per_channel=1200, progressive=True)
    stream.push(_pcm(initial_ms))
    assert stream.push(_pcm(10)) == []
    assert _durations(stream.flush()) == pytest.approx([10])
    assert stream.flush() == []
    assert stream.push(_pcm(20)) == []
    assert _durations(stream.push(_pcm(next_frame_ms - 20))) == pytest.approx([next_frame_ms])


def test_progressive_target_below_startup_minimum() -> None:
    stream = AudioByteStream(24000, 1, samples_per_channel=240, progressive=True)
    assert _durations(stream.push(_pcm(30))) == pytest.approx([10, 10, 10])
    assert stream.flush() == []


def test_progressive_default_target_with_partial_tail() -> None:
    stream = AudioByteStream(24000, 1, progressive=True)
    data = _pcm(210)
    frames = stream.push(data)
    assert _durations(frames) == pytest.approx([100, 100])
    tail = stream.flush()
    assert _durations(tail) == pytest.approx([10])
    assert b"".join(bytes(frame.data) for frame in frames + tail) == data
