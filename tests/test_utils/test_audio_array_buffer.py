import time

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.utils.audio import AudioArrayBuffer


def _frame(samples: list[int], *, sr: int = 16000, ch: int = 1) -> rtc.AudioFrame:
    data = np.array(samples, dtype=np.int16).tobytes()
    return rtc.AudioFrame(
        data=data, sample_rate=sr, num_channels=ch, samples_per_channel=len(samples) // ch
    )


def _stereo(left: list[int], right: list[int], *, sr: int = 16000) -> rtc.AudioFrame:
    interleaved = np.empty(len(left) * 2, dtype=np.int16)
    interleaved[0::2] = np.array(left, dtype=np.int16)
    interleaved[1::2] = np.array(right, dtype=np.int16)
    return rtc.AudioFrame(
        data=interleaved.tobytes(), sample_rate=sr, num_channels=2, samples_per_channel=len(left)
    )


def _assert_eq(buf: AudioArrayBuffer, expected: list[int]) -> None:
    np.testing.assert_array_equal(buf.read(), expected)


class TestInit:
    def test_defaults(self) -> None:
        buf = AudioArrayBuffer(buffer_size=100)
        assert buf._dtype == np.int16
        assert len(buf._buffer) == 100
        assert len(buf.read()) == 0

    def test_custom_dtype(self) -> None:
        buf = AudioArrayBuffer(buffer_size=50, dtype=np.float32)
        assert buf._buffer.dtype == np.float32


class TestPushFrame:
    @pytest.mark.parametrize(
        ("buf_size", "frames", "expected"),
        [
            (10, [[1, 2, 3, 4, 5]], [1, 2, 3, 4, 5]),
            (10, [[1, 2, 3], [4, 5, 6]], [1, 2, 3, 4, 5, 6]),
            (5, [[10, 20, 30, 40, 50]], [10, 20, 30, 40, 50]),
            (6, [[1, 2, 3, 4], [5, 6, 7]], [2, 3, 4, 5, 6, 7]),
            (4, [[1, 2, 3, 4], [5, 6]], [3, 4, 5, 6]),
            (4, [[-32768, -1, 0, 32767]], [-32768, -1, 0, 32767]),
        ],
        ids=[
            "single_frame",
            "multiple_frames",
            "exact_fill",
            "overflow_shifts_oldest",
            "repeated_overflow",
            "int16_boundaries",
        ],
    )
    def test_push_and_read(
        self, buf_size: int, frames: list[list[int]], expected: list[int]
    ) -> None:
        buf = AudioArrayBuffer(buffer_size=buf_size)
        for f in frames:
            buf.push_frame(_frame(f))
        _assert_eq(buf, expected)

    def test_too_large_raises(self) -> None:
        buf = AudioArrayBuffer(buffer_size=3)
        with pytest.raises(ValueError, match="greater than the buffer size"):
            buf.push_frame(_frame([1, 2, 3, 4]))


class TestStereoDownmix:
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ([100, 200], [300, 400], [200, 300]),
            ([-100, 200], [100, -200], [0, 0]),
        ],
        ids=["averages_channels", "cancellation"],
    )
    def test_downmix(self, left: list[int], right: list[int], expected: list[int]) -> None:
        buf = AudioArrayBuffer(buffer_size=10)
        buf.push_frame(_stereo(left, right))
        _assert_eq(buf, expected)

    def test_stereo_then_mono(self) -> None:
        buf = AudioArrayBuffer(buffer_size=10)
        buf.push_frame(_stereo([100, 200], [300, 400]))
        buf.push_frame(_frame([50, 60]))
        _assert_eq(buf, [200, 300, 50, 60])


class TestShift:
    @pytest.mark.parametrize(
        ("data", "shift", "expected"),
        [
            ([1, 2, 3, 4, 5], 2, [3, 4, 5]),
            ([1, 2, 3], 3, []),
            ([1, 2, 3], 100, []),
            ([1, 2, 3], 0, [1, 2, 3]),
        ],
        ids=["removes_oldest", "shift_all", "clamps_to_available", "zero_is_noop"],
    )
    def test_shift(self, data: list[int], shift: int, expected: list[int]) -> None:
        buf = AudioArrayBuffer(buffer_size=10)
        buf.push_frame(_frame(data))
        buf.shift(shift)
        _assert_eq(buf, expected)

    def test_on_empty(self) -> None:
        buf = AudioArrayBuffer(buffer_size=10)
        buf.shift(5)
        assert len(buf.read()) == 0

    def test_shift_then_push(self) -> None:
        buf = AudioArrayBuffer(buffer_size=6)
        buf.push_frame(_frame([1, 2, 3, 4]))
        buf.shift(2)
        buf.push_frame(_frame([5, 6]))
        _assert_eq(buf, [3, 4, 5, 6])


class TestRead:
    def test_returns_copy(self) -> None:
        buf = AudioArrayBuffer(buffer_size=10)
        buf.push_frame(_frame([1, 2, 3]))
        result = buf.read()
        result[0] = 999
        _assert_eq(buf, [1, 2, 3])

    def test_dtype_preserved(self) -> None:
        buf = AudioArrayBuffer(buffer_size=10)
        buf.push_frame(_frame([1, 2, 3]))
        assert buf.read().dtype == np.int16

    def test_length_matches_written(self) -> None:
        buf = AudioArrayBuffer(buffer_size=100)
        buf.push_frame(_frame([1, 2, 3, 4, 5]))
        assert len(buf.read()) == 5


class TestSlidingWindow:
    def test_continuous_pushes(self) -> None:
        buf = AudioArrayBuffer(buffer_size=4)
        buf.push_frame(_frame([1, 2, 3, 4]))
        _assert_eq(buf, [1, 2, 3, 4])
        buf.push_frame(_frame([5, 6]))
        _assert_eq(buf, [3, 4, 5, 6])
        buf.push_frame(_frame([7, 8]))
        _assert_eq(buf, [5, 6, 7, 8])

    def test_full_replacement(self) -> None:
        buf = AudioArrayBuffer(buffer_size=4)
        buf.push_frame(_frame([1, 2, 3, 4]))
        buf.push_frame(_frame([10, 20, 30, 40]))
        _assert_eq(buf, [10, 20, 30, 40])

    def test_many_small_pushes(self) -> None:
        buf = AudioArrayBuffer(buffer_size=3)
        for i in range(10):
            buf.push_frame(_frame([i]))
        _assert_eq(buf, [7, 8, 9])


class TestSampleRates:
    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000], ids=["8k", "16k", "24k", "48k"])
    def test_same_rate_no_resampling(self, sr: int) -> None:
        """Frames matching the buffer's sample rate are stored without resampling."""
        n = sr * 50 // 1000  # 50ms
        buf = AudioArrayBuffer(buffer_size=sr, sample_rate=sr)
        buf.push_frame(_frame(list(range(n)), sr=sr))
        assert len(buf.read()) == n

    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000], ids=["8k", "16k", "24k", "48k"])
    def test_overflow_at_different_rates(self, sr: int) -> None:
        """Buffer should slide correctly with realistic frame sizes at each rate."""
        n = sr * 50 // 1000
        buf = AudioArrayBuffer(buffer_size=sr, sample_rate=sr)

        # push 1.5s worth of frames to trigger overflow
        for _ in range(30):
            buf.push_frame(_frame(list(range(n)), sr=sr))
        assert len(buf.read()) == sr

    @pytest.mark.parametrize(
        ("frame_sr", "buf_sr"),
        [(8000, 16000), (48000, 16000), (24000, 16000), (16000, 48000)],
        ids=["8k_to_16k", "48k_to_16k", "24k_to_16k", "16k_to_48k"],
    )
    def test_resamples_to_buffer_rate(self, frame_sr: int, buf_sr: int) -> None:
        """Frames at a different sample rate are resampled to the buffer's rate."""
        frame_samples = frame_sr * 50 // 1000
        buf = AudioArrayBuffer(buffer_size=buf_sr, sample_rate=buf_sr)
        buf.push_frame(_frame(list(range(frame_samples)), sr=frame_sr))
        expected = buf_sr * 50 // 1000
        assert abs(len(buf.read()) - expected) <= 10  # QUICK quality resampler may round


class TestPerformance:
    @pytest.mark.parametrize("sr", [16000, 48000], ids=["16k", "48k"])
    def test_3s_buffer_50ms_frames(self, sr: int) -> None:
        """60s of audio in 50ms frames (1200 pushes) into a 3s buffer."""
        buf_s, frame_ms, total_s = 3, 50, 60
        buf = AudioArrayBuffer(buffer_size=sr * buf_s, sample_rate=sr)
        frame = _frame(list(range(sr * frame_ms // 1000)), sr=sr)

        start = time.perf_counter()
        for _ in range(total_s * 1000 // frame_ms):
            buf.push_frame(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(buf.read()) == sr * buf_s
        assert elapsed_ms < 100, f"took {elapsed_ms:.1f}ms, expected <100ms"
