import numpy as np
import pytest

from livekit import rtc
from livekit.agents.utils.audio import AudioArrayBuffer

pytestmark = pytest.mark.unit


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


class TestSampleRates:
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

    def test_upsampled_too_large_raises(self) -> None:
        """A frame that only overflows once upsampled blames the resampling, not the caller."""
        buf = AudioArrayBuffer(buffer_size=1000, sample_rate=16000)
        with pytest.raises(ValueError, match="after resampling 8000Hz to 16000Hz"):
            buf.push_frame(_frame(list(range(900)), sr=8000))

    def test_downsampled_fits_buffer(self) -> None:
        """A frame that only fits once downsampled is accepted."""
        buf = AudioArrayBuffer(buffer_size=1000, sample_rate=8000)
        written = buf.push_frame(_frame(list(range(1800)), sr=16000))
        assert abs(written - 900) <= 10  # QUICK quality resampler may round
        assert len(buf) == written

    def test_frame_shorter_than_resampler_output(self) -> None:
        """A frame the resampler holds back writes nothing instead of raising."""
        buf = AudioArrayBuffer(buffer_size=1000, sample_rate=16000)
        assert buf.push_frame(_frame([1], sr=8000)) == 0
        assert len(buf) == 0
        assert buf.push_frame(_frame([2, 3, 4, 5], sr=8000)) > 0

    @pytest.mark.parametrize("old_samples", [[30000], [30000] * 400])
    def test_reset_discards_resampler_history(self, old_samples: list[int]) -> None:
        buf = AudioArrayBuffer(buffer_size=1000, sample_rate=16000)
        buf.push_frame(_frame(old_samples, sr=8000))
        buf.reset()
        silence = _frame([0] * 400, sr=8000)
        fresh = AudioArrayBuffer(buffer_size=1000, sample_rate=16000)

        assert buf.push_frame(silence) == fresh.push_frame(silence)
        # Independent native resamplers can differ by a few PCM units due to dithering.
        np.testing.assert_allclose(buf.read(), fresh.read(), atol=2)

    @pytest.mark.parametrize("next_sr", [16000, 24000])
    def test_reset_allows_new_input_sample_rate(self, next_sr: int) -> None:
        buf = AudioArrayBuffer(buffer_size=1000, sample_rate=16000)
        buf.push_frame(_frame([1] * 400, sr=8000))
        buf.reset()
        frame = _frame([2] * 400, sr=next_sr)
        fresh = AudioArrayBuffer(buffer_size=1000, sample_rate=16000)

        assert buf.push_frame(frame) == fresh.push_frame(frame)
        np.testing.assert_allclose(buf.read(), fresh.read(), atol=2)
