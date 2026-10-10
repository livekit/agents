import pytest

from livekit import rtc
from livekit.agents.stt.multi_speaker_adapter import (
    PrimarySpeakerDetectionOptions,
    _PrimarySpeakerDetector,
)

pytestmark = pytest.mark.unit


def _constant_frame(value: int, *, sample_rate: int, samples_per_channel: int) -> rtc.AudioFrame:
    """Build a mono int16 frame where every sample equals ``value``."""
    sample = value.to_bytes(2, "little", signed=True)
    return rtc.AudioFrame(
        data=sample * samples_per_channel,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples_per_channel,
    )


class TestGetRmsForTimerange:
    def test_excludes_frame_before_start_time(self) -> None:
        """The RMS window for [start_time, end_time) must not include the frame
        immediately preceding start_time.
        """
        opts = PrimarySpeakerDetectionOptions(frame_size_ms=100, min_rms_samples=1)
        detector = _PrimarySpeakerDetector(primary_detection_options=opts)

        sample_rate = 1000
        samples_per_frame = 100  # 100ms at 1000Hz, matches frame_size_ms above

        # frame 1: silence, covers [0.0, 0.1)
        detector.push_audio(
            _constant_frame(0, sample_rate=sample_rate, samples_per_channel=samples_per_frame)
        )
        # frame 2: loud, covers [0.1, 0.2)
        detector.push_audio(
            _constant_frame(1000, sample_rate=sample_rate, samples_per_channel=samples_per_frame)
        )

        # query only the loud frame's range; the silent frame must not leak in
        rms = detector._get_rms_for_timerange(0.1, 0.2)

        assert rms == pytest.approx(1000.0)

    def test_window_contains_exactly_the_requested_frame_count(self) -> None:
        """A [start_time, end_time) window spanning N frames must return exactly
        N samples, not N+1 (which would happen if the frame before start_time
        leaked into the window).
        """
        opts = PrimarySpeakerDetectionOptions(frame_size_ms=100, min_rms_samples=1)
        detector = _PrimarySpeakerDetector(primary_detection_options=opts)

        sample_rate = 1000
        samples_per_frame = 100  # 100ms at 1000Hz

        for _ in range(5):  # silence
            detector.push_audio(
                _constant_frame(0, sample_rate=sample_rate, samples_per_channel=samples_per_frame)
            )
        for _ in range(5):  # loud
            detector.push_audio(
                _constant_frame(
                    1000, sample_rate=sample_rate, samples_per_channel=samples_per_frame
                )
            )

        # query the last 5 frames pushed (the loud ones), anchored on the detector's
        # own clock so this isn't sensitive to float drift in accumulated durations
        end_time = detector._pushed_duration
        start_time = end_time - 5 * detector._frame_size

        # exactly 5 frames fall inside the window; requiring 5 samples should succeed...
        detector._opt.min_rms_samples = 5
        assert detector._get_rms_for_timerange(start_time, end_time) == pytest.approx(1000.0)

        # ...but requiring 6 should fail, since only 5 frames actually fall in range.
        detector._opt.min_rms_samples = 6
        assert detector._get_rms_for_timerange(start_time, end_time) is None
