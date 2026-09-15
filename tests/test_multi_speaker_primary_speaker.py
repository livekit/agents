import pytest

from livekit import rtc
from livekit.agents.stt import SpeechData
from livekit.agents.stt.multi_speaker_adapter import (
    PrimarySpeakerDetectionOptions,
    _PrimarySpeakerDetector,
)

pytestmark = pytest.mark.unit

SAMPLE_RATE = 1000
SAMPLES_PER_FRAME = 100  # 100ms at 1000Hz, matches frame_size_ms below


def _constant_frame(value: int) -> rtc.AudioFrame:
    """Build a mono int16 frame where every sample equals ``value``."""
    return rtc.AudioFrame(
        data=value.to_bytes(2, "little", signed=True) * SAMPLES_PER_FRAME,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=SAMPLES_PER_FRAME,
    )


def _speech_data(speaker_id: str | None, start_time: float, end_time: float) -> SpeechData:
    return SpeechData(
        language="en",
        text="hello",
        start_time=start_time,
        end_time=end_time,
        speaker_id=speaker_id,
    )


def _detector() -> _PrimarySpeakerDetector:
    opts = PrimarySpeakerDetectionOptions(frame_size_ms=100, min_rms_samples=1)
    return _PrimarySpeakerDetector(primary_detection_options=opts)


def _push_speech(
    detector: _PrimarySpeakerDetector,
    speaker_id: str | None,
    amplitude: int,
    start_time: float,
) -> None:
    detector.push_audio(_constant_frame(amplitude))
    detector.push_audio(_constant_frame(amplitude))
    detector._update_primary_speaker(_speech_data(speaker_id, start_time, start_time + 0.2))


class TestPrimarySpeakerReset:
    def test_quiet_speaker_does_not_take_over(self) -> None:
        """Control: a much quieter speaker never becomes primary."""
        detector = _detector()

        _push_speech(detector, "A", amplitude=3000, start_time=0.0)
        assert detector._primary_speaker == "A"

        _push_speech(detector, "B", amplitude=50, start_time=0.2)
        assert detector._primary_speaker == "A"

    def test_unattributed_segment_keeps_primary_speaker(self) -> None:
        """A final transcript the diarizer could not attribute is not evidence
        that the primary speaker stopped being primary.
        """
        detector = _detector()

        _push_speech(detector, "A", amplitude=3000, start_time=0.0)
        assert detector._primary_speaker == "A"

        _push_speech(detector, None, amplitude=3000, start_time=0.2)
        assert detector._primary_speaker == "A"

        # Without the primary retained, B skips the loudness comparison entirely
        # and is crowned as "first speaker" despite being 60x quieter.
        _push_speech(detector, "B", amplitude=50, start_time=0.4)
        assert detector._primary_speaker == "A"

    def test_detect_primary_disabled_keeps_primary_unset(self) -> None:
        opts = PrimarySpeakerDetectionOptions(frame_size_ms=100, min_rms_samples=1)
        detector = _PrimarySpeakerDetector(
            detect_primary_speaker=False, primary_detection_options=opts
        )

        _push_speech(detector, "A", amplitude=3000, start_time=0.0)

        assert detector._primary_speaker is None
