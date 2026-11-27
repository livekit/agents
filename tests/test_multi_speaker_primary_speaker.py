import pytest

from livekit import rtc
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
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


@pytest.mark.parametrize("speaker_id", ["A", "B"])
@pytest.mark.parametrize("suppress_background", [False, True])
def test_timed_endpoint_is_forwarded_without_formatting(
    speaker_id: str, suppress_background: bool
) -> None:
    detector = _PrimarySpeakerDetector(
        suppress_background_speaker=suppress_background,
        primary_detection_options=PrimarySpeakerDetectionOptions(min_rms_samples=1),
        primary_format="primary: {text}",
        background_format="background: {text}",
    )
    _push_speech(detector, "A", amplitude=3000, start_time=0.0)
    _push_speech(detector, "B", amplitude=50, start_time=0.2)

    final = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[_speech_data(speaker_id, 0.2, 0.4)],
    )
    updated_final = detector.on_stt_event(final)
    if speaker_id == "B" and suppress_background:
        assert updated_final is None
    else:
        assert updated_final is final
        prefix = "primary" if speaker_id == "A" else "background"
        assert final.alternatives[0].text == f"{prefix}: hello"

    final_text = final.alternatives[0].text
    endpoint = SpeechEvent(type=SpeechEventType.END_OF_SPEECH, alternatives=final.alternatives)

    assert detector.on_stt_event(endpoint) is endpoint
    assert endpoint.alternatives[0].end_time == 0.4
    assert final.alternatives[0].text == final_text


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
