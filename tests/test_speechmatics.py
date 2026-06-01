from livekit.plugins.speechmatics.stt import _extract_segment_confidence


def test_extract_segment_confidence_uses_segment_confidence() -> None:
    assert _extract_segment_confidence({"confidence": 0.42}) == 0.42


def test_extract_segment_confidence_averages_fragment_confidences() -> None:
    assert _extract_segment_confidence(
        {"fragments": [{"confidence": 0.2}, {"confidence": 0.8}, {"content": "."}]}
    ) == 0.5


def test_extract_segment_confidence_defaults_to_zero() -> None:
    assert _extract_segment_confidence({"text": "hello"}) == 0.0
