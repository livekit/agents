import pytest
pytestmark = pytest.mark.unit

"""Tests for VAD update_options behavior, specifically deactivation_threshold recalculation."""

import pytest
from livekit.agents.inference import VAD as InferenceVAD


def test_inference_vad_update_options_recalculates_deactivation_threshold() -> None:
    """When activation_threshold is updated without explicit deactivation_threshold,
    deactivation_threshold should be recalculated using max(activation - 0.15, 0.01)."""
    vad = InferenceVAD(activation_threshold=0.5)
    # Initial: activation=0.5, deactivation=max(0.5-0.15, 0.01)=0.35
    assert vad._opts.activation_threshold == 0.5
    assert vad._opts.deactivation_threshold == pytest.approx(0.35)

    # Update to 0.7 - should recalculate deactivation to max(0.7-0.15, 0.01)=0.55
    vad.update_options(activation_threshold=0.7)
    assert vad._opts.activation_threshold == 0.7
    assert vad._opts.deactivation_threshold == pytest.approx(0.55)

    # Update to 0.2 - should recalculate deactivation to max(0.2-0.15, 0.01)=0.05
    vad.update_options(activation_threshold=0.2)
    assert vad._opts.activation_threshold == 0.2
    assert vad._opts.deactivation_threshold == pytest.approx(0.05)

    # Update to 0.01 (below 0.15) - should clamp to 0.01
    vad.update_options(activation_threshold=0.01)
    assert vad._opts.activation_threshold == 0.01
    assert vad._opts.deactivation_threshold == pytest.approx(0.01)


def test_inference_vad_update_options_with_explicit_deactivation() -> None:
    """When both activation_threshold and deactivation_threshold are provided,
    use the explicit deactivation_threshold."""
    vad = InferenceVAD(activation_threshold=0.5)
    assert vad._opts.deactivation_threshold == pytest.approx(0.35)

    # Provide explicit deactivation_threshold
    vad.update_options(activation_threshold=0.7, deactivation_threshold=0.5)
    assert vad._opts.activation_threshold == 0.7
    assert vad._opts.deactivation_threshold == 0.5  # explicit value used


def test_inference_vad_update_options_other_fields_unaffected() -> None:
    """Other fields should not affect deactivation_threshold when only they are updated."""
    vad = InferenceVAD(activation_threshold=0.5, min_speech_duration=0.1)
    assert vad._opts.deactivation_threshold == pytest.approx(0.35)
    assert vad._opts.min_speech_duration == 0.1

    # Update min_speech_duration - deactivation should remain unchanged
    vad.update_options(min_speech_duration=0.2)
    assert vad._opts.min_speech_duration == 0.2
    assert vad._opts.deactivation_threshold == pytest.approx(0.35)
    assert vad._opts.activation_threshold == 0.5


@pytest.mark.asyncio
async def test_inference_vad_stream_inherits_recalculated_thresholds() -> None:
    """VAD streams created after update_options should have the recalculated thresholds."""
    vad = InferenceVAD(activation_threshold=0.5)
    assert vad._opts.deactivation_threshold == pytest.approx(0.35)

    # Update threshold
    vad.update_options(activation_threshold=0.7)
    assert vad._opts.deactivation_threshold == pytest.approx(0.55)

    # Create new stream - should have updated thresholds
    stream = vad.stream()
    assert stream._opts.activation_threshold == 0.7
    assert stream._opts.deactivation_threshold == pytest.approx(0.55)
    await stream.aclose()


def test_inference_vad_update_options_activation_below_deactivation() -> None:
    """When activation_threshold is updated to a value below current deactivation_threshold,
    deactivation should be recalculated to be below activation (not stay higher)."""
    vad = InferenceVAD(activation_threshold=0.7)  # deactivation = 0.55
    assert vad._opts.deactivation_threshold == pytest.approx(0.55)

    # Update to 0.2 - old deactivation (0.55) > new activation (0.2)
    # This would be broken before the fix
    vad.update_options(activation_threshold=0.2)
    assert vad._opts.activation_threshold == 0.2
    assert vad._opts.deactivation_threshold == pytest.approx(0.05)  # recalculated, not 0.55
    assert vad._opts.deactivation_threshold < vad._opts.activation_threshold

# Silero plugin tests require the ONNX model, so they need the plugin mark
# They are in a separate test module to avoid loading the model for unit tests
