import math

import pytest

from livekit.plugins.silero import VAD as SileroVAD
from livekit.plugins.silero.vad import _VADOptions

# update_options / threshold-pair tests do not need the real ONNX session
pytestmark = pytest.mark.plugin("silero")


def _make_vad(activation: float = 0.5, deactivation: float = 0.35) -> SileroVAD:
    return SileroVAD(
        session=object(),  # never used by update_options
        opts=_VADOptions(
            min_speech_duration=0.05,
            min_silence_duration=0.55,
            prefix_padding_duration=0.5,
            max_buffered_speech=60.0,
            activation_threshold=activation,
            deactivation_threshold=deactivation,
            sample_rate=16000,
        ),
    )


def test_update_options_rederives_deactivation_default() -> None:
    """update_options(activation_threshold=x) without explicit deactivation must
    re-derive the paired default exactly like VAD.load does (issue #7597)."""
    vad = _make_vad()
    vad.update_options(activation_threshold=0.7)
    assert vad._opts.activation_threshold == 0.7
    assert math.isclose(vad._opts.deactivation_threshold, 0.55)


def test_update_options_low_activation_keeps_hysteresis() -> None:
    """deactivation must stay below activation (0.2 - 0.15 -> clamped to 0.05)."""
    vad = _make_vad()
    vad.update_options(activation_threshold=0.2)
    assert math.isclose(vad._opts.deactivation_threshold, 0.05)
    assert vad._opts.deactivation_threshold < vad._opts.activation_threshold


def test_update_options_explicit_deactivation_wins() -> None:
    vad = _make_vad()
    vad.update_options(activation_threshold=0.7, deactivation_threshold=0.4)
    assert vad._opts.activation_threshold == 0.7
    assert vad._opts.deactivation_threshold == 0.4


def test_update_options_deactivation_only_leaves_activation() -> None:
    vad = _make_vad()
    vad.update_options(deactivation_threshold=0.3)
    assert vad._opts.activation_threshold == 0.5
    assert vad._opts.deactivation_threshold == 0.3


async def test_update_options_propagates_to_open_stream() -> None:
    vad = _make_vad()
    stream = vad.stream()
    try:
        stream._input_sample_rate = 16000  # mark started so the resize branch runs
        import numpy as np

        stream._speech_buffer = np.empty(16000, dtype=np.int16)
        stream.update_options(activation_threshold=0.7)
        assert math.isclose(stream._opts.deactivation_threshold, 0.55)
    finally:
        await stream.aclose()


def test_with_options_shares_session_and_derives_defaults() -> None:
    """with_options (#7602) reuses the ONNX session; deactivation default re-derived
    from the new activation exactly like load()."""
    vad = _make_vad()
    vad2 = vad.with_options(activation_threshold=0.7)
    assert vad2._onnx_session is vad._onnx_session
    assert vad2._opts.activation_threshold == 0.7
    assert math.isclose(vad2._opts.deactivation_threshold, 0.55)
    # untouched options carry over
    assert vad2._opts.min_silence_duration == vad._opts.min_silence_duration
    # original is not mutated
    assert vad._opts.activation_threshold == 0.5
    assert math.isclose(vad._opts.deactivation_threshold, 0.35)


def test_with_options_rejects_invalid_deactivation() -> None:
    """`with_options` must enforce the same deactivation_threshold > 0 validation as `load`.

    Devin review (PR #7735): a zero exit threshold would keep an active stream in speech
    forever, since any positive probability exceeds it.
    """
    vad = _make_vad()
    with pytest.raises(ValueError):
        vad.with_options(deactivation_threshold=0)
    with pytest.raises(ValueError):
        vad.with_options(deactivation_threshold=-0.1)
    # an explicit bad pair must never slip through either
    with pytest.raises(ValueError):
        vad.with_options(activation_threshold=0.0, deactivation_threshold=0.0)
