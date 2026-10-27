"""Silero VAD option updates must keep the deactivation threshold consistent with `load`."""

from __future__ import annotations

from typing import Any, cast

import pytest

pytest.importorskip("onnxruntime")

from livekit.plugins.silero.vad import VAD, VADStream, _VADOptions  # noqa: E402

pytestmark = pytest.mark.plugin("silero")


def _vad(**overrides: Any) -> VAD:
    values: dict[str, Any] = {
        "min_speech_duration": 0.05,
        "min_silence_duration": 0.55,
        "prefix_padding_duration": 0.5,
        "max_buffered_speech": 60.0,
        "activation_threshold": 0.5,
        "deactivation_threshold": 0.35,
        "sample_rate": 16000,
    }
    values.update(overrides)
    opts = _VADOptions(**values)
    # The session is only used for inference, so the option handling can be exercised without one.
    return VAD(session=cast(Any, None), opts=opts)


def test_update_options_derives_deactivation_threshold_from_activation() -> None:
    """load(activation_threshold=0.7) derives 0.55, and update_options must agree."""
    vad = _vad()

    vad.update_options(activation_threshold=0.7)

    assert vad._opts.activation_threshold == pytest.approx(0.7)
    assert vad._opts.deactivation_threshold == pytest.approx(0.55)


def test_update_options_keeps_an_explicit_deactivation_threshold() -> None:
    """A threshold the caller passed to load is theirs; only a derived one follows the activation threshold."""
    vad = _vad(deactivation_threshold=0.4, deactivation_threshold_is_explicit=True)

    vad.update_options(activation_threshold=0.7)

    assert vad._opts.deactivation_threshold == pytest.approx(0.4)


def test_update_options_records_a_later_explicit_deactivation_threshold() -> None:
    vad = _vad()

    vad.update_options(activation_threshold=0.7, deactivation_threshold=0.2)
    assert vad._opts.deactivation_threshold == pytest.approx(0.2)

    vad.update_options(activation_threshold=0.9)
    assert vad._opts.deactivation_threshold == pytest.approx(0.2)


def test_stream_update_options_derives_deactivation_threshold() -> None:
    """A stream shares the VAD's options object, so its own update path must derive the threshold too."""
    vad = _vad()
    stream = object.__new__(VADStream)
    stream._opts = vad._opts
    stream._input_sample_rate = 0

    stream.update_options(activation_threshold=0.7)

    assert vad._opts.deactivation_threshold == pytest.approx(0.55)
