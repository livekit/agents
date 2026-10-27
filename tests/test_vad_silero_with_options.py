"""A VAD built with `with_options` must reuse the ONNX session and only override what it is given."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from livekit import rtc  # noqa: E402
from livekit.agents import vad as agents_vad  # noqa: E402
from livekit.plugins.silero import vad as silero_vad  # noqa: E402
from livekit.plugins.silero.vad import VAD, _VADOptions  # noqa: E402

pytestmark = pytest.mark.plugin("silero")

_SAMPLE_RATE = 16000
_WINDOW_SAMPLES = 512


def _vad(**overrides: Any) -> VAD:
    values: dict[str, Any] = {
        "min_speech_duration": 0.05,
        "min_silence_duration": 0.55,
        "prefix_padding_duration": 0.5,
        "max_buffered_speech": 60.0,
        "activation_threshold": 0.5,
        "deactivation_threshold": 0.35,
        "sample_rate": 16000,
        # as in `load`: a band the caller passes is a choice, not the derived default
        "deactivation_threshold_is_explicit": "deactivation_threshold" in overrides,
    }
    values.update(overrides)
    # Options are resolved before the session is used, so these cases need no model.
    return VAD(session=cast(Any, object()), opts=_VADOptions(**values))


def test_with_options_shares_the_onnx_session() -> None:
    """The whole point: a new VAD must not build a second session."""
    vad = _vad()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            silero_vad.onnx_model,
            "new_inference_session",
            lambda *args, **kwargs: pytest.fail("with_options built a new ONNX session"),
        )
        derived = vad.with_options(activation_threshold=0.6)

    assert derived._onnx_session is vad._onnx_session


def test_with_options_inherits_the_options_it_is_not_given() -> None:
    vad = _vad()

    derived = vad.with_options(min_silence_duration=1.2)

    assert derived._opts.min_silence_duration == pytest.approx(1.2)
    assert derived._opts.min_speech_duration == pytest.approx(vad._opts.min_speech_duration)
    assert derived._opts.prefix_padding_duration == pytest.approx(vad._opts.prefix_padding_duration)
    assert derived._opts.max_buffered_speech == pytest.approx(vad._opts.max_buffered_speech)
    assert derived._opts.activation_threshold == pytest.approx(vad._opts.activation_threshold)
    assert derived._opts.deactivation_threshold == pytest.approx(vad._opts.deactivation_threshold)
    assert derived._opts.sample_rate == vad._opts.sample_rate


def test_with_options_replaces_the_options_it_is_given() -> None:
    vad = _vad()

    derived = vad.with_options(
        min_speech_duration=0.1,
        min_silence_duration=0.2,
        prefix_padding_duration=0.3,
        max_buffered_speech=5.0,
        activation_threshold=0.8,
        deactivation_threshold=0.6,
        sample_rate=8000,
    )

    assert derived._opts.min_speech_duration == pytest.approx(0.1)
    assert derived._opts.min_silence_duration == pytest.approx(0.2)
    assert derived._opts.prefix_padding_duration == pytest.approx(0.3)
    assert derived._opts.max_buffered_speech == pytest.approx(5.0)
    assert derived._opts.activation_threshold == pytest.approx(0.8)
    assert derived._opts.deactivation_threshold == pytest.approx(0.6)
    assert derived._opts.sample_rate == 8000


def test_with_options_derives_deactivation_threshold_from_a_new_activation_threshold() -> None:
    """`load(activation_threshold=0.7)` derives 0.55; `with_options` must agree."""
    vad = _vad()

    derived = vad.with_options(activation_threshold=0.7)

    assert derived._opts.deactivation_threshold == pytest.approx(0.55)


def test_with_options_keeps_an_explicit_deactivation_threshold() -> None:
    vad = _vad(deactivation_threshold=0.2)

    derived = vad.with_options(activation_threshold=0.7, deactivation_threshold=0.4)

    assert derived._opts.deactivation_threshold == pytest.approx(0.4)


def test_with_options_keeps_the_inherited_deactivation_threshold() -> None:
    """Only a new activation threshold re-derives the hysteresis band."""
    vad = _vad(deactivation_threshold=0.45)

    derived = vad.with_options(min_silence_duration=0.9)

    assert derived._opts.deactivation_threshold == pytest.approx(0.45)


def test_with_options_keeps_a_chosen_deactivation_threshold_when_the_activation_changes() -> None:
    """A band the caller chose is theirs: only the derived one follows the activation threshold."""
    vad = _vad(deactivation_threshold=0.45)

    derived = vad.with_options(activation_threshold=0.7)

    assert derived._opts.deactivation_threshold == pytest.approx(0.45)


def test_with_options_keeps_a_chosen_band_that_matches_the_derived_one() -> None:
    """Provenance, not the value: 0.35 happens to equal load()'s default for 0.5, still a choice."""
    vad = _vad(activation_threshold=0.5, deactivation_threshold=0.35)

    derived = vad.with_options(activation_threshold=0.7)

    assert derived._opts.deactivation_threshold == pytest.approx(0.35)


def test_update_options_records_a_chosen_deactivation_threshold() -> None:
    vad = _vad()
    assert not vad._opts.deactivation_threshold_is_explicit

    vad.update_options(deactivation_threshold=0.3)

    assert vad._opts.deactivation_threshold_is_explicit
    assert vad.with_options(activation_threshold=0.7)._opts.deactivation_threshold == pytest.approx(
        0.3
    )


def test_with_options_keeps_the_type_of_a_subclass() -> None:
    class CustomVAD(VAD):
        pass

    vad = CustomVAD(session=cast(Any, object()), opts=_vad()._opts)

    assert type(vad.with_options(min_silence_duration=0.2)) is CustomVAD


def test_with_options_leaves_the_original_instance_untouched() -> None:
    vad = _vad()

    derived = vad.with_options(activation_threshold=0.9, min_silence_duration=1.5)
    derived.update_options(min_speech_duration=0.2)

    assert vad._opts.activation_threshold == pytest.approx(0.5)
    assert vad._opts.min_silence_duration == pytest.approx(0.55)
    assert vad._opts.min_speech_duration == pytest.approx(0.05)


def test_with_options_rejects_an_unsupported_sample_rate() -> None:
    vad = _vad()

    with pytest.raises(ValueError, match="8KHz and 16KHz"):
        vad.with_options(sample_rate=cast(Any, 44100))


def test_with_options_rejects_a_non_positive_deactivation_threshold() -> None:
    vad = _vad()

    with pytest.raises(ValueError, match="deactivation_threshold"):
        vad.with_options(deactivation_threshold=0.0)


async def test_streams_on_two_instances_share_the_session() -> None:
    """Sharing a session must keep per-stream model state separate."""
    vad = VAD.load(min_silence_duration=0.032)
    derived = vad.with_options(activation_threshold=0.6)

    streams = [vad.stream(), derived.stream()]
    try:
        for stream, value in zip(streams, (1, 0), strict=True):
            events = await _push_window(stream, value)
            assert any(e.type == agents_vad.VADEventType.INFERENCE_DONE for e in events)
    finally:
        for stream in streams:
            await stream.aclose()


async def _push_window(stream: agents_vad.VADStream, value: int) -> list[agents_vad.VADEvent]:
    stream.push_frame(
        rtc.AudioFrame(
            data=np.full(_WINDOW_SAMPLES, value, dtype=np.int16).tobytes(),
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=_WINDOW_SAMPLES,
        )
    )
    events = []
    while True:
        event = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
        events.append(event)
        if event.type == agents_vad.VADEventType.INFERENCE_DONE:
            return events


def test_with_options_keeps_the_band_when_update_options_moved_only_the_activation() -> None:
    """update_options can leave a band that is neither chosen nor derived; a copy keeps it."""
    vad = _vad()
    vad.update_options(activation_threshold=0.7)

    derived = vad.with_options(activation_threshold=0.8)

    assert derived._opts.deactivation_threshold == pytest.approx(0.35)


class _FakeModel:
    """Stands in for the ONNX model: these cases only exercise option handling."""

    sample_rate = _SAMPLE_RATE
    window_size_samples = _WINDOW_SAMPLES

    def predict(self, samples: np.ndarray) -> float:
        return 0.0

    __call__ = predict

    def reset(self) -> None:
        pass


async def test_a_band_chosen_on_a_stream_survives_a_copy() -> None:
    """Streams share the options object, so a threshold picked there is a chosen one."""
    vad = _vad()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(silero_vad.onnx_model, "OnnxModel", lambda **_: _FakeModel())
        stream = vad.stream()

    try:
        stream.update_options(deactivation_threshold=0.4)

        assert vad._opts.deactivation_threshold_is_explicit
        derived = vad.with_options(activation_threshold=0.7)
        assert derived._opts.deactivation_threshold == pytest.approx(0.4)
    finally:
        await stream.aclose()
