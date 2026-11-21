"""Regression tests for #7597: update_options(activation_threshold=...) must re-derive a
deactivation_threshold that was not set explicitly, the same way the VAD constructor does."""

from __future__ import annotations

from typing import Any

import pytest

from livekit.agents import inference
from livekit.plugins import silero
from livekit.plugins.silero import onnx_model

pytestmark = pytest.mark.unit

VAD_FACTORIES = [
    pytest.param(silero.VAD.load, id="silero"),
    pytest.param(inference.VAD, id="inference"),
]


@pytest.fixture(autouse=True)
def _fake_onnx_session(monkeypatch: pytest.MonkeyPatch) -> None:
    # the silero model is a Git LFS asset, and these tests only exercise option handling
    monkeypatch.setattr(onnx_model, "new_inference_session", lambda *args, **kwargs: object())


@pytest.mark.parametrize("activation_threshold", [0.7, 0.2])
@pytest.mark.parametrize("make_vad", VAD_FACTORIES)
def test_update_activation_threshold_matches_construction(
    make_vad: Any, activation_threshold: float
) -> None:
    expected = make_vad(activation_threshold=activation_threshold)._opts.deactivation_threshold

    vad = make_vad()
    vad.update_options(activation_threshold=activation_threshold)

    assert vad._opts.deactivation_threshold == expected
    assert vad._opts.deactivation_threshold < vad._opts.activation_threshold


@pytest.mark.parametrize("make_vad", VAD_FACTORIES)
def test_update_activation_threshold_keeps_explicit_deactivation_threshold(make_vad: Any) -> None:
    vad = make_vad(deactivation_threshold=0.2)
    vad.update_options(activation_threshold=0.7)
    assert vad._opts.deactivation_threshold == 0.2

    vad = make_vad()
    vad.update_options(deactivation_threshold=0.3)
    vad.update_options(activation_threshold=0.7)
    assert vad._opts.deactivation_threshold == 0.3


@pytest.mark.parametrize("make_vad", VAD_FACTORIES)
async def test_update_activation_threshold_updates_open_streams(make_vad: Any) -> None:
    expected = make_vad(activation_threshold=0.7)._opts.deactivation_threshold

    vad = make_vad()
    stream = vad.stream()
    try:
        vad.update_options(activation_threshold=0.7)
        assert stream._opts.deactivation_threshold == expected
    finally:
        await stream.aclose()
