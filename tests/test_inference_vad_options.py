"""The bundled livekit.agents.inference.VAD must derive its band the same way silero does."""

from __future__ import annotations

import pytest

pytest.importorskip("livekit.local_inference")

from livekit.agents.inference.vad import VAD  # noqa: E402

pytestmark = pytest.mark.unit


def thresholds(vad: VAD) -> tuple[float, float]:
    return vad._opts.activation_threshold, vad._opts.deactivation_threshold


def test_update_options_derives_the_band_from_a_new_activation_threshold() -> None:
    """load(activation_threshold=0.7) gives 0.55; update_options must agree (issue #7597)."""
    direct = VAD(activation_threshold=0.7)
    updated = VAD()
    updated.update_options(activation_threshold=0.7)

    assert thresholds(updated) == pytest.approx(thresholds(direct))


def test_update_options_keeps_a_band_the_caller_chose() -> None:
    vad = VAD(deactivation_threshold=0.4)

    vad.update_options(activation_threshold=0.7)

    assert vad._opts.deactivation_threshold == pytest.approx(0.4)


async def test_a_stream_follows_a_derived_band() -> None:
    """Streams hold their own snapshot, so their copy of update_options needs the same rule."""
    vad = VAD()
    stream = vad.stream()
    try:
        stream.update_options(activation_threshold=0.7)

        assert stream._opts.deactivation_threshold == pytest.approx(0.55)
    finally:
        await stream.aclose()
