# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``update_options(sample_rate=...)`` must move the public ``TTS.sample_rate`` too.

``tts.TTS`` stores the rate as ``self._sample_rate`` and exposes it as the
``sample_rate`` property. The framework reads that property, not ``_opts`` -- most
importantly ``tts/fallback_adapter.py`` uses it as ``input_rate`` when it resamples
between a primary and a fallback TTS. A plugin that updates only ``_opts.sample_rate``
still hands its emitter the new rate, so the audio really is at the new rate while
``tts.sample_rate`` reports the old one, and the adapter resamples from the wrong
input rate.

``deepgram`` and ``vakyam`` already keep the two in sync.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _inworld():
    from livekit.plugins.inworld import TTS

    return TTS(api_key="test-key")


def _lmnt():
    from livekit.plugins.lmnt import TTS

    return TTS(api_key="test-key")


def _smallestai():
    from livekit.plugins.smallestai import TTS

    return TTS(api_key="test-key")


@pytest.mark.parametrize(
    ("factory", "new_rate"),
    [
        pytest.param(_inworld, 16000, id="inworld"),
        pytest.param(_lmnt, 16000, id="lmnt"),
        pytest.param(_smallestai, 16000, id="smallestai"),
    ],
)
def test_update_options_syncs_public_sample_rate(factory, new_rate: int) -> None:
    """The property the framework reads must follow the option that reaches the wire."""
    tts = factory()
    assert tts.sample_rate != new_rate, "pick a rate that differs from the default"

    tts.update_options(sample_rate=new_rate)

    assert tts._opts.sample_rate == new_rate
    assert tts.sample_rate == new_rate, (
        "TTS.sample_rate is what fallback_adapter resamples from; leaving it stale "
        "resamples the new audio at the old input rate"
    )


@pytest.mark.parametrize(
    ("factory"),
    [
        pytest.param(_inworld, id="inworld"),
        pytest.param(_lmnt, id="lmnt"),
        pytest.param(_smallestai, id="smallestai"),
    ],
)
def test_update_options_without_sample_rate_leaves_it_alone(factory) -> None:
    """An unrelated update must not disturb the rate."""
    tts = factory()
    before = tts.sample_rate

    tts.update_options()

    assert tts.sample_rate == before


def test_rate_change_after_a_fallback_adapter_already_wraps_the_tts() -> None:
    """The adapter may already exist when the rate changes.

    ``FallbackAdapter`` builds its resampler per request, reading ``tts.sample_rate``
    as ``input_rate`` at that moment -- so syncing the property is what makes an
    already-constructed adapter resample correctly afterwards. Without the sync the
    resampler is handed the old rate while the frames arrive at the new one.

    Note the adapter's own ``sample_rate`` (the resampler's ``output_rate``) and each
    status's ``needs_resampling`` are fixed at construction by design; this test pins
    the input side, which is the half the plugin controls.
    """
    from livekit.agents.tts.fallback_adapter import FallbackAdapter
    from livekit.plugins.smallestai import TTS

    primary = TTS(api_key="test-key", sample_rate=24000)
    secondary = TTS(api_key="test-key", sample_rate=16000)

    adapter = FallbackAdapter([primary, secondary])
    assert adapter.sample_rate == 24000

    secondary.update_options(sample_rate=8000)

    # what the emitter produces and what the resampler is told must agree
    assert secondary._opts.sample_rate == 8000
    assert secondary.sample_rate == 8000, (
        "FallbackAdapter builds its resampler per request with input_rate=tts.sample_rate; "
        "a stale property resamples the new-rate frames as if they were still old-rate"
    )
    # the adapter's output rate is construction-time by design, and unchanged here
    assert adapter.sample_rate == 24000
