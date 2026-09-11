"""Tests for `_model_supports_timestamps` model gating (the `with_timestamps` REST fix).

Sarvam's `with_timestamps` request field is only honored on the plain
`/speech-to-text` endpoint, not on the legacy `/speech-to-text-translate`
endpoint used by translate-mode models (e.g. saaras:v2.5).
"""

from __future__ import annotations

import pytest

from livekit.plugins.sarvam.stt import _model_supports_timestamps

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("saarika:v2.5", True),
        ("saaras:v2.5", False),  # legacy /speech-to-text-translate endpoint
        ("saaras:v3", True),
        ("some-future-model", True),  # unknown model falls back to the plain endpoint
    ],
)
def test_model_supports_timestamps(model: str, expected: bool) -> None:
    assert _model_supports_timestamps(model) is expected
