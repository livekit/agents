"""Hermetic tests for Sarvam STT model capabilities (no network, no credentials).

Regression test for #6606: ``saaras:v3`` was marked ``supports_prompt=True``,
so the plugin sent a config/prompt message its endpoint ignores and callers
built hotword biasing on a parameter that was never wired to anything.
"""

import pytest

from livekit.plugins.sarvam.stt import _model_supports_prompt

pytestmark = pytest.mark.unit


def test_saaras_v3_does_not_support_prompt() -> None:
    assert _model_supports_prompt("saaras:v3") is False


def test_saaras_v25_still_supports_prompt() -> None:
    assert _model_supports_prompt("saaras:v2.5") is True
