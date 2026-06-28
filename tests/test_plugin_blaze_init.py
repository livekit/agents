"""Unit tests for Blaze plugin package exports."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_blaze_public_exports() -> None:
    from livekit.plugins import blaze

    assert blaze.__version__
    assert blaze.BlazeConfig is not None
    assert blaze.STT is not None
    assert blaze.TTS is not None
    assert blaze.LLM is not None
    assert blaze.ChunkedStream is not None
    assert blaze.LLMStream is not None
