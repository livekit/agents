from __future__ import annotations

import pytest

from livekit.plugins import slng

pytestmark = pytest.mark.unit


def test_public_imports_expose_client_facing_adapters() -> None:
    assert slng.STT.__name__ == "STT"
    assert slng.TTS.__name__ == "TTS"
