from __future__ import annotations

from livekit.plugins import slng


def test_public_imports_expose_client_facing_adapters() -> None:
    assert slng.STT.__name__ == "STT"
    assert slng.TTS.__name__ == "TTS"
