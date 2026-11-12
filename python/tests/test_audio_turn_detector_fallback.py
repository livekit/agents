from __future__ import annotations

import pytest

from livekit.agents.inference import TurnDetector

pytestmark = pytest.mark.audio_eot


@pytest.mark.parametrize("credentials", [False, True])
def test_cloud_auto_selection_requires_credentials(
    monkeypatch: pytest.MonkeyPatch, credentials: bool
) -> None:
    monkeypatch.setenv("LIVEKIT_REMOTE_EOT_URL", "ws://gateway")
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    if credentials:
        monkeypatch.setenv("LIVEKIT_API_KEY", "key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")

    detector = TurnDetector()

    assert detector.model == ("turn-detector-v1" if credentials else "turn-detector-v1-mini")


def test_explicit_cloud_model_rejects_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LIVEKIT_REMOTE_EOT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError):
        TurnDetector(version="v1")
