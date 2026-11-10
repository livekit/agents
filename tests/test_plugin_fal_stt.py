from __future__ import annotations

from typing import Any

import pytest

from livekit import rtc
from livekit.plugins.fal import WizperSTT

pytestmark = pytest.mark.plugin("fal")


def _frame(ms: int = 100, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = sample_rate * ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class _FakeFalClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def run(self, application: str, *, arguments: dict[str, Any], **kwargs: Any) -> Any:
        self.requests.append(arguments)
        return {"text": "hello"}


async def test_recognize_language_override_applies_to_one_call() -> None:
    fal_stt = WizperSTT(api_key="test-key", language="en")
    fake_client = _FakeFalClient()
    fal_stt._fal_client = fake_client  # type: ignore[assignment]

    overridden = await fal_stt.recognize([_frame()], language="fr")
    default = await fal_stt.recognize([_frame()])

    assert [r["language"] for r in fake_client.requests] == ["fr", "en"]
    assert overridden.alternatives[0].language == "fr"
    assert default.alternatives[0].language == "en"
    assert fal_stt._opts.language == "en"
