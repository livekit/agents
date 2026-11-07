from __future__ import annotations

from typing import Any

import fal_client
import httpx
import pytest

from livekit import rtc
from livekit.agents import APIStatusError
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
    def __init__(self, error: Exception | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._error = error

    async def run(self, application: str, *, arguments: dict[str, Any], **kwargs: Any) -> Any:
        self.requests.append(arguments)
        if self._error is not None:
            raise self._error
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


async def test_recognize_does_not_retry_a_rejected_request() -> None:
    fal_stt = WizperSTT(api_key="test-key")
    fake_client = _FakeFalClient(
        fal_client.FalClientHTTPError(
            message="invalid key credentials",
            status_code=401,
            response_headers={},
            response=httpx.Response(401),
        )
    )
    fal_stt._fal_client = fake_client  # type: ignore[assignment]

    with pytest.raises(APIStatusError) as exc_info:
        await fal_stt.recognize([_frame()])

    assert len(fake_client.requests) == 1
    assert exc_info.value.status_code == 401
