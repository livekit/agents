from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from livekit.agents import APIConnectOptions, APIError, APIStatusError
from livekit.plugins.aws import TTS as PollyTTS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("operation", "code", "attempts"),
    [
        ("SynthesizeSpeech", "InvalidSsmlException", 1),
        ("SynthesizeSpeech", "ThrottlingException", 4),
        ("AssumeRole", "Throttling", 4),
    ],
)
async def test_polly_400_fails_fast_throttling_retried(
    monkeypatch: pytest.MonkeyPatch, operation: str, code: str, attempts: int
) -> None:
    # Polly throttles with a 400, and request signing can raise STS's ClientErrors
    calls = 0

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def synthesize_speech(self, **kwargs: object) -> dict:
            nonlocal calls
            calls += 1
            raise ClientError(
                {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": 400}}, operation
            )

    instance = PollyTTS()
    monkeypatch.setattr(instance._session, "create_client", lambda *a, **kw: _Client())
    stream = instance.synthesize("hi", conn_options=APIConnectOptions(retry_interval=0))

    with pytest.raises(APIError) as exc_info:
        await stream.collect()

    assert calls == attempts
    if attempts == 1:
        assert isinstance(exc_info.value, APIStatusError)
        assert exc_info.value.status_code == 400
