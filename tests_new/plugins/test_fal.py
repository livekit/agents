"""fal plugin: batch STT request options and attempt ownership."""

from __future__ import annotations

import json

import httpx
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError
from livekit.plugins.fal import STT


async def test_request_language_does_not_change_the_next_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-call language applies to that call only; closing the STT closes its client."""
    languages: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        languages.append(json.loads(request.content)["language"])
        return httpx.Response(200, json={"text": "hello"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: client)
    model = STT(api_key="test", language="en")
    frame = rtc.AudioFrame(b"\x00\x00" * 160, 16000, 1, 160)
    options = APIConnectOptions(max_retry=0)
    try:
        first = await model.recognize(frame, language="fr", conn_options=options)
        second = await model.recognize(frame, conn_options=options)
        assert languages == ["fr", "en"]
        assert first.alternatives[0].language.language == "fr"
        assert second.alternatives[0].language.language == "en"
    finally:
        await model.aclose()
    assert client.is_closed


async def test_fal_failure_does_not_trigger_hidden_sdk_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed transcription is one request: the SDK makes no hidden retries."""
    from livekit.plugins.fal import WizperSTT

    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, text="busy")

    transport = httpx.MockTransport(respond)
    factory = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: factory(transport=transport, **kwargs)
    )
    model = WizperSTT(api_key="test")
    try:
        frame = rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=160)
        with pytest.raises(APIStatusError) as error:
            await model.recognize(frame, conn_options=APIConnectOptions(max_retry=0))
        assert error.value.status_code == 503
        assert len(requests) == 1
    finally:
        await model.aclose()
