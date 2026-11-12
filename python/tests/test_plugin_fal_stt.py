from __future__ import annotations

import json

import httpx
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions
from livekit.plugins.fal import STT, stt as fal_stt

pytestmark = pytest.mark.unit


async def test_request_language_does_not_change_the_next_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    languages: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        languages.append(json.loads(request.content)["language"])
        return httpx.Response(200, json={"text": "hello"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(fal_stt.httpx, "AsyncClient", lambda **_: client)
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
