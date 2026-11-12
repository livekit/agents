"""Spitch STT: request fields, against a local peer."""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.plugins.spitch import stt as spitch_stt


async def _transcribe_form(monkeypatch: pytest.MonkeyPatch, language: str) -> dict[str, str]:
    """Recognizes a frame in ``language`` and returns the form fields Spitch received."""
    received: list[dict[str, str]] = []

    async def transcribe(request: web.Request) -> web.Response:
        form = await request.post()
        received.append({key: value for key, value in form.items() if isinstance(value, str)})
        return web.json_response({"request_id": "r", "text": "hello", "segments": []})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", transcribe)
    async with TestServer(app) as server:
        monkeypatch.setenv("SPITCH_BASE_URL", str(server.make_url("")))
        monkeypatch.setenv("SPITCH_API_KEY", "test")
        model = spitch_stt.STT(language=language)
        event = await model.recognize(rtc.AudioFrame.create(16000, 1, 160))

    assert event.alternatives[0].text == "hello"
    return received[0]


@pytest.mark.parametrize(("language", "timestamp"), [("en", "word"), ("yo", None)])
async def test_transcribe_sends_a_valid_timestamp_granularity(
    monkeypatch: pytest.MonkeyPatch, language: str, timestamp: str | None
) -> None:
    """The timestamp form field is a granularity string or absent, never a boolean.

    https://github.com/livekit/agents/pull/4702
    """
    form = await _transcribe_form(monkeypatch, language)

    assert form.get("timestamp") == timestamp


@pytest.mark.xfail(strict=True, reason="the plugin picks a model but never sends it")
@pytest.mark.parametrize(("language", "model"), [("en", "mansa_v1"), ("yo", "legacy")])
async def test_transcribe_sends_the_model_chosen_for_the_language(
    monkeypatch: pytest.MonkeyPatch, language: str, model: str
) -> None:
    """English is transcribed by mansa_v1 (the model word timestamps are asked of), others by legacy."""
    form = await _transcribe_form(monkeypatch, language)

    assert form.get("model") == model
