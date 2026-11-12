"""Request-scoped language overrides across batch transcription plugins."""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from mistralai.client import Mistral

from livekit import rtc
from livekit.agents import APIConnectOptions, LanguageCode, http
from livekit.plugins.clova import STT as ClovaSTT
from livekit.plugins.mistralai import STT as MistralSTT


@pytest.mark.parametrize(
    ("provider", "default_language", "override", "expected_languages"),
    [
        pytest.param("clova", "en-US", "ko-KR", ["ko-KR", "en-US"], id="clova"),
        pytest.param(
            "mistralai",
            "en",
            "fr",
            ["fr", "en"],
            id="mistralai-default",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/pull/8054 reproduces: "
                "Mistral persists the per-call language override",
            ),
        ),
        pytest.param(
            "mistralai",
            None,
            "fr",
            ["fr", None],
            id="mistralai-auto",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="https://github.com/livekit/agents/pull/8054 reproduces: "
                "Mistral persists the override and disables later segment timestamps",
            ),
        ),
    ],
)
async def test_language_override_does_not_change_later_requests(
    provider: str,
    default_language: str | None,
    override: str,
    expected_languages: list[str | None],
) -> None:
    """Keep overrides local: https://github.com/livekit/agents/pull/8054 https://github.com/livekit/agents/issues/6096."""
    languages: list[object] = []
    timestamps: list[list[object]] = []

    async def transcribe(request: web.Request) -> web.Response:
        fields = await request.post()
        if provider == "clova":
            params = fields["params"]
            assert isinstance(params, str)
            languages.append(json.loads(params)["language"])
            return web.json_response({"text": "hello", "confidence": 0.99})
        languages.append(fields.get("language"))
        timestamps.append(list(fields.getall("timestamp_granularities", [])))
        return web.json_response(
            {
                "model": "voxtral-mini-latest",
                "text": "hello",
                "language": None,
                "usage": {},
                "segments": [{"text": "hello", "start": 0.25, "end": 0.75}],
            }
        )

    app = web.Application()
    app.router.add_post("/{path:.*}", transcribe)
    frame = rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=160)
    options = APIConnectOptions(max_retry=0)
    async with TestServer(app) as server, http.open() as session:
        url = str(server.make_url("")).rstrip("/")
        client = None
        model: ClovaSTT | MistralSTT
        if provider == "clova":
            assert default_language is not None
            model = ClovaSTT(
                secret="test", invoke_url=url, language=default_language, http_session=session
            )
        else:
            client = Mistral(api_key="test", server_url=url)
            model = (
                MistralSTT(client=client, language=default_language)
                if default_language is not None
                else MistralSTT(client=client)
            )
        try:
            first = await model.recognize(frame, language=override, conn_options=options)
            second = await model.recognize(frame, conn_options=options)
            assert languages == expected_languages
            assert first.alternatives[0].language.language == LanguageCode(override).language
            assert (
                second.alternatives[0].language.language
                == LanguageCode(default_language or "").language
            )
            assert [result.alternatives[0].text for result in (first, second)] == ["hello", "hello"]
            if provider == "mistralai":
                assert timestamps == [[], ["segment"] if default_language is None else []]
                for result in (first, second):
                    words = result.alternatives[0].words
                    assert words is not None
                    word = words[0]
                    assert (word.start_time, word.end_time) == (0.25, 0.75)
        finally:
            await model.aclose()
            if client is not None:
                await client.__aexit__(None, None, None)
                client.__exit__(None, None, None)
