"""Hume payload, error and atomic option update regressions."""

import base64
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, APIStatusError, http
from livekit.plugins.hume import TTS
from livekit.plugins.hume.tts import AudioFormat, Utterance, VoiceById, VoiceProvider


@pytest.mark.parametrize("context", ["generation", [{"text": "Earlier."}]])
async def test_custom_voice_context_and_pcm_metadata(context: str | list[Utterance]) -> None:
    """Custom voices and context synthesize valid 48 kHz audio. https://github.com/livekit/agents/pull/2591 https://github.com/livekit/agents/issues/2619"""
    requests = []
    pcm = b"\x07\x00" * 9600

    async def speak(request: web.Request) -> web.StreamResponse:
        requests.append(await request.json())
        return web.Response(
            text=json.dumps({"audio": base64.b64encode(pcm).decode()}) + "\n",
            content_type="application/json",
        )

    app = web.Application()
    app.router.add_post("/v0/tts/stream/json", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            voice=VoiceById(id="custom-voice", provider=VoiceProvider.custom),
            context=context,
            audio_format=AudioFormat.pcm,
            base_url=str(server.make_url("")),
            http_session=session,
        )
        provider.update_options(speed=1.2)
        async with provider.synthesize(
            "Hello.", conn_options=APIConnectOptions(max_retry=0)
        ) as stream:
            frames = [ev async for ev in stream]
        audio = b"".join(bytes(ev.frame.data) for ev in frames)
        assert audio[: len(pcm)] == pcm
        assert not any(audio[len(pcm) :])
        assert all(ev.frame.sample_rate == 48000 for ev in frames)
        assert requests[0]["utterances"] == [
            {
                "text": "Hello.",
                "voice": {"id": "custom-voice", "provider": "CUSTOM_VOICE"},
                "speed": 1.2,
            }
        ]
        assert requests[0]["context"] == (
            {"generation_id": context} if isinstance(context, str) else {"utterances": context}
        )
        assert "sample_rate" not in requests[0]["format"]


@pytest.mark.parametrize("status", [400, 429, "stream"])
async def test_http_error_keeps_status_and_body(status: int | str) -> None:
    """HTTP failures retain status and provider diagnostics. https://github.com/livekit/agents/pull/2982"""

    async def speak(request: web.Request) -> web.StreamResponse:
        if isinstance(status, int):
            return web.Response(status=status, text="provider diagnostic")
        return web.Response(
            text=json.dumps({"type": "error", "message": "provider diagnostic"}) + "\n",
            content_type="application/json",
        )

    app = web.Application()
    app.router.add_post("/v0/tts/stream/json", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(api_key="key", base_url=str(server.make_url("")), http_session=session)
        async with provider.synthesize(
            "Hello.", conn_options=APIConnectOptions(max_retry=0)
        ) as stream:
            with pytest.raises(APIError) as error:
                _ = [ev async for ev in stream]
        if status == "stream":
            assert "provider diagnostic" in str(error.value.__cause__)
        else:
            assert isinstance(error.value, APIStatusError)
            assert error.value.status_code == status
            assert "provider diagnostic" in str(error.value.body)


@pytest.mark.parametrize("clear_voice", [False, True])
@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/8090 reproduces: update_options permits instant mode without a voice",
)
def test_invalid_instant_mode_update_is_rejected(clear_voice: bool) -> None:
    """Instant mode cannot be enabled without a voice through updates. https://github.com/livekit/agents/pull/8090"""
    provider = TTS(api_key="key") if clear_voice else TTS(api_key="key", voice=None)
    with pytest.raises(ValueError, match="voice"):
        if clear_voice:
            provider.update_options(speed=1.2, voice=None)
        else:
            provider.update_options(speed=1.2, instant_mode=True)
