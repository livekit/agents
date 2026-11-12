"""MiniMax oversized synthesis response regression."""

import json
from collections.abc import Callable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http
from livekit.plugins.minimax import TTS


async def test_oversized_stream_event_preserves_audio() -> None:
    """Large synthesis events deliver all PCM without line-buffer errors. https://github.com/livekit/agents/pull/4590"""
    pcm = b"\x02\x00" * 96000

    async def speak(request: web.Request) -> web.StreamResponse:
        return web.Response(
            text="data: "
            + json.dumps({"data": {"audio": pcm.hex()}, "base_resp": {"status_code": 0}})
            + "\n\n",
            content_type="text/event-stream",
            headers={"Trace-Id": "trace"},
        )

    app = web.Application()
    app.router.add_post("/v1/t2a_v2", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            audio_format="pcm",
            base_url=str(server.make_url("")),
            http_session=session,
        )
        try:
            async with provider.synthesize(
                "Hello.", conn_options=APIConnectOptions(max_retry=0)
            ) as stream:
                frames = [ev async for ev in stream]
            audio = b"".join(bytes(ev.frame.data) for ev in frames)
            assert audio[: len(pcm)] == pcm
            assert not any(audio[len(pcm) :])
        finally:
            await provider.aclose()


@pytest.mark.parametrize(
    "update",
    [
        lambda tts: tts.update_options(speed=2.5),
        lambda tts: tts.update_options(intensity=500),
        lambda tts: tts.update_options(timbre=-101),
        lambda tts: tts.update_options(emotion="fluent"),
    ],
    ids=["speed", "intensity", "timbre", "emotion"],
)
@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/8009 reproduces: invalid updates bypass constructor validation",
)
def test_option_updates_reject_invalid_synthesis_parameters(update: Callable[[TTS], None]) -> None:
    """Updates reject invalid synthesis parameters at the call site. https://github.com/livekit/agents/pull/8009"""
    provider = TTS(api_key="key", model="speech-02-hd")
    with pytest.raises(ValueError):
        update(provider)
