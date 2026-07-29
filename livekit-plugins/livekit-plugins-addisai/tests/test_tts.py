# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import base64

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions
from livekit.plugins import addisai

pytestmark = pytest.mark.unit


def _wav_audio() -> bytes:
    samples = 1600
    frame = rtc.AudioFrame(
        data=b"\0\0" * samples,
        sample_rate=16_000,
        num_channels=1,
        samples_per_channel=samples,
    )
    return rtc.combine_audio_frames([frame]).to_wav_bytes()


async def test_synthesize_downloads_clip_and_reuses_idempotency_key() -> None:
    requests: list[dict[str, object]] = []
    audio = _wav_audio()

    async def generate(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        if len(requests) == 1:
            return web.json_response(
                {"error": {"code": "TEMPORARY_ERROR", "message": "try again"}},
                status=503,
            )

        return web.json_response(
            {
                "data": {
                    "id": "clip-123",
                    "language": "om",
                    "voice_id": "om-test-voice",
                    "output_format": "pcm_16000",
                    "mime_type": "audio/wav",
                    "audio_url": f"{request.scheme}://{request.host}/audio.wav",
                }
            }
        )

    async def download(_: web.Request) -> web.Response:
        return web.Response(body=audio, content_type="audio/wav")

    app = web.Application()
    app.router.add_post("/api/v1/voice/generations", generate)
    app.router.add_get("/audio.wav", download)

    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        client = addisai.TTS(
            language="om",
            voice="om-test-voice",
            speed=1.0,
            api_key="test-key",
            base_url=str(server.make_url("/")).rstrip("/"),
            generation_timeout=5,
            download_timeout=5,
            http_session=session,
        )

        async with client.synthesize(
            "Akkam jirtu?",
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=2),
        ) as stream:
            events = [event async for event in stream]

    assert len(requests) == 2
    assert requests[0]["client_request_id"] == requests[1]["client_request_id"]
    assert requests[1] == {
        "text": "Akkam jirtu?",
        "voice_id": "om-test-voice",
        "language": "om",
        "output_format": "pcm_16000",
        "client_request_id": requests[1]["client_request_id"],
        "voice_settings": {"speed": 1.0},
    }
    assert events
    assert all(event.request_id == "clip-123" for event in events)
    assert sum(event.frame.duration for event in events) == pytest.approx(0.1)


async def test_synthesize_accepts_inline_audio_data_url() -> None:
    audio = _wav_audio()
    encoded_audio = base64.b64encode(audio).decode("ascii")

    async def generate(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "data": {
                    "id": "inline-clip",
                    "mime_type": "audio/wav",
                    "audio_url": f"data:audio/wav;base64,{encoded_audio}",
                }
            }
        )

    app = web.Application()
    app.router.add_post("/api/v1/voice/generations", generate)

    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        client = addisai.TTS(
            api_key="test-key",
            base_url=str(server.make_url("/")).rstrip("/"),
            generation_timeout=5,
            http_session=session,
        )
        async with client.synthesize(
            "ሰላም",
            conn_options=APIConnectOptions(max_retry=0, timeout=2),
        ) as stream:
            events = [event async for event in stream]

    assert events
    assert all(event.request_id == "inline-clip" for event in events)


@pytest.mark.parametrize("language", ["en", "ha", "sw", ""])
def test_tts_rejects_unsupported_languages(language: str) -> None:
    with pytest.raises(ValueError, match="language must be either"):
        addisai.TTS(language=language, api_key="test-key")
