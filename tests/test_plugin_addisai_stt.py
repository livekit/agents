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

import json

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt
from livekit.plugins import addisai

pytestmark = pytest.mark.plugin("addisai")


def _audio_frame() -> rtc.AudioFrame:
    samples = 1600
    return rtc.AudioFrame(
        data=b"\0\0" * samples,
        sample_rate=16_000,
        num_channels=1,
        samples_per_channel=samples,
    )


async def test_recognize_maps_addisai_response() -> None:
    async def recognize(request: web.Request) -> web.Response:
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["X-Addis-Client"].startswith("livekit-plugins-addisai/")

        reader = await request.multipart()
        fields: dict[str, bytes | str] = {}
        while part := await reader.next():
            if part.name == "audio":
                fields["audio"] = await part.read()
            elif part.name:
                if part.name == "request_data":
                    assert "Content-Type" not in part.headers
                fields[part.name] = await part.text()

        assert bytes(fields["audio"]).startswith(b"RIFF")
        assert json.loads(str(fields["request_data"])) == {"language_code": "am"}
        return web.json_response(
            {
                "status": "success",
                "data": {
                    "transcription": "ሰላም እንኳን ደህና መጣችሁ",
                    "usage_metadata": {
                        "totalBilledDuration": "1s",
                        "requestId": "request-123",
                    },
                },
                "confidence": 0.982,
            }
        )

    app = web.Application()
    app.router.add_post("/api/v2/stt", recognize)

    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        client = addisai.STT(
            language="am",
            api_key="test-key",
            base_url=str(server.make_url("/")).rstrip("/"),
            http_session=session,
        )
        event = await client.recognize(
            _audio_frame(),
            conn_options=APIConnectOptions(max_retry=0, timeout=2),
        )

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.request_id == "request-123"
    assert len(event.alternatives) == 1
    assert event.alternatives[0].text == "ሰላም እንኳን ደህና መጣችሁ"
    assert event.alternatives[0].language == "am"
    assert event.alternatives[0].confidence == pytest.approx(0.982)
    assert event.alternatives[0].metadata == {
        "usage": {
            "totalBilledDuration": "1s",
            "requestId": "request-123",
        }
    }


async def test_recognize_preserves_http_error() -> None:
    async def recognize(_: web.Request) -> web.Response:
        return web.json_response(
            {"error": {"code": "INVALID_API_KEY", "message": "invalid key"}},
            status=401,
            headers={"x-request-id": "failed-request"},
        )

    app = web.Application()
    app.router.add_post("/api/v2/stt", recognize)

    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        client = addisai.STT(
            api_key="test-key",
            base_url=str(server.make_url("/")).rstrip("/"),
            http_session=session,
        )
        with pytest.raises(APIStatusError) as exc_info:
            await client.recognize(
                _audio_frame(),
                conn_options=APIConnectOptions(max_retry=0, timeout=2),
            )

    assert exc_info.value.status_code == 401
    assert exc_info.value.request_id == "failed-request"
    assert exc_info.value.message == "invalid key"
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("language", ["en", "ha", "sw", ""])
def test_stt_rejects_unsupported_languages(language: str) -> None:
    with pytest.raises(ValueError, match="language must be one of"):
        addisai.STT(language=language, api_key="test-key")


async def test_tigrinya_batch_transcription() -> None:
    async def recognize(request: web.Request) -> web.Response:
        reader = await request.multipart()
        while part := await reader.next():
            if part.name == "request_data":
                assert json.loads(await part.text()) == {"language_code": "ti"}
        return web.json_response({"data": {"transcription": "ሰላም"}})

    app = web.Application()
    app.router.add_post("/api/v2/stt", recognize)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        client = addisai.STT(
            language="ti", api_key="test-key",
            base_url=str(server.make_url("/")).rstrip("/"), http_session=session,
        )
        event = await client.recognize(
            _audio_frame(), conn_options=APIConnectOptions(max_retry=0, timeout=2),
        )
    assert event.alternatives[0].language == "ti"
    assert event.alternatives[0].text == "ሰላም"


def test_rejects_invalid_request_timeout() -> None:
    with pytest.raises(ValueError, match="request_timeout must be positive"):
        addisai.STT(api_key="test-key", request_timeout=0)


async def test_scribe_recognize_uses_current_api() -> None:
    async def recognize(request: web.Request) -> web.Response:
        assert request.query["backend"] == "turbo"
        assert request.query["stream"] == "false"
        assert request.query["speakers"] == "true"
        reader = await request.multipart()
        part = await reader.next()
        assert part is not None and part.name == "audio"
        assert (await part.read()).startswith(b"RIFF")
        return web.json_response(
            {"data": {"text": "ሰላም", "usage": {"settled": True, "credits_used": 1}}}
        )

    app = web.Application()
    app.router.add_post("/api/v1/scribe/transcribe", recognize)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        client = addisai.ScribeSTT(
            backend="turbo",
            speakers=True,
            api_key="test-key",
            base_url=str(server.make_url("/")).rstrip("/"),
            http_session=session,
        )
        event = await client.recognize(
            _audio_frame(), conn_options=APIConnectOptions(max_retry=0, timeout=2)
        )

    assert client.capabilities.streaming is True
    assert client.capabilities.interim_results is True
    assert event.alternatives[0].text == "ሰላም"


def test_latest_realtime_adapters_expose_livekit_capabilities() -> None:
    tts_client = addisai.StreamingTTS(api_key="test-key")
    assert tts_client.capabilities.streaming is True
    assert tts_client.sample_rate == 44_100

    model = addisai.RealtimeModel(api_key="test-key")
    assert model.model == "አሌፍ-1.2-realtime-audio"
    assert model.capabilities.user_transcription is False
