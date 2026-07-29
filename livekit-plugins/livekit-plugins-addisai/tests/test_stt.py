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

pytestmark = pytest.mark.unit


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
    with pytest.raises(ValueError, match="language must be either"):
        addisai.STT(language=language, api_key="test-key")
