"""Both Gradium websockets identify the plugin to the API through `x-api-source`."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

pytestmark = pytest.mark.unit

# Pinned on purpose rather than imported: the value is what the API records, so a
# rename in the plugin must fail here.
API_SOURCE = "livekit-plugin"


@dataclass
class _Msg:
    type: Any
    data: Any = None
    extra: Any = None


class _TTSSocket:
    """Replays a minimal successful TTS session once the plugin sent its setup:
    ready, one audio chunk, end_of_stream, tagged with the setup's client_req_id."""

    def __init__(self) -> None:
        import aiohttp

        audio = base64.b64encode(bytes(48000)).decode()
        self._frames: list[Any] = [
            {"type": "ready", "request_id": "req", "model_name": "default"},
            {"type": "audio", "audio": audio},
            {"type": "end_of_stream"},
        ]
        self._ws_type = aiohttp.WSMsgType
        self.closed = False
        self.close_code: int | None = None
        self._setup = asyncio.Event()
        self._client_req_id: str | None = None

    async def send_str(self, data: str) -> None:
        msg = json.loads(data)
        if msg.get("type") == "setup":
            self._client_req_id = msg.get("client_req_id")
            self._setup.set()

    async def receive(self, timeout: float | None = None) -> _Msg:
        await self._setup.wait()
        if not self._frames:
            await asyncio.Event().wait()  # idle, like a live socket between replies
        frame = self._frames.pop(0)
        if self._client_req_id is not None:
            frame = {**frame, "client_req_id": self._client_req_id}
        return _Msg(self._ws_type.TEXT, json.dumps(frame))

    async def close(self) -> bool:
        self.closed = True
        return True

    async def __aenter__(self) -> _TTSSocket:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass


class _RecordingSession:
    """Records the kwargs of every ws_connect and hands out a replayed socket."""

    def __init__(self) -> None:
        self.connects: list[dict[str, Any]] = []

    async def ws_connect(self, url: str, **kwargs: Any) -> _TTSSocket:
        self.connects.append(kwargs)
        return _TTSSocket()


async def test_tts_sends_api_source_header():
    from livekit.plugins.gradium import TTS

    session = _RecordingSession()
    tts = TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]
    async for _ in tts.synthesize("hello", conn_options=APIConnectOptions(max_retry=0)):
        pass

    assert session.connects[0]["headers"] == {
        "x-api-key": "test-key",
        "x-api-source": API_SOURCE,
    }
    await tts.aclose()


async def test_tts_stream_sends_api_source_header():
    from livekit.plugins.gradium import TTS

    session = _RecordingSession()
    tts = TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]
    async with tts.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
        stream.push_text("hello ")
        stream.end_input()
        async for _ in stream:
            pass

    assert session.connects[0]["headers"]["x-api-source"] == API_SOURCE
    await tts.aclose()


async def test_stt_sends_api_source_header():
    from livekit.plugins.gradium import STT
    from livekit.plugins.gradium.stt import SpeechStream, STTOptions

    recorded: dict[str, Any] = {}

    class FakeWebSocket:
        async def send_str(self, message: str) -> None:
            pass

    class FakeSession:
        async def ws_connect(self, url: str, headers: dict[str, str]) -> FakeWebSocket:
            recorded["headers"] = headers
            return FakeWebSocket()

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    stt = STT(api_key="test-key")
    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=STTOptions(),
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            api_key="test-key",
            model_endpoint="wss://api.gradium.ai/api/speech/asr",
            model_name="default",
            http_session=FakeSession(),  # type: ignore[arg-type]
        )
    await stream._connect_ws()

    assert recorded["headers"] == {"x-api-key": "test-key", "x-api-source": API_SOURCE}
