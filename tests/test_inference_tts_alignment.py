from __future__ import annotations

import base64
import json
from collections import deque
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

import livekit.agents.inference.tts as inference_tts
from livekit.agents import APIConnectOptions
from livekit.agents.inference._utils import HEADER_SESSION_ID
from livekit.agents.inference.tts import TTS
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT

pytestmark = pytest.mark.unit


class _FakeWebSocket:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = deque(messages)
        self.sent: list[dict[str, Any]] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive(self, *, timeout: float) -> SimpleNamespace:
        if not self._messages:
            return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED)
        return SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(self._messages.popleft()),
        )


class _FakePool:
    def __init__(self, websocket: _FakeWebSocket, *, session_id: str | None = None) -> None:
        self._websocket = websocket
        self._session_id = session_id
        self.last_acquire_time = 0.0
        self.last_connection_reused = False

    def connection(self, *, timeout: float):  # noqa: ANN201, ARG002
        websocket = self._websocket
        session_id = self._session_id

        class _Context:
            async def __aenter__(self):  # noqa: ANN204
                return SimpleNamespace(ws=websocket, session_id=session_id)

            async def __aexit__(self, *_exc: object) -> bool:  # noqa: ANN204
                return False

        return _Context()


async def test_cartesia_inference_preserves_word_separators() -> None:
    websocket = _FakeWebSocket(
        [
            {"type": "session.created", "session_id": "session-1"},
            {
                "type": "output_audio",
                "audio": base64.b64encode(b"\0\0" * 2400).decode(),
            },
            {
                "type": "output_alignment",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.2},
                    {"word": "world.", "start": 0.2, "end": 0.5},
                ],
            },
            {"type": "done"},
        ]
    )
    tts = TTS(
        model="cartesia/sonic-3",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
        extra_kwargs={"add_timestamps": True},
    )
    tts._pool = _FakePool(websocket, session_id="inference_connection")  # type: ignore[assignment]

    async with tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0)) as stream:
        stream.push_text("hello world.")
        stream.end_input()
        events = [event async for event in stream]

    timed_transcript = [
        timed_text
        for event in events
        for timed_text in event.frame.userdata[USERDATA_TIMED_TRANSCRIPT]
    ]

    assert all(event.request_id.startswith("inference_connection_tts_") for event in events)
    assert list(map(str, timed_transcript)) == ["hello ", "world. "]
    assert [(word.start_time, word.end_time) for word in timed_transcript] == [
        (0.0, 0.2),
        (0.2, 0.5),
    ]


async def test_connection_retains_header_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = _FakeWebSocket([])

    class _FakeHTTPSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        async def ws_connect(self, url: str, *, headers: dict[str, str]) -> _FakeWebSocket:
            self.headers = headers
            return websocket

    http_session = _FakeHTTPSession()
    tts = TTS(
        model="cartesia/sonic-3",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
        http_session=http_session,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        inference_tts,
        "get_inference_headers",
        lambda: {HEADER_SESSION_ID: "inference_connection"},
    )
    monkeypatch.setattr(inference_tts, "create_access_token", lambda *_args: "token")

    connection = await tts._connect_ws(timeout=1.0)

    assert http_session.headers[HEADER_SESSION_ID] == "inference_connection"
    assert connection.session_id == "inference_connection"
    assert connection.ws is websocket
