from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from livekit.plugins.openai import responses, stt

pytestmark = pytest.mark.unit


class _FakeWebSocket:
    async def send_json(self, _data: object) -> None:
        pass


class _FakeSession:
    def __init__(self) -> None:
        self.url = ""
        self.kwargs: dict[str, object] = {}

    async def ws_connect(self, url: str, **kwargs: object) -> _FakeWebSocket:
        self.url = url
        self.kwargs = kwargs
        return _FakeWebSocket()


async def test_realtime_stt_connect_url_includes_model() -> None:
    model = "gateway-transcribe"
    instance = stt.STT(
        api_key="test-key",
        base_url="https://gateway.example.com/v1",
        model=model,
    )
    session = _FakeSession()
    instance._session = session

    await instance._connect_ws(5)

    parsed_url = urlparse(session.url)
    assert parsed_url.scheme == "wss"
    assert parsed_url.netloc == "gateway.example.com"
    assert parsed_url.path == "/v1/realtime"
    assert parse_qs(parsed_url.query) == {"intent": ["transcription"], "model": [model]}


async def test_native_realtime_stt_connect_url_omits_model() -> None:
    instance = stt.STT(api_key="test-key", model="gpt-realtime-whisper")
    session = _FakeSession()
    instance._session = session

    await instance._connect_ws(5)

    parsed_url = urlparse(session.url)
    assert parsed_url.scheme == "wss"
    assert parsed_url.netloc == "api.openai.com"
    assert parsed_url.path == "/v1/realtime"
    assert parse_qs(parsed_url.query) == {"intent": ["transcription"]}


async def test_responses_websocket_connect_url_includes_model() -> None:
    model = "gateway-responses"
    instance = responses.LLM(
        api_key="test-key",
        base_url="wss://gateway.example.com/v1/responses",
        model=model,
        use_websocket=True,
    )
    assert instance._ws is not None
    session = _FakeSession()
    instance._ws._session = session

    await instance._ws._create_ws(5)

    parsed_url = urlparse(session.url)
    assert parsed_url.scheme == "wss"
    assert parsed_url.netloc == "gateway.example.com"
    assert parsed_url.path == "/v1/responses"
    assert parse_qs(parsed_url.query) == {"model": [model]}


async def test_responses_websocket_connect_url_converts_http_scheme() -> None:
    model = "gateway-responses"
    instance = responses.LLM(
        api_key="test-key",
        base_url="https://gateway.example.com/v1/responses",
        model=model,
        use_websocket=True,
    )
    assert instance._ws is not None
    session = _FakeSession()
    instance._ws._session = session

    await instance._ws._create_ws(5)

    parsed_url = urlparse(session.url)
    assert parsed_url.scheme == "wss"
    assert parsed_url.netloc == "gateway.example.com"
    assert parsed_url.path == "/v1/responses"
    assert parse_qs(parsed_url.query) == {"model": [model]}


async def test_native_responses_websocket_connect_url_omits_model() -> None:
    instance = responses.LLM(api_key="test-key", model="gpt-4.1", use_websocket=True)
    assert instance._ws is not None
    session = _FakeSession()
    instance._ws._session = session

    await instance._ws._create_ws(5)

    parsed_url = urlparse(session.url)
    assert parsed_url.scheme == "wss"
    assert parsed_url.netloc == "api.openai.com"
    assert parsed_url.path == "/v1/responses"
    assert parse_qs(parsed_url.query) == {}


async def test_realtime_stt_socket_is_opened_with_a_heartbeat() -> None:
    """aiohttp defaults `heartbeat` to None, so a socket that dies without a
    FIN/RST leaves the receive side parked forever: the stream never raises, and
    neither the reconnect nor a FallbackAdapter switch is ever triggered."""
    instance = stt.STT(api_key="test-key", base_url="https://gateway.example.com/v1")
    session = _FakeSession()
    instance._session = session

    await instance._connect_ws(5)

    assert session.kwargs.get("heartbeat") == 30.0
