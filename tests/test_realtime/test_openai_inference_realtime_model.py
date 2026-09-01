from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from livekit.plugins.openai.realtime import (
    InferenceRealtimeModel,
    inference_realtime_model as inference_realtime,
)
from livekit.plugins.openai.realtime.realtime_model import RealtimeSession

pytestmark = pytest.mark.unit


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        pass


class _FakeHTTPSession:
    def __init__(self) -> None:
        self.connections: list[tuple[str, dict[str, str], _FakeWebSocket]] = []

    async def ws_connect(self, *, url: str, headers: dict[str, str]) -> _FakeWebSocket:
        ws = _FakeWebSocket()
        self.connections.append((url, headers, ws))
        return ws


@pytest.fixture
def paused_realtime_main(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _paused_main(self: RealtimeSession) -> None:
        await self._msg_ch._close_ev.wait()

    monkeypatch.setattr(RealtimeSession, "_main_task", _paused_main)


def test_requires_provider_prefixed_model() -> None:
    with pytest.raises(ValueError, match="provider-prefixed"):
        InferenceRealtimeModel("gpt-realtime", api_key="key", api_secret="secret")


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("key", "api_key is required"),
        ("secret", "api_secret is required"),
    ],
)
def test_requires_livekit_credentials(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    message: str,
) -> None:
    for name in (
        "LIVEKIT_INFERENCE_API_KEY",
        "LIVEKIT_API_KEY",
        "LIVEKIT_INFERENCE_API_SECRET",
        "LIVEKIT_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    kwargs = {"api_key": "key", "api_secret": "secret"}
    kwargs[f"api_{missing}"] = None

    with pytest.raises(ValueError, match=message):
        InferenceRealtimeModel("openai/gpt-realtime", **kwargs)


def test_credentials_and_url_follow_inference_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVEKIT_INFERENCE_API_KEY", "inference-key")
    monkeypatch.setenv("LIVEKIT_INFERENCE_API_SECRET", "inference-secret")
    monkeypatch.setenv("LIVEKIT_INFERENCE_URL", "https://inference.example/v1")

    model = InferenceRealtimeModel("openai/gpt-realtime")

    assert model._inference_opts.api_key == "inference-key"
    assert model._inference_opts.api_secret == "inference-secret"
    assert model._opts.base_url == "https://inference.example/v1"


async def test_connection_refreshes_livekit_auth_and_custom_headers(
    monkeypatch: pytest.MonkeyPatch,
    paused_realtime_main: None,
) -> None:
    tokens = iter(("token-one", "token-two"))
    monkeypatch.setattr(
        inference_realtime,
        "create_access_token",
        lambda key, secret: f"{next(tokens)}:{key}:{secret}",
    )
    monkeypatch.setattr(
        inference_realtime,
        "get_inference_headers",
        lambda *, inference_class: {"X-Test-Class": inference_class or ""},
    )
    http_session = _FakeHTTPSession()
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        provider="openai",
        base_url="https://inference.example/v1",
        api_key="key",
        api_secret="secret",
        inference_class="priority",
        http_session=http_session,  # type: ignore[arg-type]
    )
    session = model.session()

    await session._create_ws_conn()
    await session._create_ws_conn()

    parsed = urlparse(http_session.connections[0][0])
    assert parsed.scheme == "wss"
    assert parsed.path == "/v1/realtime"
    assert parse_qs(parsed.query) == {"model": ["openai/gpt-realtime"]}
    assert [headers["Authorization"] for _, headers, _ in http_session.connections] == [
        "Bearer token-one:key:secret",
        "Bearer token-two:key:secret",
    ]
    assert http_session.connections[0][1]["X-Test-Class"] == "priority"
    assert http_session.connections[0][1]["X-LiveKit-Inference-Provider"] == "openai"
    await session.aclose()


async def test_initial_session_update_omits_gateway_model_field(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    event = session._msg_ch.recv_nowait()
    dumped = event.model_dump(exclude_unset=True) if hasattr(event, "model_dump") else event

    assert dumped["type"] == "session.update"
    assert dumped["session"]["type"] == "realtime"
    assert "model" not in dumped["session"]
    await session.aclose()


@pytest.mark.parametrize(
    "code",
    [
        "unsupported_transcription_model",
        "unsupported_audio_transport",
        "unsupported_audio_format",
        "invalid_audio_payload",
        "insufficient_quota",
    ],
)
async def test_gateway_configuration_errors_are_fatal(
    code: str,
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel(
        "openai/gpt-realtime",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    assert session._is_fatal_error(SimpleNamespace(code=code))
    await session.aclose()


def test_new_api_does_not_expose_deprecated_temperature() -> None:
    parameters = inspect.signature(InferenceRealtimeModel).parameters

    assert "temperature" not in parameters


async def test_xai_models_use_gateway_compatible_defaults(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel(
        "xai/grok-voice-latest",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    event = session._msg_ch.recv_nowait()
    dumped = event.model_dump(exclude_unset=True) if hasattr(event, "model_dump") else event

    assert model._opts.voice == "eve"
    assert dumped["session"]["audio"]["input"]["transcription"]["model"] == "grok-transcribe"
    assert dumped["session"]["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    await session.aclose()


async def test_xai_gateway_defaults_can_be_overridden(
    paused_realtime_main: None,
) -> None:
    model = InferenceRealtimeModel(
        "xai/grok-voice-latest",
        api_key="key",
        api_secret="secret",
        voice="Ara",
        input_audio_transcription=None,
        turn_detection=None,
    )
    session = model.session()

    event = session._msg_ch.recv_nowait()
    dumped = event.model_dump(exclude_unset=True) if hasattr(event, "model_dump") else event

    assert model._opts.voice == "Ara"
    assert dumped["session"]["audio"]["input"].get("transcription") is None
    assert dumped["session"]["audio"]["input"].get("turn_detection") is None
    await session.aclose()
