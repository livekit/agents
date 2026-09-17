from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from livekit.agents import inference, llm
from livekit.agents._exceptions import APIError
from livekit.agents.inference.realtime import gpt_live as inference_gpt_live
from livekit.agents.llm._realtime import gpt_live_types
from livekit.plugins.openai.realtime import (
    GPTLiveModel as DirectGPTLiveModel,
    GPTLiveSession as DirectGPTLiveSession,
)
from livekit.plugins.openai.tools import WebSearch

GPTLiveModel = inference.GPTLiveModel

pytestmark = pytest.mark.unit


def test_plugin_import_is_canonical_class() -> None:
    import livekit.plugins.openai.realtime as openai_realtime
    from livekit.plugins.openai.realtime import (
        InferenceGPTLiveModel,
        InferenceResponsesDelegationOptions,
        gpt_live_types as plugin_gpt_live_types,
        utils as plugin_realtime_utils,
    )
    from livekit.plugins.openai.realtime.inference_gpt_live_model import (
        InferenceGPTLiveSession,
    )

    assert InferenceGPTLiveModel is GPTLiveModel
    assert InferenceGPTLiveSession is inference_gpt_live.GPTLiveSession
    assert (
        InferenceResponsesDelegationOptions is inference_gpt_live.GPTLiveResponsesDelegationOptions
    )
    assert openai_realtime.gpt_live_types is plugin_gpt_live_types
    assert openai_realtime.utils is plugin_realtime_utils


class _FakeWebSocket:
    pass


class _FakeHTTPSession:
    def __init__(self) -> None:
        self.connections: list[tuple[str, dict[str, str]]] = []

    async def ws_connect(self, *, url: str, headers: dict[str, str]) -> _FakeWebSocket:
        self.connections.append((url, headers))
        return _FakeWebSocket()


@pytest.fixture
def paused_gpt_live_main(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _paused_main(self: DirectGPTLiveSession) -> None:
        await self._msg_ch._close_ev.wait()

    monkeypatch.setattr(DirectGPTLiveSession, "_main_task", _paused_main)


async def test_direct_model_keeps_openai_url_and_auth(paused_gpt_live_main: None) -> None:
    model = DirectGPTLiveModel(api_key="sk-openai")
    session = model.session()

    url, headers = session._create_ws_url_and_headers()
    parsed = urlparse(url)

    assert type(session) is DirectGPTLiveSession
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.openai.com"
    assert parsed.path == "/v1/live/sessions"
    assert parsed.query == ""
    assert headers == {
        "User-Agent": "LiveKit Agents",
        "Authorization": "Bearer sk-openai",
    }
    await session.aclose()
    await model.aclose()


def test_requires_provider_prefixed_model() -> None:
    with pytest.raises(ValueError, match="provider-prefixed"):
        GPTLiveModel("gpt-live-1", api_key="key", api_secret="secret")


@pytest.mark.parametrize("tier", ["auto", "flex", "priority", None])
def test_rejects_unpriced_inference_service_tiers(tier: object) -> None:
    with pytest.raises(ValueError, match="service_tier='default'"):
        GPTLiveModel(
            "openai/gpt-live-1",
            responses_options={"service_tier": tier},  # type: ignore[typeddict-item]
            api_key="key",
            api_secret="secret",
        )


def test_accepts_default_inference_service_tier() -> None:
    GPTLiveModel(
        "openai/gpt-live-1",
        responses_options={"service_tier": "default"},
        api_key="key",
        api_secret="secret",
    )


async def test_rejects_openai_hosted_tools(paused_gpt_live_main: None) -> None:
    model = GPTLiveModel(
        "openai/gpt-live-1",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    with pytest.raises(llm.RealtimeError, match="does not support OpenAI-hosted tools"):
        await session._update_tools([WebSearch()])

    await session.aclose()
    await model.aclose()


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
        GPTLiveModel("openai/gpt-live-1", **kwargs)


def test_credentials_and_url_follow_inference_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVEKIT_INFERENCE_API_KEY", "inference-key")
    monkeypatch.setenv("LIVEKIT_INFERENCE_API_SECRET", "inference-secret")
    monkeypatch.setenv("LIVEKIT_API_KEY", "fallback-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "fallback-secret")
    monkeypatch.setenv("LIVEKIT_INFERENCE_URL", "https://inference.example/v1")

    model = GPTLiveModel("openai/gpt-live-1")

    assert model._inference_opts.api_key == "inference-key"
    assert model._inference_opts.api_secret == "inference-secret"
    assert model._opts.base_url == "https://inference.example/v1"


def test_credentials_fall_back_to_livekit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVEKIT_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_INFERENCE_API_SECRET", raising=False)
    monkeypatch.setenv("LIVEKIT_API_KEY", "livekit-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "livekit-secret")

    model = GPTLiveModel("openai/gpt-live-1")

    assert model._inference_opts.api_key == "livekit-key"
    assert model._inference_opts.api_secret == "livekit-secret"


async def test_native_session_start_keeps_gateway_model_and_options(
    paused_gpt_live_main: None,
) -> None:
    model = GPTLiveModel(
        "openai/gpt-live-1",
        voice={"id": "voice_123"},
        responses_options={"model": "gpt-5.6-sol"},
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    event = session._session_start_event().model_dump(exclude_none=True)

    assert isinstance(session, inference_gpt_live.GPTLiveSession)
    assert model.model == "openai/gpt-live-1"
    assert model.provider == "livekit"
    assert event["type"] == "session.start"
    assert event["session"]["model"] == "openai/gpt-live-1"
    assert event["session"]["audio"]["output"]["voice"] == {"id": "voice_123"}
    assert event["session"]["delegation"]["responses"]["model"] == "gpt-5.6-sol"
    await session.aclose()
    await model.aclose()


async def test_default_native_gateway_url(paused_gpt_live_main: None) -> None:
    model = GPTLiveModel(
        "openai/gpt-live-1",
        base_url="https://agent-gateway.livekit.cloud/v1",
        api_key="key",
        api_secret="secret-with-at-least-32-bytes-for-jwt",
    )
    session = model.session()

    url, _ = session._create_ws_url_and_headers()
    parsed = urlparse(url)

    assert parsed.scheme == "wss"
    assert parsed.netloc == "agent-gateway.livekit.cloud"
    assert parsed.path == "/v1/live/sessions"
    assert parse_qs(parsed.query) == {"model": ["openai/gpt-live-1"]}
    await session.aclose()
    await model.aclose()


async def test_connection_uses_native_gateway_url_and_refreshes_auth(
    monkeypatch: pytest.MonkeyPatch,
    paused_gpt_live_main: None,
) -> None:
    tokens = iter(("token-one", "token-two"))
    monkeypatch.setattr(
        inference_gpt_live,
        "create_access_token",
        lambda key, secret: f"{next(tokens)}:{key}:{secret}",
    )
    monkeypatch.setattr(
        inference_gpt_live,
        "get_inference_headers",
        lambda *, inference_class: {"X-Test-Class": inference_class or ""},
    )
    http_session = _FakeHTTPSession()
    model = GPTLiveModel(
        "openai/gpt-live-1",
        provider="openai",
        base_url="https://inference.example/custom/v1/",
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
    assert parsed.netloc == "inference.example"
    assert parsed.path == "/custom/v1/live/sessions"
    assert parse_qs(parsed.query) == {"model": ["openai/gpt-live-1"]}
    assert [headers["Authorization"] for _, headers in http_session.connections] == [
        "Bearer token-one:key:secret",
        "Bearer token-two:key:secret",
    ]
    assert http_session.connections[0][1]["X-Test-Class"] == "priority"
    assert http_session.connections[0][1]["X-LiveKit-Inference-Provider"] == "openai"
    await session.aclose()
    await model.aclose()


@pytest.mark.parametrize(
    "code",
    [
        "invalid_event",
        "session_start_required",
        "invalid_session",
        "invalid_model",
        "invalid_delegated_model",
        "unsupported_delegated_model",
        "unsupported_server_tool",
        "unsupported_service_tier",
        "insufficient_quota",
    ],
)
async def test_gateway_rejections_are_fatal(
    code: str,
    paused_gpt_live_main: None,
) -> None:
    model = GPTLiveModel(
        "openai/gpt-live-1",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    assert session._is_fatal_error(gpt_live_types.ErrorBody(code=code))

    await session.aclose()
    await model.aclose()


async def test_transient_provider_error_remains_recoverable(
    paused_gpt_live_main: None,
) -> None:
    model = GPTLiveModel(
        "openai/gpt-live-1",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    assert not session._is_fatal_error(gpt_live_types.ErrorBody(code="server_error"))
    session._handle_session_started(
        gpt_live_types.SessionStartedEvent.construct(
            session=gpt_live_types.SessionResource.construct(id=None)
        )
    )
    assert session._session_id is None
    assert not session._is_fatal_error(gpt_live_types.ErrorBody(code="unsupported_delegated_model"))

    await session.aclose()
    await model.aclose()


async def test_gateway_rejection_stops_reconnects(
    paused_gpt_live_main: None,
) -> None:
    model = GPTLiveModel(
        "openai/gpt-live-1",
        api_key="key",
        api_secret="secret",
    )
    session = model.session()

    with pytest.raises(APIError) as exc_info:
        session._handle_error(gpt_live_types.ErrorBody(code="unsupported_delegated_model"))
    assert exc_info.value.retryable is False

    await session.aclose()
    await model.aclose()
