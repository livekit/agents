from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx2 as httpx
import pytest

from livekit.agents import APIError, APIStatusError, llm, utils
from livekit.agents.llm import ChatContext, decisions
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins import vlmrun
from livekit.plugins.vlmrun import llm as vlmrun_llm

pytestmark = pytest.mark.unit

_STREAM_RESPONSE = b"""data: {"id":"c1","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"qwen/qwen3.5-0.8b","object":"chat.completion.chunk"}

data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"qwen/qwen3.5-0.8b","object":"chat.completion.chunk"}

data: [DONE]

"""

_PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_STREAM_RESPONSE,
            request=request,
        )


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> _Transport:
    t = _Transport()

    class _Client(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            kwargs.pop("limits", None)
            super().__init__(transport=t, **kwargs)

    monkeypatch.setattr(vlmrun_llm.httpx, "AsyncClient", _Client)
    monkeypatch.setenv("VLMRUN_API_KEY", "vlm-default")
    monkeypatch.delenv("VLMRUN_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("VLMRUN_GATEWAY_URL", raising=False)
    return t


async def _run(model: vlmrun.LLM, content: list[Any] | None = None) -> None:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content=content or ["hi"])
    try:
        stream = model.chat(chat_ctx=chat_ctx)
        try:
            async for _ in stream:
                pass
        finally:
            await stream.aclose()
    finally:
        await model.aclose()


async def test_default_route_and_headers(transport: _Transport) -> None:
    await _run(vlmrun.LLM())

    req = transport.requests[0]
    assert req.headers["authorization"] == "Bearer vlm-default"
    assert req.headers["x-vlmrun-client"].startswith("livekit-agents/")
    assert str(req.url) == "https://gateway.vlm.run/openai/chat/completions"
    assert json.loads(req.content)["model"] == "qwen/qwen3.8-27b", "default model"


async def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VLMRUN_API_KEY is required"):
        vlmrun.LLM()


async def test_api_key_sent_as_bearer(transport: _Transport) -> None:
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b", api_key="vlm-test"))

    req = transport.requests[0]
    assert req.headers["authorization"] == "Bearer vlm-test"
    assert req.headers["x-vlmrun-client"].startswith("livekit-agents/")


async def test_api_key_from_env(transport: _Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLMRUN_API_KEY", "vlm-env")
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b"))

    assert transport.requests[0].headers["authorization"] == "Bearer vlm-env"


async def test_base_url_override(transport: _Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLMRUN_GATEWAY_BASE_URL", "https://gw.example.com/")
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b"))
    assert str(transport.requests[0].url) == "https://gw.example.com/openai/chat/completions"

    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b", base_url="https://other.example.com"))
    assert str(transport.requests[1].url) == "https://other.example.com/openai/chat/completions"


async def test_unset_sampling_params_not_sent(transport: _Transport) -> None:
    await _run(vlmrun.LLM(model="google/diffusiongemma-26b-a4b-it"))

    body = json.loads(transport.requests[0].content)
    for field in ("temperature", "top_p", "seed", "tool_choice", "parallel_tool_calls"):
        assert field not in body
    assert body["model"] == "google/diffusiongemma-26b-a4b-it"


async def test_set_sampling_params_sent(transport: _Transport) -> None:
    await _run(vlmrun.LLM(model="qwen/qwen3.8-27b", temperature=0.2, top_p=0.9))

    body = json.loads(transport.requests[0].content)
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9


def test_unsupported_sampling_param_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        vlmrun.LLM(model="google/diffusiongemma-26b-a4b-it", temperature=0.5)


async def test_image_sent_as_image_url_part(transport: _Transport) -> None:
    await _run(
        vlmrun.LLM(model="qwen/qwen3.5-0.8b"),
        content=["What is this?", llm.ImageContent(image=_PIXEL)],
    )

    body = json.loads(transport.requests[0].content)
    parts = body["messages"][-1]["content"]
    image_parts = [p for p in parts if p["type"] == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"].startswith("data:image/")


def test_provider_and_model() -> None:
    model = vlmrun.LLM(model="qwen/qwen3.5-0.8b", api_key="vlm-test")
    assert model.model == "qwen/qwen3.5-0.8b"
    assert model.provider == "gateway.vlm.run"


async def _messages_sent(transport: _Transport, chat_ctx: llm.ChatContext) -> list[dict[str, Any]]:
    model = vlmrun.LLM(model="qwen/qwen3.5-0.8b")
    try:
        async with model.chat(chat_ctx=chat_ctx) as stream:
            async for _ in stream:
                pass
    finally:
        await model.aclose()
    return json.loads(transport.requests[-1].content)["messages"]


async def test_instructions_only_request_gets_placeholder_user_message(
    transport: _Transport,
) -> None:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="Greet the user.")
    messages = await _messages_sent(transport, chat_ctx)
    assert messages[-1] == {"role": "user", "content": "(empty)"}
    assert len(chat_ctx.items) == 1, "the caller's context is not modified"


async def test_placeholder_user_message_goes_first_not_after_the_greeting(
    transport: _Transport,
) -> None:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="Be terse.")
    chat_ctx.add_message(role="assistant", content="Hello! How can I help?")
    messages = await _messages_sent(transport, chat_ctx)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[1]["content"] == "(empty)"

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="assistant", content="Hello! How can I help?")
    messages = await _messages_sent(transport, chat_ctx)
    assert [m["role"] for m in messages] == ["user", "assistant"], "no system message to follow"


async def test_existing_user_message_is_left_alone(transport: _Transport) -> None:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="Be terse.")
    chat_ctx.add_message(role="user", content="hi")
    messages = await _messages_sent(transport, chat_ctx)
    assert [m["role"] for m in messages] == ["system", "user"]


async def test_later_instructions_keep_their_place(transport: _Transport) -> None:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="You can see the camera.")
    chat_ctx.add_message(role="user", content="hi")
    chat_ctx.add_message(role="assistant", content="hello")
    chat_ctx.add_message(role="system", content="Offer to describe the view.")
    messages = await _messages_sent(transport, chat_ctx)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == "You can see the camera.", "only the preamble is system"
    assert "Offer to describe the view." in messages[-1]["content"], "comes after the history"


async def test_base_url_with_sdk_v1_suffix(transport: _Transport, monkeypatch) -> None:
    monkeypatch.setenv("VLMRUN_GATEWAY_BASE_URL", "https://gw.internal/v1")
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b"))
    assert str(transport.requests[0].url) == "https://gw.internal/openai/chat/completions"


async def test_sdk_gateway_url_env(transport: _Transport, monkeypatch) -> None:
    monkeypatch.setenv("VLMRUN_GATEWAY_URL", "https://sdk.example.com/v1")
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b"))
    assert str(transport.requests[0].url) == "https://sdk.example.com/openai/chat/completions"

    monkeypatch.setenv("VLMRUN_GATEWAY_BASE_URL", "https://base.example.com")
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b"))
    assert str(transport.requests[1].url) == "https://base.example.com/openai/chat/completions", (
        "VLMRUN_GATEWAY_BASE_URL wins when both are set"
    )


async def _request_timeout(transport: _Transport, **chat_kwargs: Any) -> dict[str, Any]:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    model = vlmrun.LLM(model="qwen/qwen3.8-27b")
    try:
        async with model.chat(chat_ctx=chat_ctx, **chat_kwargs) as stream:
            async for _ in stream:
                pass
    finally:
        await model.aclose()
    return transport.requests[-1].extensions["timeout"]


async def test_default_request_timeout_outlasts_gateway_first_token_deadline(
    transport: _Transport,
) -> None:
    assert (await _request_timeout(transport))["read"] > 60, (
        "qwen/qwen3.8-27b may take up to 60 s to its first token"
    )
    session_default = await _request_timeout(transport, conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert session_default["read"] > 60, "an AgentSession passes the framework default"


async def test_explicit_conn_options_timeout_is_kept(transport: _Transport) -> None:
    timeout = await _request_timeout(transport, conn_options=APIConnectOptions(timeout=10.0))
    assert timeout["read"] == 10.0


async def test_base_url_with_openai_suffix(transport: _Transport) -> None:
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b", base_url="https://gateway.vlm.run/openai"))
    assert str(transport.requests[0].url) == "https://gateway.vlm.run/openai/chat/completions"


URL = "https://gateway.vlm.run/typesafe/v1/systemone"

QUESTIONS = {
    "human": decisions.Probability("The caller wants a human."),
    "intent": decisions.Choice("Intent?", options={"booking": "Reservation", "other": "Other"}),
    "frustration": decisions.Score("Frustration?", levels=["Calm", "Annoyed", "Angry"]),
}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    monkeypatch.delenv("VLMRUN_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("VLMRUN_GATEWAY_URL", raising=False)


def response_body() -> dict[str, Any]:
    return {
        "id": "sysone-1",
        "model": "qwen/qwen3.8-27b",
        "answers": {
            "human": {"type": "noul", "noul": 0.9},
            "intent": {
                "type": "choice",
                "choice": "booking",
                "probabilities": {"booking": 0.8, "other": 0.2},
                "confidence": 0.5,
            },
            "frustration": {
                "type": "score",
                "score": 1.4,
                "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
                "legend": {"0": "Calm", "1": "Annoyed", "2": "Angry"},
                "confidence": 0.1,
            },
        },
        "usage": {"input_tokens": 50, "output_tokens": 5, "reads": 1, "cost": 0.001},
    }


def model_for(
    body: Any, status: int = 200, **kwargs: Any
) -> tuple[vlmrun.DecisionModel, MagicMock]:
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    response.status = status
    response.headers = {"x-request-id": "request-1"}
    response.json = AsyncMock(return_value=body)
    response.text = AsyncMock(return_value=json.dumps(body))
    http_session = MagicMock(spec=aiohttp.ClientSession)
    http_session.post.return_value = response
    kwargs.setdefault("api_key", "vlm-test")
    return vlmrun.DecisionModel(http_session=http_session, **kwargs), http_session


async def test_batch_is_sent_to_system_one_with_a_bearer_key() -> None:
    model, http = model_for(response_body())
    context = ChatContext.empty()
    context.add_message(role="user", content="Please book a table.")

    await model.evaluate(input=context, decisions=QUESTIONS)

    http.post.assert_called_once()
    assert http.post.call_args.args[0] == URL
    headers = http.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer vlm-test"
    assert headers["X-VLMRun-Client"].startswith("livekit-agents/")
    payload = http.post.call_args.kwargs["json"]
    assert payload["model"] == "qwen/qwen3.8-27b", "default model"
    assert payload["state"] == "user: Please book a table."
    assert payload["questions"]["human"]["type"] == "noul"
    assert payload["questions"]["intent"]["criteria"] == QUESTIONS["intent"].options
    assert payload["questions"]["frustration"]["criteria"] == QUESTIONS["frustration"].levels


def test_missing_api_key_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="VLMRUN_API_KEY is required"):
        vlmrun.DecisionModel()


async def _decision_timeout(**evaluate_kwargs: Any) -> float:
    model, http_session = model_for(response_body())
    await model.evaluate(input="hi", decisions=QUESTIONS, **evaluate_kwargs)
    return http_session.post.call_args.kwargs["timeout"].total


async def test_decision_timeout_outlasts_the_gateway_deadline() -> None:
    assert await _decision_timeout() > 60
    assert await _decision_timeout(conn_options=DEFAULT_API_CONNECT_OPTIONS) > 60
    assert await _decision_timeout(conn_options=APIConnectOptions(timeout=10.0)) == 10.0


def test_ipv6_provider_keeps_its_brackets() -> None:
    model = vlmrun.DecisionModel(api_key="vlm-test", base_url="http://[::1]:8080")
    assert model.provider == "[::1]:8080"


def test_key_and_gateway_url_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLMRUN_API_KEY", "vlm-env")
    monkeypatch.setenv("VLMRUN_GATEWAY_BASE_URL", "https://user:secret@gw.example.com/v1")
    model = vlmrun.DecisionModel(model="qwen/qwen3.5-0.8b")
    assert (model.model, model.provider) == ("qwen/qwen3.5-0.8b", "gw.example.com")
    assert model._url == "https://user:secret@gw.example.com/typesafe/v1/systemone"
    assert model._headers["Authorization"] == "Bearer vlm-env"


async def test_every_kind_is_decoded_with_distributions_and_usage() -> None:
    model, _ = model_for(response_body())
    metrics = []
    model.on("metrics_collected", metrics.append)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert response.results["human"].value == 0.9
    assert response.results["intent"].value == "booking"
    score = response.results["frustration"]
    assert (score.value, score.probabilities) == (1.4, {0: 0.1, 1: 0.4, 2: 0.5})
    assert score.levels == ["Calm", "Annoyed", "Angry"]
    assert response.errors == {}
    assert (response.model, response.provider) == ("qwen/qwen3.8-27b", "gateway.vlm.run")
    assert response.request_id == "sysone-1"
    assert (response.input_tokens, response.output_tokens) == (50, 5)
    assert metrics[0].input_tokens == 50


async def test_invalid_answer_is_reported_without_dropping_the_batch() -> None:
    body = response_body()
    body["answers"]["intent"]["choice"] = "invented"
    model, _ = model_for(body)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert set(response.errors) == {"intent"}
    assert set(response.results) == {"human", "frustration"}


async def test_auth_errors_are_not_retried() -> None:
    model, http = model_for({"detail": "Invalid API key"}, status=401)
    with pytest.raises(APIStatusError) as error:
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
        )
    assert error.value.status_code == 401
    assert error.value.request_id == "request-1"
    assert "Invalid API key" in str(error.value.body)
    http.post.assert_called_once()


async def test_rate_limits_obey_connection_options() -> None:
    model, http = model_for({}, status=429)
    with pytest.raises(APIStatusError):
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )
    assert http.post.call_count == 2


async def test_non_json_success_response_is_not_retried() -> None:
    model, http = model_for({})
    http.post.return_value.json = AsyncMock(
        side_effect=aiohttp.ContentTypeError(MagicMock(), (), message="text/html")
    )
    with pytest.raises(APIError) as error:
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
        )
    assert not error.value.retryable
    http.post.assert_called_once()


async def test_default_session_comes_from_the_http_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, http = model_for(response_body())
    monkeypatch.setattr(utils.http_context, "http_session", lambda: http)
    model = vlmrun.DecisionModel(api_key="vlm-test")

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert set(response.results) == set(QUESTIONS)
    http.post.assert_called_once()
