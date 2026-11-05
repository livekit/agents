"""Unit tests for livekit.agents.inference.DecisionModel.

Covers credential resolution, the gateway request (URL, minted token, headers,
SystemOne body), answer decoding including the gateway's ``refusal`` answer,
the request id header, and error mapping for the gateway's error body.
"""

from __future__ import annotations

import contextlib
from typing import Any

import aiohttp
import jwt
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIError, APIStatusError
from livekit.agents.inference import DecisionModel
from livekit.agents.inference._utils import HEADER_USER_AGENT
from livekit.agents.llm import ChatContext, decisions
from livekit.agents.types import APIConnectOptions

pytestmark = pytest.mark.unit

QUESTIONS = {
    "human": decisions.Probability("The caller wants a human."),
    "intent": decisions.Choice("Intent?", options={"booking": "Reservation", "other": "Other"}),
    "frustration": decisions.Score("Frustration?", levels=["Calm", "Annoyed", "Angry"]),
}


def response_body() -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
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
        "usage": {"input_tokens": 50, "output_tokens": 0},
    }


@contextlib.asynccontextmanager
async def _gateway(handler):  # type: ignore[no-untyped-def]
    app = web.Application()
    app.router.add_post("/v1/decisions", handler)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession()
    try:
        yield str(server.make_url("/v1")), session
    finally:
        await session.close()
        await server.close()


def _model(base_url: str, session: aiohttp.ClientSession, **kwargs: Any) -> DecisionModel:
    defaults: dict[str, Any] = {
        "api_key": "test-key",
        "api_secret": "test-secret",
        "base_url": base_url,
        "http_session": session,
    }
    defaults.update(kwargs)
    return DecisionModel(**defaults)


def _json_handler(body: dict[str, Any], status: int = 200, captured: dict[str, Any] | None = None):  # type: ignore[no-untyped-def]
    async def handler(request: web.Request) -> web.Response:
        if captured is not None:
            captured["headers"] = dict(request.headers)
            captured["body"] = await request.json()
            captured["calls"] = captured.get("calls", 0) + 1
        return web.json_response(body, status=status, headers={"X-LiveKit-Request-Id": "req-1"})

    return handler


class TestCredentials:
    def test_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LIVEKIT_INFERENCE_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_INFERENCE_API_SECRET", raising=False)
        monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "env-secret")
        model = DecisionModel(base_url="https://x/v1")
        assert model._api_key == "env-key"
        assert model._api_secret == "env-secret"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "LIVEKIT_INFERENCE_API_KEY",
            "LIVEKIT_API_KEY",
            "LIVEKIT_INFERENCE_API_SECRET",
            "LIVEKIT_API_SECRET",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ValueError):
            DecisionModel(base_url="https://x/v1")

    def test_model_identity(self) -> None:
        model = DecisionModel(api_key="k", api_secret="s", base_url="https://x/v1")
        assert model.model == "openai/gpt-6-luna-decisions"
        assert model.provider == "livekit"
        assert model.capabilities.decision_kinds == ["probability", "choice", "score"]


async def test_request_targets_gateway_with_minted_token() -> None:
    captured: dict[str, Any] = {}
    async with _gateway(_json_handler(response_body(), captured=captured)) as (url, session):
        model = _model(url, session)
        context = ChatContext.empty()
        context.add_message(role="user", content="Please book a table.")
        response = await model.evaluate(input=context, decisions=QUESTIONS)

    headers = captured["headers"]
    scheme, token = headers["Authorization"].split(" ", 1)
    assert scheme == "Bearer"
    claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
    assert claims["iss"] == "test-key"
    assert claims["inference"] == {"perform": True}
    assert headers[HEADER_USER_AGENT].startswith("LiveKit Agents/")
    assert headers["Content-Type"] == "application/json"

    body = captured["body"]
    assert body == {
        "model": "openai/gpt-6-luna-decisions",
        "state": "user: Please book a table.",
        "questions": {
            "human": {"type": "noul", "instructions": "The caller wants a human."},
            "intent": {
                "type": "choice",
                "instructions": "Intent?",
                "criteria": {"booking": "Reservation", "other": "Other"},
            },
            "frustration": {
                "type": "score",
                "instructions": "Frustration?",
                "criteria": ["Calm", "Annoyed", "Angry"],
            },
        },
    }

    assert response.results["human"].value == 0.9
    assert response.results["intent"].value == "booking"
    assert response.results["intent"].provider_data["confidence"] == 0.5
    score = response.results["frustration"]
    assert score.kind == "score"
    assert score.value == 1.4
    assert score.probabilities == {0: 0.1, 1: 0.4, 2: 0.5}
    assert score.levels == ["Calm", "Annoyed", "Angry"]
    assert response.errors == {}
    assert response.model == "jev-1.13.0"
    assert response.provider == "livekit"
    assert response.input_tokens == 50
    assert response.output_tokens == 0


async def test_request_id_comes_from_gateway_header() -> None:
    async with _gateway(_json_handler(response_body())) as (url, session):
        response = await _model(url, session).evaluate(input="", decisions=QUESTIONS)
    assert response.request_id == "req-1"


async def test_model_falls_back_to_configured_model() -> None:
    body = response_body()
    del body["model"]
    async with _gateway(_json_handler(body)) as (url, session):
        model = _model(url, session, model="livekit/clef-omni")
        response = await model.evaluate(input="", decisions=QUESTIONS)
    assert response.model == "livekit/clef-omni"


async def test_positional_legend_is_accepted() -> None:
    # The gateway's OpenAI adapter sends the legend as an array, not keyed by index.
    body = response_body()
    body["answers"]["frustration"]["legend"] = ["Calm", "Annoyed", "Angry"]
    async with _gateway(_json_handler(body)) as (url, session):
        response = await _model(url, session).evaluate(input="", decisions=QUESTIONS)
    assert response.errors == {}
    assert response.results["frustration"].levels == ["Calm", "Annoyed", "Angry"]


@pytest.mark.parametrize("legend", [["Calm", "Annoyed"], ["Angry", "Annoyed", "Calm"]])
async def test_positional_legend_must_match_levels(legend: list[str]) -> None:
    body = response_body()
    body["answers"]["frustration"]["legend"] = legend
    async with _gateway(_json_handler(body)) as (url, session):
        response = await _model(url, session).evaluate(input="", decisions=QUESTIONS)
    assert set(response.errors) == {"frustration"}


async def test_refusal_is_a_per_decision_error() -> None:
    body = response_body()
    body["answers"]["intent"] = {"type": "refusal"}
    async with _gateway(_json_handler(body)) as (url, session):
        response = await _model(url, session).evaluate(input="", decisions=QUESTIONS)
    assert set(response.results) == {"human", "frustration"}
    assert "declined" in response.errors["intent"]


@pytest.mark.parametrize("failure", ["missing", "malformed", "wrong_kind", "unknown_label"])
async def test_invalid_answer_is_reported_without_dropping_the_batch(failure: str) -> None:
    body = response_body()
    if failure == "missing":
        del body["answers"]["intent"]
    elif failure == "malformed":
        body["answers"]["intent"] = {"type": "choice"}
    elif failure == "wrong_kind":
        body["answers"]["intent"] = {"type": "noul", "noul": 0.5}
    else:
        body["answers"]["intent"]["choice"] = "invented"
    captured: dict[str, Any] = {}
    async with _gateway(_json_handler(body, captured=captured)) as (url, session):
        response = await _model(url, session).evaluate(input="", decisions=QUESTIONS)
    assert set(response.results) == {"human", "frustration"}
    assert set(response.errors) == {"intent"}
    assert captured["calls"] == 1


async def test_gateway_error_body_is_surfaced_and_not_retried() -> None:
    body = {
        "type": "not_enabled",
        "error": "decisions are not enabled for this project",
        "hint": "",
    }
    captured: dict[str, Any] = {}
    async with _gateway(_json_handler(body, status=403, captured=captured)) as (url, session):
        with pytest.raises(APIStatusError) as error:
            await _model(url, session).evaluate(
                input="",
                decisions=QUESTIONS,
                conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
            )
    assert error.value.status_code == 403
    assert "not enabled" in error.value.message
    assert error.value.request_id == "req-1"
    assert error.value.body == body
    assert captured["calls"] == 1


async def test_capacity_error_is_retried() -> None:
    body = {"type": "provider_capacity", "error": "shared capacity", "hint": "retry later"}
    captured: dict[str, Any] = {}
    async with _gateway(_json_handler(body, status=503, captured=captured)) as (url, session):
        with pytest.raises(APIStatusError) as error:
            await _model(url, session).evaluate(
                input="",
                decisions=QUESTIONS,
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
            )
    assert error.value.status_code == 503
    assert captured["calls"] == 2


async def test_non_json_success_response_is_not_retried() -> None:
    calls = {"n": 0}

    async def handler(request: web.Request) -> web.Response:
        calls["n"] += 1
        return web.Response(text="<html>", content_type="text/html")

    async with _gateway(handler) as (url, session):
        with pytest.raises(APIError) as error:
            await _model(url, session).evaluate(
                input="",
                decisions=QUESTIONS,
                conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
            )
    assert not error.value.retryable
    assert calls["n"] == 1


async def test_unknown_answer_rejects_the_batch() -> None:
    body = response_body()
    body["answers"]["unexpected"] = {"type": "noul", "noul": 0.5}
    async with _gateway(_json_handler(body)) as (url, session):
        with pytest.raises(APIError):
            await _model(url, session).evaluate(input="", decisions=QUESTIONS)
