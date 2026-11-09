from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit.agents import APIError, APIStatusError, utils
from livekit.agents.llm import ChatContext, decisions
from livekit.agents.types import APIConnectOptions
from livekit.plugins import vlmrun

pytestmark = pytest.mark.unit

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
