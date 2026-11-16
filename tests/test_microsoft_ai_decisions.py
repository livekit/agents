from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit.agents import APIError, APIStatusError, utils
from livekit.agents.llm import ChatContext, decisions
from livekit.agents.types import APIConnectOptions
from livekit.plugins import microsoft_ai

from .microsoft_ai_fakes import no_network as no_network

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

URL = "https://resource.example.invalid/providers/microsoft/v1/systemone"

QUESTIONS = {
    "human": decisions.Probability("The caller wants a human."),
    "intent": decisions.Choice("Intent?", options={"booking": "Reservation", "other": "Other"}),
    "frustration": decisions.Score("Frustration?", levels=["Calm", "Annoyed", "Angry"]),
}


def response_body() -> dict[str, Any]:
    return {
        "model": "Microsoft-Decision-1",
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
        "usage": {"input_tokens": 50, "output_tokens": 5},
    }


def model_for(
    body: Any, status: int = 200, **kwargs: Any
) -> tuple[microsoft_ai.DecisionModel, MagicMock]:
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    response.status = status
    response.headers = {"x-request-id": "request-1"}
    response.json = AsyncMock(return_value=body)
    response.text = AsyncMock(return_value=json.dumps(body))
    http_session = MagicMock(spec=aiohttp.ClientSession)
    http_session.post.return_value = response
    kwargs.setdefault("api_key", "test-key")
    model = microsoft_ai.DecisionModel(
        url=URL, model="decision-deployment", http_session=http_session, **kwargs
    )
    return model, http_session


async def test_batch_is_sent_to_the_deployment_with_an_api_key() -> None:
    model, http = model_for(response_body())
    context = ChatContext.empty()
    context.add_message(role="user", content="Please book a table.")

    await model.evaluate(input=context, decisions=QUESTIONS)

    http.post.assert_called_once()
    assert http.post.call_args.args[0] == URL
    assert http.post.call_args.kwargs["headers"]["api-key"] == "test-key"
    assert "Authorization" not in http.post.call_args.kwargs["headers"]
    payload = http.post.call_args.kwargs["json"]
    assert payload["model"] == "decision-deployment"
    assert payload["state"] == "user: Please book a table."
    assert payload["questions"]["human"]["type"] == "noul"
    assert payload["questions"]["intent"]["criteria"] == QUESTIONS["intent"].options
    assert payload["questions"]["frustration"]["criteria"] == QUESTIONS["frustration"].levels


async def test_entra_token_is_sent_as_bearer() -> None:
    model, http = model_for(response_body(), api_key="entra-token", auth_header="Authorization")
    await model.evaluate(input="", decisions=QUESTIONS)
    headers = http.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer entra-token"
    assert "api-key" not in headers


def test_configuration_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MICROSOFT_AI_DECISION_URL", URL)
    monkeypatch.setenv("MICROSOFT_AI_DECISION_MODEL", "env-deployment")
    monkeypatch.setenv("MICROSOFT_AI_DECISION_API_KEY", "env-key")
    model = microsoft_ai.DecisionModel()
    assert (model.model, model.provider) == ("env-deployment", "Microsoft AI")
    assert model._client.url == URL
    assert model._client.headers["api-key"] == "env-key"


@pytest.mark.parametrize(
    ("missing", "match"),
    [
        ("url", "MICROSOFT_AI_DECISION_URL"),
        ("model", "MICROSOFT_AI_DECISION_MODEL"),
        ("api_key", "MICROSOFT_AI_DECISION_API_KEY"),
    ],
)
def test_missing_configuration_fails_at_construction(missing: str, match: str) -> None:
    kwargs: dict[str, Any] = {"url": URL, "model": "decision-deployment", "api_key": "key"}
    del kwargs[missing]
    with pytest.raises(ValueError, match=match):
        microsoft_ai.DecisionModel(**kwargs)


def test_insecure_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="TLS"):
        microsoft_ai.DecisionModel(
            url="http://resource.example.invalid/providers/microsoft/v1/systemone",
            model="decision-deployment",
            api_key="key",
        )


async def test_every_kind_is_decoded_with_distributions_and_usage() -> None:
    model, _ = model_for(response_body())
    metrics = []
    model.on("metrics_collected", metrics.append)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert response.results["human"].value == 0.9
    assert response.results["intent"].value == "booking"
    assert response.results["intent"].provider_data["confidence"] == 0.5
    score = response.results["frustration"]
    assert score.kind == "score"
    assert (score.value, score.probabilities) == (1.4, {0: 0.1, 1: 0.4, 2: 0.5})
    assert score.levels == ["Calm", "Annoyed", "Angry"]
    assert response.errors == {}
    assert (response.model, response.provider) == ("Microsoft-Decision-1", "Microsoft AI")
    assert response.request_id == "request-1"
    assert (response.input_tokens, response.output_tokens) == (50, 5)
    assert metrics[0].input_tokens == 50
    assert metrics[0].metadata.model_provider == "Microsoft AI"


async def test_invalid_answer_is_reported_without_dropping_the_batch() -> None:
    body = response_body()
    body["answers"]["intent"]["choice"] = "invented"
    model, _ = model_for(body)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert set(response.errors) == {"intent"}
    assert set(response.results) == {"human", "frustration"}


async def test_unknown_answer_rejects_the_batch() -> None:
    body = response_body()
    body["answers"]["unexpected"] = {"type": "noul", "noul": 0.5}
    model, http = model_for(body)
    with pytest.raises(APIError):
        await model.evaluate(input="", decisions=QUESTIONS)
    http.post.assert_called_once()


async def test_auth_errors_are_not_retried() -> None:
    model, http = model_for({}, status=401)
    with pytest.raises(APIStatusError) as error:
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
        )
    assert error.value.status_code == 401
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


async def test_borrowed_session_is_not_closed() -> None:
    model, http = model_for(response_body())
    await model.aclose()
    http.close.assert_not_called()


async def test_remote_499_keeps_its_status_and_is_not_retried() -> None:
    model, http = model_for({}, status=499)
    with pytest.raises(APIStatusError) as error:
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
        )
    assert error.value.status_code == 499
    http.post.assert_called_once()


async def test_error_carries_the_foundry_body_and_request_id() -> None:
    body = {"error": {"code": "DeploymentNotFound", "message": "No such deployment."}}
    model, http = model_for(body, status=404)
    http.post.return_value.headers = {"apim-request-id": "apim-1"}
    with pytest.raises(APIStatusError) as error:
        await model.evaluate(input="", decisions=QUESTIONS)
    assert error.value.request_id == "apim-1"
    assert "DeploymentNotFound" in str(error.value.body)


async def test_default_session_comes_from_the_http_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, http = model_for(response_body())
    monkeypatch.setattr(utils.http_context, "http_session", lambda: http)
    model = microsoft_ai.DecisionModel(url=URL, model="decision-deployment", api_key="test-key")

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert set(response.results) == set(QUESTIONS)
    http.post.assert_called_once()
