from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit.agents import APIError, APIStatusError, APITimeoutError, utils
from livekit.agents.llm import ChatContext, decisions
from livekit.agents.types import APIConnectOptions
from livekit.plugins import perplexity
from livekit.plugins.perplexity import __version__

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

QUESTIONS = {
    "human": decisions.Probability("The caller wants a human."),
    "intent": decisions.Choice("Intent?", options={"booking": "Reservation", "other": "Other"}),
    "frustration": decisions.Score("Frustration?", levels=["Calm", "Annoyed", "Angry"]),
}


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)


def response_body() -> dict[str, Any]:
    # Shape of the documented /v1/decisions response: JSON object keys are strings.
    return {
        "model": "pplx-decider-flash-v1.3",
        "answers": {
            "human": {"type": "noul", "noul": 0.9},
            "intent": {
                "type": "choice",
                "choice": "booking",
                "confidence": 0.6,
                "probabilities": {"booking": 0.8, "other": 0.2},
            },
            "frustration": {
                "type": "score",
                "score": 1.4,
                "confidence": 0.4,
                "legend": {"0": "Calm", "1": "Annoyed", "2": "Angry"},
                "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
            },
        },
        "usage": {"input_tokens": 50, "output_tokens": 3},
    }


def model_for(
    body: Any, status: int = 200, **kwargs: Any
) -> tuple[perplexity.DecisionModel, MagicMock]:
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
    return perplexity.DecisionModel(http_session=http_session, **kwargs), http_session


async def test_batch_is_sent_as_one_request_with_bearer_auth_and_attribution() -> None:
    model, http = model_for(response_body())
    context = ChatContext.empty()
    context.add_message(role="system", content="Never shown to the decision model.")
    context.add_message(role="user", content="Please book a table.")

    await model.evaluate(input=context, decisions=QUESTIONS)

    http.post.assert_called_once()
    assert http.post.call_args.args[0] == "https://api.perplexity.ai/v1/decisions"
    headers = http.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["X-Pplx-Integration"] == f"livekit-agents/{__version__}"
    assert http.post.call_args.kwargs["json"] == {
        "model": "pplx-decider-flash-v1.3",
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


async def test_every_kind_is_decoded_from_string_keyed_distributions() -> None:
    model, _ = model_for(response_body())

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert response.errors == {}
    assert response.results["human"].value == 0.9
    intent = response.results["intent"]
    assert intent.kind == "choice"
    assert (intent.value, intent.probabilities) == ("booking", {"booking": 0.8, "other": 0.2})
    score = response.results["frustration"]
    assert score.kind == "score"
    assert score.value == 1.4
    assert score.probabilities == {0: 0.1, 1: 0.4, 2: 0.5}
    assert score.levels == ["Calm", "Annoyed", "Angry"]
    assert (response.model, response.provider, response.request_id) == (
        "pplx-decider-flash-v1.3",
        "Perplexity",
        "request-1",
    )


async def test_web_grounding_options_are_sent_when_configured() -> None:
    model, http = model_for(
        response_body(), model="pplx-decider-v1.3", mode="web_grounding", max_staleness=3600
    )
    await model.evaluate(input="https://example.com/menu", decisions=QUESTIONS)
    payload = http.post.call_args.kwargs["json"]
    assert (payload["model"], payload["mode"], payload["max_staleness"]) == (
        "pplx-decider-v1.3",
        "web_grounding",
        3600,
    )


def test_configuration_errors_fail_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
        perplexity.DecisionModel()
    monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key")
    with pytest.raises(ValueError, match="web_grounding"):
        perplexity.DecisionModel(max_staleness=60)


async def test_env_key_and_default_session_from_http_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, http = model_for(response_body())
    monkeypatch.setattr(utils.http_context, "http_session", lambda: http)
    monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key")

    response = await perplexity.DecisionModel().evaluate(input="", decisions=QUESTIONS)

    assert set(response.results) == set(QUESTIONS)
    assert http.post.call_args.kwargs["headers"]["Authorization"] == "Bearer env-key"


async def test_auth_error_is_not_retried() -> None:
    body = {"error": {"message": "Invalid API key provided.", "type": "invalid_api_key"}}
    model, http = model_for(body, status=401)
    with pytest.raises(APIStatusError) as error:
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
        )
    http.post.assert_called_once()
    assert error.value.status_code == 401
    assert error.value.request_id == "request-1"
    assert "invalid_api_key" in str(error.value.body)


@pytest.mark.parametrize("status", [429, 503])
async def test_transient_errors_obey_connection_options(status: int) -> None:
    model, http = model_for({"error": {"message": "slow down"}}, status=status)
    with pytest.raises(APIStatusError):
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )
    assert http.post.call_count == 2


async def test_transport_timeout_is_retried_then_raised_as_timeout() -> None:
    model, http = model_for(response_body())
    http.post.side_effect = TimeoutError()
    with pytest.raises(APITimeoutError):
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )
    assert http.post.call_count == 2


def _content_type_error() -> aiohttp.ContentTypeError:
    return aiohttp.ContentTypeError(MagicMock(), (), message="text/html")


@pytest.mark.parametrize(
    ("body", "json_error"),
    [
        pytest.param({"answers": []}, None, id="malformed_batch"),
        pytest.param(None, _content_type_error(), id="content_type"),
    ],
)
async def test_invalid_success_response_is_not_retried(
    body: Any, json_error: Exception | None
) -> None:
    model, http = model_for(body)
    if json_error is not None:
        http.post.return_value.json = AsyncMock(side_effect=json_error)
    with pytest.raises(APIError) as error:
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
        )
    assert not error.value.retryable
    http.post.assert_called_once()


async def test_custom_base_url_never_receives_the_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key")
    http_session = model_for(response_body())[1]
    model = perplexity.DecisionModel(base_url="http://127.0.0.1:8080/v1", http_session=http_session)

    await model.evaluate(input="", decisions=QUESTIONS)

    assert http_session.post.call_args.args[0] == "http://127.0.0.1:8080/v1/decisions"
    assert "Authorization" not in http_session.post.call_args.kwargs["headers"]

    explicit = perplexity.DecisionModel(
        base_url="http://127.0.0.1:8080/v1", api_key="gateway-key", http_session=http_session
    )
    await explicit.evaluate(input="", decisions=QUESTIONS)
    assert http_session.post.call_args.kwargs["headers"]["Authorization"] == "Bearer gateway-key"
