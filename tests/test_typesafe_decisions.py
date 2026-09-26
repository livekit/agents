from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit.agents import APIError, APIStatusError, decisions
from livekit.agents.llm import ChatContext
from livekit.agents.types import APIConnectOptions
from livekit.plugins import typesafe

pytestmark = pytest.mark.unit

QUESTIONS = {
    "human": decisions.Probability("The caller wants a human."),
    "intent": decisions.Choice("Intent?", options={"booking": "Reservation", "other": "Other"}),
    "frustration": decisions.Score("Frustration?", levels=["Calm", "Annoyed", "Angry"]),
}


def response_body():
    return {
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
                "confidence": 0.1,
            },
        },
        "usage": {"input_tokens": 50, "output_tokens": 5},
    }


def model_for(body, status=200):
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    response.status = status
    response.headers = {"x-request-id": "request-1"}
    response.json = AsyncMock(return_value=body)
    http_session = MagicMock(spec=aiohttp.ClientSession)
    http_session.post.return_value = response
    return typesafe.Jev.with_openrouter(api_key="test-key", http_session=http_session), http_session


async def test_jev_batches_all_kinds_and_preserves_distributions_and_usage() -> None:
    model, http = model_for(response_body())
    context = ChatContext.empty()
    context.add_message(role="user", content="Please book a table.")
    metrics = []
    model.on("metrics_collected", metrics.append)
    response = await model.evaluate(chat_ctx=context, decisions=QUESTIONS)
    http.post.assert_called_once()
    assert http.post.call_args.args[0] == "https://openrouter.ai/api/alpha/decisions"
    payload = http.post.call_args.kwargs["json"]
    assert payload["model"] == "typesafe/jev-1.13"
    assert payload["state"] == [{"role": "user", "content": "Please book a table."}]
    assert payload["questions"]["human"]["type"] == "noul"
    assert payload["questions"]["intent"]["criteria"] == QUESTIONS["intent"].options
    assert payload["questions"]["frustration"]["criteria"] == QUESTIONS["frustration"].levels
    assert response.results["human"].value == 0.9
    assert response.results["intent"].value == "booking"
    assert response.results["intent"].provider_data["confidence"] == 0.5
    score = response.results["frustration"]
    assert score.kind == "score"
    assert score.value == 1.4
    assert score.probabilities == {0: 0.1, 1: 0.4, 2: 0.5}
    assert response.request_id == "request-1"
    assert metrics[0].input_tokens == 50
    assert metrics[0].metadata.model_provider == "openrouter"


@pytest.mark.parametrize(
    "failure", ["missing", "extra", "wrong_kind", "unknown_label", "bad_sum", "bad_score", "nan"]
)
async def test_invalid_response_is_rejected_as_a_whole(failure: str) -> None:
    body = response_body()
    if failure == "missing":
        del body["answers"]["human"]
    elif failure == "extra":
        body["answers"]["unexpected"] = {"type": "noul", "noul": 0.5}
    elif failure == "wrong_kind":
        body["answers"]["human"] = body["answers"]["intent"]
    elif failure == "unknown_label":
        body["answers"]["intent"]["choice"] = "invented"
    elif failure == "bad_sum":
        body["answers"]["intent"]["probabilities"]["booking"] = 0.9
    elif failure == "bad_score":
        body["answers"]["frustration"]["score"] = 2.0
    else:
        body["answers"]["human"]["noul"] = float("nan")
    model, http = model_for(body)
    with pytest.raises(APIError):
        await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)
    http.post.assert_called_once()


async def test_auth_errors_are_not_retried() -> None:
    model, http = model_for({}, status=401)
    with pytest.raises(APIStatusError) as error:
        await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)
    assert error.value.status_code == 401
    http.post.assert_called_once()


async def test_retryable_errors_obey_connection_options() -> None:
    model, http = model_for({}, status=503)
    with pytest.raises(APIStatusError):
        await model.evaluate(
            chat_ctx=ChatContext.empty(),
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )
    assert http.post.call_count == 2
