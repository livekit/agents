from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from google import genai
from google.genai import types

from livekit.agents import APIError, APIStatusError
from livekit.agents.llm import ChatContext, decisions
from livekit.agents.types import APIConnectOptions
from livekit.plugins.google import DecisionModel

pytestmark = pytest.mark.unit

QUESTIONS = {
    "human": decisions.Probability("The caller wants a human."),
    "intent": decisions.Choice("Intent?", options={"booking": "Reservation", "other": "Other"}),
    "frustration": decisions.Score("Frustration?", levels=["Calm", "Annoyed", "Angry"]),
}


def response_body() -> dict[str, Any]:
    return {
        "answers": {
            "human": {"type": "noul", "noul": 0.9, "confidence": 0.9},
            "intent": {
                "type": "choice",
                "choice": "booking",
                "confidence": 0.8,
                "probabilities": {"booking": 0.8, "other": 0.2},
            },
            "frustration": {
                "type": "score",
                "score": 1.4,
                "confidence": 0.5,
                "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
                "legend": {"0": "Calm", "1": "Annoyed", "2": "Angry"},
            },
        },
        "usage": {"total_input_tokens": 50, "total_output_tokens": 5},
    }


_handlers: list[Any] = []

# One client for the module: building a genai.Client loads the system certificates,
# which takes far longer than a unit test may. Each test swaps in its own handler.
_client = genai.Client(
    api_key="test-key",
    http_options=types.HttpOptions(
        httpx_async_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: _handlers[-1](request))
        ),
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)


def model_for(body: Any, status: int = 200) -> tuple[DecisionModel, list[Any]]:
    """A DecisionModel whose transport records every request and replays ``body``."""
    requests: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/decisions"
        requests.append(json.loads(request.content))
        # dumped by hand: httpx rejects the NaN a provider can still send
        return httpx.Response(
            status,
            content=json.dumps(body),
            headers={"content-type": "application/json", "x-request-id": "request-1"},
        )

    _handlers.append(handler)
    return DecisionModel(client=_client), requests


async def test_batch_is_sent_as_questions_keyed_by_name() -> None:
    model, requests = model_for(response_body())
    context = ChatContext.empty()
    context.add_message(role="user", content="Please book a table.")

    await model.evaluate(input=context, decisions=QUESTIONS)

    assert requests == [
        {
            "model": "gemini-decision-1",
            "input": "user: Please book a table.",
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
    ]


async def test_every_kind_is_decoded_with_distributions_and_usage() -> None:
    model, requests = model_for(response_body())
    metrics = []
    model.on("metrics_collected", metrics.append)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert len(requests) == 1
    human = response.results["human"]
    assert (human.value, human.provider_data["confidence"]) == (0.9, 0.9)
    choice = response.results["intent"]
    assert choice.kind == "choice"
    assert (choice.value, choice.probabilities) == ("booking", {"booking": 0.8, "other": 0.2})
    assert choice.provider_data["confidence"] == 0.8
    score = response.results["frustration"]
    assert score.kind == "score"
    assert (score.value, score.probabilities) == (1.4, {0: 0.1, 1: 0.4, 2: 0.5})
    assert score.levels == ["Calm", "Annoyed", "Angry"]
    assert response.errors == {}
    assert (response.model, response.provider) == ("gemini-decision-1", "google")
    assert response.request_id == "request-1"
    assert (response.input_tokens, response.output_tokens) == (50, 5)
    assert metrics[0].input_tokens == 50
    assert metrics[0].metadata.model_provider == "google"


@pytest.mark.parametrize(
    ("failure", "broken"),
    [
        ("missing", "human"),
        ("wrong_kind", "human"),
        ("unknown_type", "human"),
        ("non_numeric", "human"),
        ("nan", "human"),
        ("unknown_option", "intent"),
        ("out_of_range_score", "frustration"),
        ("shifted_levels", "frustration"),
        ("unindexed_levels", "frustration"),
        ("repeated_level", "frustration"),
    ],
)
async def test_invalid_answer_is_reported_per_decision(failure: str, broken: str) -> None:
    body = response_body()
    answers = body["answers"]
    if failure == "missing":
        del answers["human"]
    elif failure == "wrong_kind":
        answers["human"] = answers["intent"]
    elif failure == "unknown_type":
        answers["human"] = {"type": "rank", "order": ["a"]}
    elif failure == "non_numeric":
        answers["human"]["noul"] = "likely"
    elif failure == "nan":
        answers["human"]["noul"] = float("nan")
    elif failure == "unknown_option":
        answers["intent"]["choice"] = "invented"
    elif failure == "out_of_range_score":
        answers["frustration"]["score"] = 3.1
    elif failure == "shifted_levels":
        answers["frustration"]["probabilities"] = {"1": 0.1, "2": 0.4, "3": 0.5}
    elif failure == "unindexed_levels":
        answers["frustration"]["probabilities"] = {"low": 0.1, "mid": 0.4, "high": 0.5}
    else:
        # "00" names level 0 a second time and would replace its probability
        answers["frustration"]["probabilities"] = {"0": 0.1, "00": 0.9, "1": 0.4, "2": 0.5}
    model, requests = model_for(body)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert set(response.errors) == {broken}
    assert set(response.results) == QUESTIONS.keys() - {broken}
    assert response.input_tokens == 50
    assert len(requests) == 1


async def test_an_answer_for_an_unknown_decision_rejects_the_batch() -> None:
    body = response_body()
    body["answers"]["invented"] = {"type": "noul", "noul": 0.5, "confidence": 0.5}
    model, _ = model_for(body)

    with pytest.raises(APIError, match="unknown decisions"):
        await model.evaluate(input="", decisions=QUESTIONS)


async def test_blocked_input_rejects_the_batch_without_retrying() -> None:
    body = {"prompt_feedback": {"block_reason": "safety"}}
    model, requests = model_for(body)

    with pytest.raises(APIError, match="blocked the decision input: safety"):
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
        )

    assert len(requests) == 1


async def test_auth_errors_are_not_retried() -> None:
    model, requests = model_for({"error": {"message": "bad key"}}, status=401)

    with pytest.raises(APIStatusError) as error:
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
        )

    assert error.value.status_code == 401
    assert error.value.request_id == "request-1"
    assert len(requests) == 1


async def test_server_errors_obey_connection_options() -> None:
    model, requests = model_for({"error": {"message": "overloaded"}}, status=503)

    with pytest.raises(APIStatusError):
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )

    assert len(requests) == 2


def test_a_missing_api_key_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        DecisionModel()
