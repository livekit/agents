from __future__ import annotations

import json
from typing import Any

import httpx
import openai
import pytest

from livekit.agents import APIError, APIStatusError
from livekit.agents.llm import ChatContext, decisions
from livekit.agents.types import APIConnectOptions
from livekit.plugins.openai import DecisionModel

pytestmark = pytest.mark.unit

QUESTIONS = {
    "human": decisions.Probability("The caller wants a human."),
    "intent": decisions.Choice("Intent?", options={"booking": "Reservation", "other": "Other"}),
    "frustration": decisions.Score("Frustration?", levels=["Calm", "Annoyed", "Angry"]),
}


def response_body() -> dict[str, Any]:
    return {
        "model": "gpt-6-luna-decisions-2026-10-01",
        "answers": [
            {"type": "predicate", "name": "human", "probability": 0.9},
            {
                "type": "choice",
                "name": "intent",
                "choice": "booking",
                "confidence": 0.5,
                "probabilities": [
                    {"value": "booking", "probability": 0.8},
                    {"value": "other", "probability": 0.2},
                ],
            },
            {
                "type": "score",
                "name": "frustration",
                "score": 1.4,
                "confidence": 0.1,
                "probabilities": [
                    {"value": 0, "label": "Calm", "probability": 0.1},
                    {"value": 1, "label": "Annoyed", "probability": 0.4},
                    {"value": 2, "label": "Angry", "probability": 0.5},
                ],
            },
        ],
        "usage": {
            "input_tokens": 50,
            "output_tokens": 5,
            "total_tokens": 55,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def model_for(body: Any, status: int = 200, **kwargs: Any) -> tuple[DecisionModel, list[Any]]:
    """A DecisionModel whose transport records every request and replays ``body``."""
    requests: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        # dumped by hand: httpx rejects the NaN a provider can still send
        return httpx.Response(
            status,
            content=json.dumps(body),
            headers={"content-type": "application/json", "x-request-id": "request-1"},
        )

    client = openai.AsyncClient(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return DecisionModel(client=client, **kwargs), requests


async def test_batch_is_sent_as_named_questions_in_order() -> None:
    model, requests = model_for(response_body())
    context = ChatContext.empty()
    context.add_message(role="user", content="Please book a table.")

    await model.evaluate(input=context, decisions=QUESTIONS)

    body = requests[0]
    assert body["model"] == "gpt-6-luna-decisions"
    assert body["input"] == "user: Please book a table."
    assert body["questions"] == [
        {"type": "predicate", "name": "human", "instructions": "The caller wants a human."},
        {
            "type": "choice",
            "name": "intent",
            "instructions": "Intent?",
            "choices": [
                {"value": "booking", "description": "Reservation"},
                {"value": "other", "description": "Other"},
            ],
        },
        {
            "type": "score",
            "name": "frustration",
            "instructions": "Frustration?",
            "levels": [{"label": "Calm"}, {"label": "Annoyed"}, {"label": "Angry"}],
        },
    ]
    assert "safety_identifier" not in body


async def test_safety_identifier_is_forwarded_when_given() -> None:
    model, requests = model_for(response_body(), safety_identifier="caller-7")
    await model.evaluate(input="", decisions=QUESTIONS)
    assert requests[0]["safety_identifier"] == "caller-7"


async def test_every_kind_is_decoded_with_distributions_and_usage() -> None:
    model, requests = model_for(response_body())
    metrics = []
    model.on("metrics_collected", metrics.append)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert len(requests) == 1
    assert response.results["human"].value == 0.9
    choice = response.results["intent"]
    assert choice.kind == "choice"
    assert (choice.value, choice.probabilities) == ("booking", {"booking": 0.8, "other": 0.2})
    assert choice.provider_data["confidence"] == 0.5
    score = response.results["frustration"]
    assert score.kind == "score"
    assert (score.value, score.probabilities) == (1.4, {0: 0.1, 1: 0.4, 2: 0.5})
    assert score.levels == ["Calm", "Annoyed", "Angry"]
    assert response.errors == {}
    assert (response.model, response.provider) == ("gpt-6-luna-decisions-2026-10-01", "openai")
    assert response.request_id == "request-1"
    assert (response.input_tokens, response.output_tokens) == (50, 5)
    assert metrics[0].input_tokens == 50
    assert metrics[0].metadata.model_name == response.model
    assert metrics[0].metadata.model_provider == "openai"


async def test_unnamed_answers_are_paired_with_the_questions_in_order() -> None:
    body = response_body()
    for answer in body["answers"]:
        del answer["name"]
    model, _ = model_for(body)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert set(response.results) == set(QUESTIONS)
    assert response.results["intent"].value == "booking"


@pytest.mark.parametrize("named_first", [True, False])
async def test_an_unnamed_answer_after_a_dropped_one_rejects_the_batch(named_first: bool) -> None:
    # Three same-kind decisions: an unnamed answer that shifted onto the wrong one would
    # be a plausible probability, so the batch has to be rejected rather than decoded.
    questions = {name: decisions.Probability(f"{name}?") for name in ("a", "b", "c")}
    answered = {"type": "predicate", "probability": 0.9}
    body = response_body()
    body["answers"] = [{**answered, "name": "a"} if named_first else answered, answered]
    model, _ = model_for(body)

    with pytest.raises(APIError, match="do not line up"):
        await model.evaluate(input="", decisions=questions)


async def test_a_named_answer_out_of_order_rejects_an_unnamed_batch() -> None:
    body = response_body()
    body["answers"][0]["name"] = "frustration"
    del body["answers"][1]["name"]
    del body["answers"][2]["name"]
    model, _ = model_for(body)

    with pytest.raises(APIError, match="do not line up"):
        await model.evaluate(input="", decisions=QUESTIONS)


async def test_refusal_is_reported_without_dropping_the_batch() -> None:
    body = response_body()
    body["answers"][0] = {"type": "refusal", "name": "human"}
    model, _ = model_for(body)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert set(response.results) == {"intent", "frustration"}
    assert response.errors["human"] == "the model declined this decision"


@pytest.mark.parametrize(
    ("failure", "broken"),
    [
        ("missing", "human"),
        ("wrong_kind", "human"),
        ("unknown_option", "intent"),
        ("boolean_choice", "intent"),
        ("out_of_range_score", "frustration"),
        ("shifted_levels", "frustration"),
        ("relabeled_levels", "frustration"),
        ("nan", "human"),
    ],
)
async def test_invalid_answer_is_reported_per_decision(failure: str, broken: str) -> None:
    body = response_body()
    human, intent, frustration = body["answers"]
    if failure == "missing":
        body["answers"].remove(human)
    elif failure == "wrong_kind":
        body["answers"][0] = {**intent, "name": "human"}
    elif failure == "unknown_option":
        intent["choice"] = "invented"
    elif failure == "boolean_choice":
        intent["choice"] = True
        intent["probabilities"][0]["value"] = True
    elif failure == "out_of_range_score":
        frustration["score"] = 3.1
    elif failure == "shifted_levels":
        for index, probability in enumerate(frustration["probabilities"]):
            probability["value"] = index + 1
    elif failure == "relabeled_levels":
        frustration["probabilities"][0]["label"] = "Serene"
    else:
        human["probability"] = float("nan")
    model, requests = model_for(body)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert set(response.errors) == {broken}
    assert set(response.results) == QUESTIONS.keys() - {broken}
    assert response.input_tokens == 50
    assert len(requests) == 1


async def test_an_answer_for_an_unknown_decision_rejects_the_batch() -> None:
    body = response_body()
    body["answers"].append({"type": "predicate", "name": "invented", "probability": 0.5})
    model, _ = model_for(body)

    with pytest.raises(APIError):
        await model.evaluate(input="", decisions=QUESTIONS)


async def test_a_repeated_answer_rejects_the_batch() -> None:
    body = response_body()
    body["answers"].append(body["answers"][0])
    model, _ = model_for(body)

    with pytest.raises(APIError):
        await model.evaluate(input="", decisions=QUESTIONS)


async def test_every_answer_missing_is_reported_per_decision() -> None:
    body = response_body()
    body["answers"] = []
    model, _ = model_for(body)

    response = await model.evaluate(input="", decisions=QUESTIONS)

    assert response.results == {}
    assert set(response.errors) == set(QUESTIONS)


async def test_auth_errors_are_not_retried() -> None:
    model, requests = model_for({"error": {"message": "bad key"}}, status=401)

    with pytest.raises(APIStatusError) as error:
        await model.evaluate(
            input="",
            decisions=QUESTIONS,
            conn_options=APIConnectOptions(max_retry=2, retry_interval=0),
        )

    assert error.value.status_code == 401
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


def test_an_empty_api_key_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        DecisionModel(api_key="")
