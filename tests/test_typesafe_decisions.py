from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit.agents import APIError, APIStatusError, decisions
from livekit.agents.llm import (
    AgentConfigUpdate,
    AgentHandoff,
    AudioContent,
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
)
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
        "model": "typesafe/jev-1.13.0",
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
    assert response.model == "typesafe/jev-1.13.0"
    assert response.provider == "openrouter"
    assert response.errors == {}
    assert metrics[0].input_tokens == 50
    assert metrics[0].metadata.model_name == response.model
    assert metrics[0].metadata.model_provider == "openrouter"


@pytest.mark.parametrize(
    ("value", "probabilities"),
    [
        # Captured Jev response: score 1.07 with probabilities implying 1.08.
        (1.07, {"0": 0.0, "1": 0.92, "2": 0.08}),
        (1.0, {"0": 0.33, "1": 0.33, "2": 0.33}),
        (1.4, {"0": 0.01, "1": 0.59, "2": 0.41}),
    ],
)
async def test_rounded_score_and_distribution_preserve_the_whole_batch(
    value, probabilities
) -> None:
    body = response_body()
    body["answers"]["frustration"].update(score=value, probabilities=probabilities)
    model, http = model_for(body)
    response = await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)
    assert response.results.keys() == QUESTIONS.keys()
    result = response.results["frustration"]
    assert result.kind == "score"
    assert result.value == value
    assert result.probabilities == {int(key): p for key, p in probabilities.items()}
    http.post.assert_called_once()


async def test_rounded_choice_distribution_is_preserved() -> None:
    body = response_body()
    body["answers"]["intent"]["probabilities"] = {"booking": 0.66, "other": 0.33}
    model, _ = model_for(body)
    response = await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)
    result = response.results["intent"]
    assert result.kind == "choice"
    assert result.probabilities == {"booking": 0.66, "other": 0.33}


async def test_five_level_score_rounding_includes_level_weights() -> None:
    # Independently rounded from [0.2049, 0.2049, 0.2049, 0.2049, 0.1804]
    # and expected score 1.951. The reported probabilities imply 1.92.
    body = response_body()
    body["answers"]["frustration"].update(
        score=1.95, probabilities={"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.18}
    )
    questions = QUESTIONS | {
        "frustration": decisions.Score(
            "Frustration?", levels=["Calm", "Uneasy", "Annoyed", "Angry", "Furious"]
        )
    }
    body["answers"]["frustration"]["legend"] = dict(enumerate(questions["frustration"].levels))
    model, _ = model_for(body)
    response = await model.evaluate(chat_ctx=ChatContext.empty(), decisions=questions)
    assert response.results.keys() == questions.keys()
    assert response.results["frustration"].value == 1.95


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "extra",
        "wrong_kind",
        "unknown_label",
        "bad_sum",
        "bad_score",
        "out_of_range_score",
        "nan",
    ],
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
    elif failure == "out_of_range_score":
        body["answers"]["frustration"].update(
            score=2.01, probabilities={"0": 0.0, "1": 0.0, "2": 1.0}
        )
    else:
        body["answers"]["human"]["noul"] = float("nan")
    model, http = model_for(body)
    with pytest.raises(APIError):
        await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)
    http.post.assert_called_once()


@pytest.mark.parametrize("allow_partial", [False, True])
async def test_auth_errors_are_not_retried(allow_partial: bool) -> None:
    model, http = model_for({}, status=401)
    with pytest.raises(APIStatusError) as error:
        await model.evaluate(
            chat_ctx=ChatContext.empty(), decisions=QUESTIONS, allow_partial=allow_partial
        )
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


@pytest.mark.parametrize("failure", ["missing", "malformed", "wrong_kind", "bad_score"])
async def test_partial_results_preserve_valid_answers_and_usage(failure: str) -> None:
    body = response_body()
    if failure == "missing":
        del body["answers"]["frustration"]
    elif failure == "malformed":
        body["answers"]["frustration"] = {"type": "score"}
    elif failure == "wrong_kind":
        body["answers"]["frustration"] = {"type": "noul", "noul": 0.5}
    else:
        body["answers"]["frustration"]["score"] = 2.0
    model, http = model_for(body)
    response = await model.evaluate(
        chat_ctx=ChatContext.empty(), decisions=QUESTIONS, allow_partial=True
    )
    assert set(response.results) == {"human", "intent"}
    assert set(response.errors) == {"frustration"}
    assert response.errors["frustration"]
    assert response.input_tokens == 50
    http.post.assert_called_once()


@pytest.mark.parametrize("legend", [None, {"0": "Angry", "1": "Annoyed", "2": "Calm"}])
async def test_missing_or_mismatched_legend_is_rejected(legend) -> None:
    body = response_body()
    if legend is None:
        del body["answers"]["frustration"]["legend"]
    else:
        body["answers"]["frustration"]["legend"] = legend
    model, _ = model_for(body)
    with pytest.raises(APIError):
        await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)
    response = await model.evaluate(
        chat_ctx=ChatContext.empty(), decisions=QUESTIONS, allow_partial=True
    )
    assert set(response.errors) == {"frustration"}
    assert set(response.results) == {"human", "intent"}


async def test_legend_key_order_does_not_change_level_order() -> None:
    body = response_body()
    body["answers"]["frustration"]["legend"] = {"2": "Angry", "0": "Calm", "1": "Annoyed"}
    model, _ = model_for(body)
    response = await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)
    assert response.results["frustration"].levels == ["Calm", "Annoyed", "Angry"]


@pytest.mark.parametrize("shift_probabilities", [False, True])
async def test_score_legend_cannot_shift_level_indices(shift_probabilities: bool) -> None:
    body = response_body()
    score = body["answers"]["frustration"]
    score["legend"] = {"1": "Calm", "2": "Annoyed", "3": "Angry"}
    if shift_probabilities:
        score["probabilities"] = {"1": 0.1, "2": 0.4, "3": 0.5}
    model, _ = model_for(body)
    with pytest.raises(APIError):
        await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)


async def test_model_identity_falls_back_to_configured_model() -> None:
    body = response_body()
    del body["model"]
    model, _ = model_for(body)
    response = await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)
    assert (response.model, response.provider) == (model.model, model.provider)


async def test_partial_results_can_report_every_answer_missing() -> None:
    model, _ = model_for({"answers": {}})
    response = await model.evaluate(
        chat_ctx=ChatContext.empty(), decisions=QUESTIONS, allow_partial=True
    )
    assert response.results == {}
    assert set(response.errors) == set(QUESTIONS)


@pytest.mark.parametrize("body", [{}, {"answers": []}, {"answers": {"unknown": None}}])
async def test_partial_results_do_not_accept_invalid_batch_structure(body) -> None:
    model, _ = model_for(body)
    with pytest.raises(APIError):
        await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS, allow_partial=True)


@pytest.mark.parametrize("include_context_events", [False, True])
async def test_context_events_are_opt_in(include_context_events: bool) -> None:
    context = ChatContext(
        [
            ChatMessage(role="user", content=["Book a table."]),
            FunctionCall(call_id="call-1", name="book", arguments='{"party_size": 2}'),
            FunctionCallOutput(call_id="call-1", name="book", output="No tables", is_error=False),
            AgentHandoff(old_agent_id="booking", new_agent_id="receptionist"),
            ChatMessage(role="assistant", content=["I can offer"], interrupted=True),
        ]
    )
    model, http = model_for(response_body())
    await model.evaluate(
        chat_ctx=context, decisions=QUESTIONS, include_context_events=include_context_events
    )
    state = http.post.call_args.kwargs["json"]["state"]
    if include_context_events:
        assert [item["type"] for item in state] == [item.type for item in context.items]
        assert state[1]["arguments"] == '{"party_size": 2}'
        assert state[2]["output"] == "No tables"
        assert state[3]["new_agent_id"] == "receptionist"
        assert state[4]["interrupted"] is True
    else:
        assert state == [
            {"role": "user", "content": "Book a table."},
            {"role": "assistant", "content": "I can offer"},
        ]


@pytest.mark.parametrize("include_context_events", [False, True])
async def test_decision_context_excludes_instructions_and_media(
    include_context_events: bool,
) -> None:
    context = ChatContext(
        [
            ChatMessage(role="system", content=["Answer only in Spanish."]),
            ChatMessage(role="developer", content=["Internal instructions."]),
            AgentConfigUpdate(instructions="More internal instructions."),
            ChatMessage(
                role="user",
                content=[
                    "Reserve a table.",
                    ImageContent(image="https://example.com/image.png"),
                    AudioContent(frame=[]),
                ],
            ),
            ChatMessage(role="assistant", content=["For how many?"], interrupted=True),
        ]
    )
    model, http = model_for(response_body())
    await model.evaluate(
        chat_ctx=context, decisions=QUESTIONS, include_context_events=include_context_events
    )
    state = http.post.call_args.kwargs["json"]["state"]
    assert [item["role"] for item in state] == ["user", "assistant"]
    if include_context_events:
        assert [item["content"] for item in state] == [["Reserve a table."], ["For how many?"]]
        assert state[1]["interrupted"] is True
    else:
        assert [item["content"] for item in state] == ["Reserve a table.", "For how many?"]
    assert len(context.items) == 5
    assert len(context.items[3].content) == 3
