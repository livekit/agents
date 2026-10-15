from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from livekit.agents import APIError, APIStatusError, APITimeoutError
from livekit.agents.llm import (
    AgentConfigUpdate,
    AgentHandoff,
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
    decisions,
)
from livekit.agents.llm.decisions import (
    ChoiceResult,
    DecisionResponse,
    ProbabilityResult,
    ScoreResult,
    _format_transcript,
)
from livekit.agents.metrics import DecisionMetrics
from livekit.agents.types import APIConnectOptions

pytestmark = pytest.mark.unit

QUESTIONS: dict[str, decisions.Decision] = {
    "human": decisions.Probability("The caller wants a human."),
    "intent": decisions.Choice("Intent?", options={"booking": "Reservation", "other": "Other"}),
    "frustration": decisions.Score("Frustration?", levels=["Calm", "Annoyed", "Angry"]),
}


def valid_results() -> dict[str, decisions.DecisionResult]:
    return {
        "human": ProbabilityResult(value=0.9),
        "intent": ChoiceResult(value="booking", probabilities={"booking": 0.8, "other": 0.2}),
        "frustration": ScoreResult(
            value=1.4,
            probabilities={0: 0.1, 1: 0.4, 2: 0.5},
            levels=["Calm", "Annoyed", "Angry"],
        ),
    }


class StaticModel(decisions.DecisionModel):
    """Returns queued responses or raises queued errors, one per request."""

    def __init__(
        self,
        *outcomes: DecisionResponse | Exception,
        kinds: list[decisions.DecisionKind] | None = None,
    ) -> None:
        super().__init__(
            capabilities=decisions.DecisionCapabilities(
                decision_kinds=kinds or ["probability", "choice", "score"]
            )
        )
        self.outcomes = list(outcomes)
        self.requests: list[tuple[str, Mapping[str, decisions.Decision]]] = []

    @property
    def model(self) -> str:
        return "configured-model"

    @property
    def provider(self) -> str:
        return "test-provider"

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, decisions.Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        self.requests.append((input, decisions))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome.model_copy(deep=True)


class HangingModel(StaticModel):
    async def _evaluate_impl(self, **kwargs) -> DecisionResponse:  # type: ignore[override]
        self.requests.append((kwargs["input"], kwargs["decisions"]))
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def evaluate(model: decisions.DecisionModel, **kwargs) -> DecisionResponse:
    return await model.evaluate(
        input=kwargs.pop("input", "The caller asked for a table."),
        decisions=kwargs.pop("decisions", QUESTIONS),
        **kwargs,
    )


async def test_results_usage_and_identity_are_reported_once() -> None:
    model = StaticModel(
        DecisionResponse(
            results=valid_results(),
            model="resolved-model",
            request_id="request-1",
            input_tokens=50,
            output_tokens=5,
        )
    )
    collected = []
    model.on("metrics_collected", collected.append)
    response = await evaluate(model)

    assert response.results.keys() == QUESTIONS.keys()
    assert response.errors == {}
    assert (response.model, response.provider) == ("resolved-model", "test-provider")
    [(_, definitions)] = model.requests
    assert definitions == QUESTIONS
    [metrics] = collected
    assert isinstance(metrics, DecisionMetrics)
    assert (metrics.input_tokens, metrics.output_tokens) == (50, 5)
    assert metrics.request_id == "request-1"
    assert metrics.metadata.model_name == "resolved-model"
    assert metrics.metadata.model_provider == "test-provider"


async def test_model_identity_falls_back_to_configured_model() -> None:
    model = StaticModel(DecisionResponse(results=valid_results()))
    response = await evaluate(model)
    assert (response.model, response.provider) == ("configured-model", "test-provider")


async def test_definitions_are_copied_before_reaching_the_provider() -> None:
    model = StaticModel(DecisionResponse(results=valid_results()))
    await evaluate(model)
    [(_, definitions)] = model.requests
    assert definitions is not QUESTIONS
    assert definitions["intent"].options is not QUESTIONS["intent"].options


@pytest.mark.parametrize(
    ("name", "result", "error"),
    [
        ("human", ChoiceResult(value="booking"), "wrong result kind"),
        ("intent", ChoiceResult(value="invented"), "unknown choice"),
        (
            "intent",
            ChoiceResult(value="booking", probabilities={"booking": 1.0}),
            "choice distribution does not match options",
        ),
        (
            "intent",
            ChoiceResult(value="other", probabilities={"booking": 0.8, "other": 0.2}),
            "choice is not a most probable option",
        ),
        (
            "frustration",
            ScoreResult(value=1.0, probabilities={0: 0.5, 1: 0.5}, levels=["Calm", "Annoyed"]),
            "score distribution does not match levels",
        ),
        (
            "frustration",
            ScoreResult(
                value=2.5,
                probabilities={0: 0.0, 1: 0.0, 2: 1.0},
                levels=["Calm", "Annoyed", "Angry"],
            ),
            "score is outside the defined scale",
        ),
    ],
)
async def test_invalid_answer_is_reported_without_dropping_the_batch(
    name: str, result: decisions.DecisionResult, error: str
) -> None:
    results = valid_results()
    results[name] = result
    model = StaticModel(DecisionResponse(results=results))
    response = await evaluate(model)
    assert response.errors == {name: error}
    assert response.results.keys() == QUESTIONS.keys() - {name}


async def test_missing_answer_is_reported_per_decision() -> None:
    results = valid_results()
    del results["human"], results["intent"]
    model = StaticModel(DecisionResponse(results=results, errors={"intent": "provider refused"}))
    response = await evaluate(model)
    assert response.errors == {"human": "missing decision answer", "intent": "provider refused"}
    assert set(response.results) == {"frustration"}


async def test_core_does_not_enforce_provider_rounding() -> None:
    # Distributions that don't sum to one, or a score away from its expected index, are
    # left to each provider to judge against its own rounding behavior.
    results = valid_results()
    results["intent"] = ChoiceResult(value="booking", probabilities={"booking": 0.9, "other": 0.3})
    results["frustration"] = ScoreResult(
        value=0.2, probabilities={0: 0.1, 1: 0.4, 2: 0.5}, levels=["Calm", "Annoyed", "Angry"]
    )
    model = StaticModel(DecisionResponse(results=results))
    response = await evaluate(model)
    assert response.errors == {}
    assert response.results.keys() == QUESTIONS.keys()


@pytest.mark.parametrize("failure", ["unknown decisions", "both a result and an error"])
async def test_malformed_response_raises_after_reporting_usage(failure: str) -> None:
    response = DecisionResponse(results=valid_results(), input_tokens=50, output_tokens=5)
    if failure == "unknown decisions":
        response.results["extra"] = ProbabilityResult(value=0.5)
    else:
        response.errors["human"] = "invalid answer"
    model = StaticModel(response)
    collected = []
    model.on("metrics_collected", collected.append)
    with pytest.raises(APIError, match=failure):
        await evaluate(model)
    [metrics] = collected
    assert (metrics.input_tokens, metrics.output_tokens) == (50, 5)


async def test_retryable_errors_obey_connection_options() -> None:
    model = StaticModel(
        APIStatusError("unavailable", status_code=503),
        DecisionResponse(results=valid_results()),
    )
    response = await evaluate(model, conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
    assert len(model.requests) == 2
    assert response.errors == {}


async def test_non_retryable_errors_are_not_retried() -> None:
    model = StaticModel(APIStatusError("unauthorized", status_code=401))
    with pytest.raises(APIStatusError):
        await evaluate(model, conn_options=APIConnectOptions(max_retry=2, retry_interval=0))
    assert len(model.requests) == 1


@pytest.mark.parametrize("max_retry", [0, 1])
async def test_request_timeout_preserves_retry_count_and_exception_cause(max_retry: int) -> None:
    model = HangingModel()
    collected = []
    model.on("metrics_collected", collected.append)
    with pytest.raises(APITimeoutError) as error:
        await evaluate(
            model,
            decisions={"check": decisions.Probability("True?")},
            conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0, timeout=0.01),
        )
    assert len(model.requests) == max_retry + 1
    assert isinstance(error.value.__cause__, asyncio.TimeoutError)
    assert collected == []


async def test_empty_batch_skips_the_provider() -> None:
    model = StaticModel()
    response = await evaluate(model, decisions={})
    assert response.results == {}
    assert model.requests == []


async def test_unsupported_kind_fails_before_provider_request() -> None:
    model = StaticModel(kinds=["probability"])
    with pytest.raises(ValueError, match="does not support"):
        await evaluate(model)
    assert model.requests == []


@pytest.mark.parametrize(
    ("definition", "error"),
    [
        (decisions.Probability("  "), "non-empty prompt"),
        (decisions.Choice("Intent?", options={"only": "One option"}), "at least two named"),
        (decisions.Score("Level?", levels=["Only"]), "at least two described"),
    ],
)
async def test_invalid_definitions_fail_before_provider_request(
    definition: decisions.Decision, error: str
) -> None:
    model = StaticModel()
    with pytest.raises(ValueError, match=error):
        await evaluate(model, decisions={"check": definition})
    assert model.requests == []


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_probabilities_must_be_finite_and_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        decisions.ProbabilityResult(value=value)


async def test_string_input_is_sent_as_is() -> None:
    model = StaticModel(DecisionResponse(results=valid_results()))
    await evaluate(model, input="  Caller: I want a refund.\n")
    [(sent, _)] = model.requests
    assert sent == "  Caller: I want a refund.\n"


async def test_chat_context_input_is_sent_as_a_transcript() -> None:
    context = ChatContext.empty()
    context.add_message(role="user", content="Book a table, please.")
    context.add_message(role="assistant", content="For how many?")
    model = StaticModel(DecisionResponse(results=valid_results()))
    await evaluate(model, input=context)
    [(sent, _)] = model.requests
    assert sent == "user: Book a table, please.\nassistant: For how many?"


def conversation() -> ChatContext:
    return ChatContext(
        [
            ChatMessage(role="system", content=["Answer only in Spanish."]),
            ChatMessage(role="developer", content=["Internal instructions."]),
            AgentConfigUpdate(instructions="More internal instructions."),
            ChatMessage(
                role="user",
                content=["Book a table.", ImageContent(image="https://example.com/image.png")],
            ),
            FunctionCall(call_id="call-1", name="book", arguments='{"party_size": 2}'),
            FunctionCallOutput(call_id="call-1", name="book", output="No tables", is_error=True),
            AgentHandoff(old_agent_id="booking", new_agent_id="receptionist"),
            ChatMessage(
                role="assistant",
                content=['<expr value="warm"/>I can offer seven.'],
                interrupted=True,
            ),
        ]
    )


def test_transcript_keeps_only_user_and_assistant_text_by_default() -> None:
    assert _format_transcript(conversation()) == (
        "user: Book a table.\nassistant: I can offer seven."
    )


def test_transcript_includes_context_events_when_asked() -> None:
    assert _format_transcript(conversation(), include_context_events=True) == (
        "user: Book a table.\n"
        '[function call: book({"party_size": 2})]\n'
        "[function error: No tables]\n"
        "[agent handoff: booking -> receptionist]\n"
        "assistant: I can offer seven. [interrupted]"
    )
