from __future__ import annotations

import asyncio
import copy
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from opentelemetry import trace
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from livekit import rtc

from .._exceptions import APIError, APITimeoutError
from ..llm import ChatContext
from ..metrics import DecisionMetrics
from ..metrics.base import Metadata
from ..telemetry import gen_ai, trace_types, tracer
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

DecisionKind = Literal["probability", "choice", "score"]
DecisionInputModality = Literal["text", "image"]
_Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
# Jev rounds scores and probabilities independently to two decimal places.
_ROUNDING_ERROR = 0.005
_FLOAT_EPSILON = 1e-9


@dataclass(frozen=True)
class Probability:
    """Estimate the probability that ``prompt`` is true. No boolean threshold is applied."""

    prompt: str


@dataclass(frozen=True)
class Choice:
    """Select one named option. Descriptions explain when each option applies."""

    prompt: str
    options: dict[str, str]


@dataclass(frozen=True)
class Score:
    """Estimate a position on ordered descriptive levels, numbered from zero.

    The result is the expected level index, including fractional positions.
    """

    prompt: str
    levels: list[str]


Decision = Probability | Choice | Score


@dataclass(frozen=True)
class DecisionCapabilities:
    decision_kinds: list[DecisionKind]
    """Decision kinds the model can evaluate"""
    input_modalities: list[DecisionInputModality] = field(default_factory=lambda: ["text"])
    """Chat content the model can consume. AgentSession leaves other content out of
    background evaluations.
    """


class ProbabilityResult(BaseModel):
    kind: Literal["probability"] = "probability"
    value: _Probability
    provider_data: dict[str, Any] = Field(default_factory=dict)


class ChoiceResult(BaseModel):
    kind: Literal["choice"] = "choice"
    value: str
    probabilities: dict[str, _Probability] | None = None
    """Provider distribution (possibly rounded), or None if unavailable."""
    provider_data: dict[str, Any] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    kind: Literal["score"] = "score"
    value: float = Field(ge=0, allow_inf_nan=False)
    probabilities: dict[int, _Probability]
    """Provider distribution over zero-based level indices. The probabilities and
    expected index can be independently rounded.
    """
    levels: list[str]
    provider_data: dict[str, Any] = Field(default_factory=dict)


DecisionResult = Annotated[
    ProbabilityResult | ChoiceResult | ScoreResult, Field(discriminator="kind")
]


class DecisionResponse(BaseModel):
    results: dict[str, DecisionResult]
    errors: dict[str, str] = Field(default_factory=dict)
    """Errors keyed by decision name. Failed decisions are absent from results."""
    model: str = ""
    provider: str = ""
    request_id: str = ""
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class DecisionsCompletedEvent(BaseModel):
    type: Literal["decisions_completed"] = "decisions_completed"
    results: dict[str, DecisionResult]
    errors: dict[str, str] = Field(default_factory=dict)
    model: str = ""
    provider: str = ""
    source_message_id: str
    agent_id: str
    activity_id: str
    request_id: str = ""
    created_at: float = Field(default_factory=time.time)


class DecisionOptions(TypedDict, total=False):
    """Background evaluation of the active agent's decisions.

    One request runs at a time. Eligible updates replace one pending snapshot;
    intermediate snapshots can be skipped. Results identify the snapshot evaluated.
    """

    turn_interval: int
    """Evaluate every N committed, non-empty user text turns. Default: 1."""
    max_context_turns: int
    """Include the last N user turns and intervening assistant text, ending at the
    triggering user message. By default, tools and other context events are excluded.
    Default: 6. Context can include conversation from before an agent handoff.
    """
    include_context_events: bool
    """Include tool calls, tool outputs, handoffs, and interruption metadata.
    Default: False. Instructions are always excluded, and images are included only when
    the decision model accepts image input.
    """
    allow_partial: bool
    """Emit valid results and per-decision errors when some answers fail validation.
    Default: False, which rejects the whole batch.
    """
    timeout: float
    """Maximum seconds per background evaluation, including retries. Default: 10."""


def _resolve_options(options: DecisionOptions | None) -> DecisionOptions:
    resolved = DecisionOptions(
        turn_interval=1,
        max_context_turns=6,
        timeout=10.0,
        include_context_events=False,
        allow_partial=False,
    )
    resolved.update(options or {})
    for name in ("turn_interval", "max_context_turns"):
        value = resolved[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"decision_options.{name} must be a positive integer")
    if not math.isfinite(resolved["timeout"]) or resolved["timeout"] <= 0:
        raise ValueError("decision_options.timeout must be positive and finite")
    return resolved


def _kind(decision: Decision) -> DecisionKind:
    if isinstance(decision, Probability):
        return "probability"
    if isinstance(decision, Choice):
        return "choice"
    return "score"


def _validate_decisions(
    decisions: Mapping[str, Decision], *, capabilities: DecisionCapabilities | None = None
) -> None:
    for name, decision in decisions.items():
        if not name or not decision.prompt.strip():
            raise ValueError("decisions require a name and non-empty prompt")
        if isinstance(decision, Choice):
            if len(decision.options) < 2 or any(not key for key in decision.options):
                raise ValueError(f"decision {name!r} requires at least two named options")
        elif isinstance(decision, Score):
            if len(decision.levels) < 2 or any(not level.strip() for level in decision.levels):
                raise ValueError(f"decision {name!r} requires at least two described levels")
        kind = _kind(decision)
        if capabilities is not None and kind not in capabilities.decision_kinds:
            raise ValueError(
                f"decision model does not support {kind!r} required by decision {name!r}"
            )


def _validate_response(response: DecisionResponse, decisions: Mapping[str, Decision]) -> None:
    if (response.results.keys() | response.errors.keys()) - decisions.keys():
        raise APIError("decision response contains unknown decisions", retryable=False)
    if response.results.keys() & response.errors.keys():
        raise APIError("decision response contains both a result and an error", retryable=False)
    for name, decision in decisions.items():
        if name not in response.results:
            response.errors.setdefault(name, "missing decision answer")
            continue
        try:
            _validate_result(response.results[name], decision)
        except APIError as error:
            del response.results[name]
            response.errors[name] = error.message


def _validate_result(result: DecisionResult, decision: Decision) -> None:
    if result.kind != _kind(decision):
        raise APIError("wrong result kind", retryable=False)
    if isinstance(decision, Choice) and isinstance(result, ChoiceResult):
        if result.value not in decision.options:
            raise APIError("unknown choice", retryable=False)
        if result.probabilities is not None:
            if result.probabilities.keys() != decision.options.keys():
                raise APIError("choice distribution does not match options", retryable=False)
            if result.probabilities[result.value] != max(result.probabilities.values()):
                raise APIError("choice is not a most probable option", retryable=False)
    if isinstance(decision, Score) and isinstance(result, ScoreResult):
        if result.levels != decision.levels or set(result.probabilities) != set(
            range(len(decision.levels))
        ):
            raise APIError("score distribution does not match levels", retryable=False)
        if result.value > len(decision.levels) - 1:
            raise APIError("score is outside the defined scale", retryable=False)
        expected_score = sum(index * p for index, p in result.probabilities.items())
        score_tolerance = _ROUNDING_ERROR * (1 + sum(range(len(decision.levels)))) + _FLOAT_EPSILON
        if not math.isclose(result.value, expected_score, abs_tol=score_tolerance):
            raise APIError("score must be the expected level index", retryable=False)
    if isinstance(result, (ChoiceResult, ScoreResult)) and result.probabilities is not None:
        probability_sum_tolerance = _ROUNDING_ERROR * len(result.probabilities) + _FLOAT_EPSILON
        if not math.isclose(
            sum(result.probabilities.values()), 1.0, abs_tol=probability_sum_tolerance
        ):
            raise APIError("decision probabilities must sum to one", retryable=False)


class DecisionModel(ABC, rtc.EventEmitter[Literal["metrics_collected"]]):
    """Provider-neutral batched decisions, also usable outside AgentSession.

    For on-demand evaluation, pass an explicit ChatContext, including any new message
    received separately by ``on_user_turn_completed``.
    """

    def __init__(self, *, capabilities: DecisionCapabilities) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        return self._label

    @property
    def capabilities(self) -> DecisionCapabilities:
        return self._capabilities

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "unknown"

    @tracer.start_as_current_span("decision_model.evaluate")
    async def evaluate(
        self,
        *,
        chat_ctx: ChatContext,
        decisions: Mapping[str, Decision],
        allow_partial: bool = False,
        include_context_events: bool = False,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> DecisionResponse:
        """Evaluate a batch, rejecting invalid answers unless ``allow_partial`` is set.

        With partial results, each requested name appears in either ``results`` or
        ``errors``. Request failures still raise. ``include_context_events`` includes
        tools, handoffs, and interruption metadata in the provider input.
        """
        span = trace.get_current_span()
        span.set_attributes(
            {
                trace_types.ATTR_GEN_AI_REQUEST_MODEL: self.model,
                trace_types.ATTR_GEN_AI_PROVIDER_NAME: self.provider,
                "lk.decision_count": len(decisions),
            }
        )
        definitions = copy.deepcopy(dict(decisions))
        _validate_decisions(definitions, capabilities=self.capabilities)
        if not definitions:
            return DecisionResponse(results={}, model=self.model, provider=self.provider)

        started = time.perf_counter()
        for attempt in range(conn_options.max_retry + 1):
            try:
                response = await asyncio.wait_for(
                    self._evaluate_impl(
                        chat_ctx=chat_ctx,
                        decisions=definitions,
                        include_context_events=include_context_events,
                        conn_options=conn_options,
                    ),
                    timeout=conn_options.timeout,
                )
            except (APIError, asyncio.TimeoutError) as error:
                if isinstance(error, APIError) and not error.retryable:
                    raise
                if attempt == conn_options.max_retry:
                    if isinstance(error, asyncio.TimeoutError):
                        raise APITimeoutError("decision request timed out") from error
                    raise
                await asyncio.sleep(conn_options._interval_for_retry(attempt))
                continue

            response.model = response.model or self.model
            response.provider = response.provider or self.provider
            gen_ai.set_response_attributes(
                span, response_id=response.request_id, model=response.model
            )
            if response.input_tokens is not None:
                span.set_attribute(
                    trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS, response.input_tokens
                )
            if response.output_tokens is not None:
                span.set_attribute(
                    trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, response.output_tokens
                )
            self.emit(
                "metrics_collected",
                DecisionMetrics(
                    label=self.label,
                    request_id=response.request_id,
                    timestamp=time.time(),
                    duration=time.perf_counter() - started,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    metadata=Metadata(model_name=response.model, model_provider=response.provider),
                ),
            )
            _validate_response(response, definitions)
            span.set_attribute("lk.decision_error_count", len(response.errors))
            if response.errors and not allow_partial:
                raise APIError(
                    "invalid decision answers: "
                    + "; ".join(f"{name}: {error}" for name, error in response.errors.items()),
                    retryable=False,
                )
            return response
        raise RuntimeError("unreachable")

    @abstractmethod
    async def _evaluate_impl(
        self,
        *,
        chat_ctx: ChatContext,
        decisions: Mapping[str, Decision],
        include_context_events: bool,
        conn_options: APIConnectOptions,
    ) -> DecisionResponse: ...

    async def aclose(self) -> None:
        """Release resources owned by this model, if any."""
