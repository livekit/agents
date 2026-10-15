"""Batched probability estimates, choices, and scores over a conversation snapshot."""

from __future__ import annotations

import asyncio
import copy
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from opentelemetry import trace
from pydantic import BaseModel, Field

from livekit import rtc

from .._exceptions import APIError, APITimeoutError
from ..metrics import DecisionMetrics
from ..metrics.base import Metadata
from ..telemetry import gen_ai, trace_types, tracer
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from .chat_context import ChatContext

DecisionKind = Literal["probability", "choice", "score"]
_Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


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


class ProbabilityResult(BaseModel):
    kind: Literal["probability"] = "probability"
    value: _Probability
    provider_data: dict[str, Any] = Field(default_factory=dict)


class ChoiceResult(BaseModel):
    kind: Literal["choice"] = "choice"
    value: str
    probabilities: dict[str, _Probability] | None = None
    """Provider distribution over option names, or None if unavailable."""
    provider_data: dict[str, Any] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    kind: Literal["score"] = "score"
    value: float = Field(ge=0, allow_inf_nan=False)
    probabilities: dict[int, _Probability]
    """Provider distribution over zero-based level indices."""
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


def _format_transcript(chat_ctx: ChatContext, *, include_context_events: bool = False) -> str:
    """Convert a conversation into the transcript decision models receive.

    Only user and assistant text is kept, one ``role: text`` line per message, with
    expressive markup removed. Instructions, system and developer messages, and
    non-text content are always left out.

    Args:
        chat_ctx: The conversation to convert.
        include_context_events: Also include tool calls, tool outputs, agent handoffs,
            and interrupted assistant replies.
    """
    lines: list[str] = []
    for item in chat_ctx.items:
        if item.type == "message":
            if item.role not in ("user", "assistant") or not (text := item.text_content):
                continue
            if include_context_events and item.interrupted:
                text += " [interrupted]"
            lines.append(f"{item.role}: {text}")
        elif not include_context_events:
            continue
        elif item.type == "function_call":
            lines.append(f"[function call: {item.name}({item.arguments})]")
        elif item.type == "function_call_output":
            label = "function error" if item.is_error else "function output"
            lines.append(f"[{label}: {item.output}]")
        elif item.type == "agent_handoff":
            lines.append(f"[agent handoff: {item.old_agent_id} -> {item.new_agent_id}]")
    return "\n".join(lines)


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


class DecisionModel(ABC, rtc.EventEmitter[Literal["metrics_collected"]]):
    """Provider-neutral batched decisions over a text input or conversation.

    A ChatContext is converted to a ``role: text`` transcript, so every provider receives
    the same text. In ``on_user_turn_completed``, add the new message to the context first.
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
        input: str | ChatContext,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> DecisionResponse:
        """Evaluate a batch of decisions.

        Args:
            input: Text to evaluate, sent as-is, or a conversation converted to a
                transcript of user and assistant text.
            decisions: Decisions keyed by name.
            conn_options: Retry and per-attempt timeout settings.

        Each requested name appears in either ``results`` or ``errors``. Request failures
        and malformed responses raise.
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
        text = input if isinstance(input, str) else _format_transcript(input)
        if not definitions:
            return DecisionResponse(results={}, model=self.model, provider=self.provider)

        started = time.perf_counter()
        for attempt in range(conn_options.max_retry + 1):
            try:
                response = await asyncio.wait_for(
                    self._evaluate_impl(
                        input=text, decisions=definitions, conn_options=conn_options
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
            return response
        raise RuntimeError("unreachable")

    @abstractmethod
    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse: ...

    async def aclose(self) -> None:
        """Release resources owned by this model, if any."""


__all__ = [
    "Choice",
    "ChoiceResult",
    "Decision",
    "DecisionCapabilities",
    "DecisionKind",
    "DecisionModel",
    "DecisionResponse",
    "DecisionResult",
    "Probability",
    "ProbabilityResult",
    "Score",
    "ScoreResult",
]
