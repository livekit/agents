from __future__ import annotations

import asyncio
import copy
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from livekit import rtc

from .._exceptions import APIError, APITimeoutError
from ..llm import ChatContext
from ..metrics import DecisionMetrics
from ..metrics.base import Metadata
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

DecisionKind = Literal["probability", "choice", "score"]
_Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
# Providers such as Jev independently round scores and probabilities to two
# decimal places. Each reported number can differ by half a hundredth.
_ROUNDING_ERROR = 0.005


@dataclass(frozen=True)
class Probability:
    """Estimate the probability that ``instructions`` is true. No boolean threshold is applied."""

    instructions: str


@dataclass(frozen=True)
class Choice:
    """Select one named option. Descriptions explain when each option applies."""

    instructions: str
    options: dict[str, str]


@dataclass(frozen=True)
class Score:
    """Estimate a position on ordered descriptive levels, numbered from zero.

    The result is the expected level index, including fractional positions.
    """

    instructions: str
    levels: list[str]


Decision = Probability | Choice | Score


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
    request_id: str = ""
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class DecisionsCompletedEvent(BaseModel):
    type: Literal["decisions_completed"] = "decisions_completed"
    results: dict[str, DecisionResult]
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
    triggering user message. Instructions, tools, and non-text content are excluded.
    Default: 6. Context can include conversation from before an agent handoff.
    """
    timeout: float
    """Maximum seconds per background evaluation, including retries. Default: 10."""


def _resolve_options(options: DecisionOptions | None) -> DecisionOptions:
    resolved = DecisionOptions(turn_interval=1, max_context_turns=6, timeout=10.0)
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
    decisions: Mapping[str, Decision], *, capabilities: frozenset[DecisionKind] | None = None
) -> None:
    for name, decision in decisions.items():
        if not name or not decision.instructions.strip():
            raise ValueError("decisions require a name and non-empty instructions")
        if isinstance(decision, Choice):
            if len(decision.options) < 2 or any(not key for key in decision.options):
                raise ValueError(f"decision {name!r} requires at least two named options")
        elif isinstance(decision, Score):
            if len(decision.levels) < 2 or any(not level.strip() for level in decision.levels):
                raise ValueError(f"decision {name!r} requires at least two described levels")
        kind = _kind(decision)
        if capabilities is not None and kind not in capabilities:
            raise ValueError(
                f"decision model does not support {kind!r} required by decision {name!r}"
            )


def _validate_response(response: DecisionResponse, decisions: Mapping[str, Decision]) -> None:
    if response.results.keys() != decisions.keys():
        raise APIError("decision response must answer every requested decision", retryable=False)
    for name, decision in decisions.items():
        result = response.results[name]
        if result.kind != _kind(decision):
            raise APIError(f"wrong result kind for decision {name!r}", retryable=False)
        if isinstance(decision, Choice) and isinstance(result, ChoiceResult):
            if result.value not in decision.options:
                raise APIError(f"unknown choice for decision {name!r}", retryable=False)
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
            expected = sum(index * p for index, p in result.probabilities.items())
            # Accumulate the weighted probability errors and the score's own
            # rounding error. The small epsilon covers binary float arithmetic.
            tolerance = _ROUNDING_ERROR * (1 + sum(result.probabilities)) + 1e-9
            if not math.isclose(result.value, expected, abs_tol=tolerance):
                raise APIError("score must be the expected level index", retryable=False)
        if isinstance(result, (ChoiceResult, ScoreResult)) and result.probabilities is not None:
            tolerance = _ROUNDING_ERROR * len(result.probabilities) + 1e-9
            if not math.isclose(sum(result.probabilities.values()), 1.0, abs_tol=tolerance):
                raise APIError("decision probabilities must sum to one", retryable=False)


class DecisionModel(ABC, rtc.EventEmitter[Literal["metrics_collected"]]):
    """Provider-neutral batched decisions, also usable outside AgentSession.

    Providers implement ``_evaluate_impl`` and declare the supported question kinds.
    For on-demand evaluation, pass an explicit ChatContext, including any new message
    received separately by ``on_user_turn_completed``.
    """

    def __init__(self, *, capabilities: frozenset[DecisionKind]) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        return self._label

    @property
    def capabilities(self) -> frozenset[DecisionKind]:
        return self._capabilities

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "unknown"

    async def evaluate(
        self,
        *,
        chat_ctx: ChatContext,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> DecisionResponse:
        definitions = copy.deepcopy(dict(decisions))
        _validate_decisions(definitions, capabilities=self.capabilities)
        if not definitions:
            return DecisionResponse(results={})

        started = time.perf_counter()
        for attempt in range(conn_options.max_retry + 1):
            try:
                response = await asyncio.wait_for(
                    self._evaluate_impl(
                        chat_ctx=chat_ctx, decisions=definitions, conn_options=conn_options
                    ),
                    timeout=conn_options.timeout,
                )
                _validate_response(response, definitions)
                self.emit(
                    "metrics_collected",
                    DecisionMetrics(
                        label=self.label,
                        request_id=response.request_id,
                        timestamp=time.time(),
                        duration=time.perf_counter() - started,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        metadata=Metadata(model_name=self.model, model_provider=self.provider),
                    ),
                )
                return response
            except asyncio.TimeoutError as exc:
                error: APIError = APITimeoutError("decision request timed out")
                error.__cause__ = exc
            except APIError as exc:
                error = exc
            if not error.retryable or attempt == conn_options.max_retry:
                raise error
            await asyncio.sleep(conn_options._interval_for_retry(attempt))
        raise RuntimeError("unreachable")

    @abstractmethod
    async def _evaluate_impl(
        self,
        *,
        chat_ctx: ChatContext,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse: ...

    async def aclose(self) -> None:
        """Release resources owned by this model, if any."""
