"""Batched probability estimates, choices, and scores over a conversation snapshot.

The host type of ``main``'s ``llm.DecisionModel``: a provider implements ``_evaluate_impl``, one
attempt at the request, and ``evaluate`` checks the decisions before it and the response after
it. ``AMD(llm=...)`` classifies with one ``Choice`` per turn. Unlike ``main``, ``evaluate`` makes
one attempt (``conn_options`` reaches the provider as given), and the model emits no metrics.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .._bindings import APIError
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
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


def _format_transcript(chat_ctx: ChatContext) -> str:
    """The transcript a decision model receives: one ``role: text`` line per user or assistant
    message. Instructions, system and developer messages, and non-text content are left out."""
    lines: list[str] = []
    for message in chat_ctx.messages():
        if message.role in ("user", "assistant") and (text := message.text_content):
            lines.append(f"{message.role}: {text}")
    return "\n".join(lines)


def _kind(decision: Decision) -> DecisionKind:
    if isinstance(decision, Probability):
        return "probability"
    if isinstance(decision, Choice):
        return "choice"
    return "score"


def _validate_decisions(
    decisions: Mapping[str, Decision], *, capabilities: DecisionCapabilities
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
        if kind not in capabilities.decision_kinds:
            raise ValueError(
                f"decision model does not support {kind!r} required by decision {name!r}"
            )


def _validate_response(response: DecisionResponse, decisions: Mapping[str, Decision]) -> None:
    if (response.results.keys() | response.errors.keys()) - decisions.keys():
        raise APIError("decision response contains unknown decisions", kind="protocol")
    if response.results.keys() & response.errors.keys():
        raise APIError("decision response contains both a result and an error", kind="protocol")
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
        raise APIError("wrong result kind", kind="protocol")
    if isinstance(decision, Choice) and isinstance(result, ChoiceResult):
        if result.value not in decision.options:
            raise APIError("unknown choice", kind="protocol")
        if result.probabilities is not None:
            if result.probabilities.keys() != decision.options.keys():
                raise APIError("choice distribution does not match options", kind="protocol")
            if result.probabilities[result.value] != max(result.probabilities.values()):
                raise APIError("choice is not a most probable option", kind="protocol")
    if isinstance(decision, Score) and isinstance(result, ScoreResult):
        if result.levels != decision.levels or set(result.probabilities) != set(
            range(len(decision.levels))
        ):
            raise APIError("score distribution does not match levels", kind="protocol")
        if result.value > len(decision.levels) - 1:
            raise APIError("score is outside the defined scale", kind="protocol")


class DecisionModel(ABC):
    """Provider-neutral batched decisions over a text input or conversation.

    A ChatContext is converted to a ``role: text`` transcript, so every provider receives
    the same text.
    """

    def __init__(self, *, capabilities: DecisionCapabilities) -> None:
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

    async def evaluate(
        self,
        *,
        input: str | ChatContext,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        extra_headers: NotGivenOr[Mapping[str, str]] = NOT_GIVEN,
    ) -> DecisionResponse:
        """Evaluate a batch of decisions.

        Args:
            input: Text to evaluate, sent as-is, or a conversation converted to a
                transcript of user and assistant text.
            decisions: Decisions keyed by name.
            conn_options: The request's timeouts, passed to the provider.
            extra_headers: Headers sent with the request.

        Each requested name appears in either ``results`` or ``errors``. Request failures
        and malformed responses raise.
        """
        definitions = copy.deepcopy(dict(decisions))
        _validate_decisions(definitions, capabilities=self.capabilities)
        text = input if isinstance(input, str) else _format_transcript(input)
        headers = dict(extra_headers) if is_given(extra_headers) else {}
        if not definitions:
            return DecisionResponse(results={}, model=self.model, provider=self.provider)
        response = await self._evaluate_impl(
            input=text, decisions=definitions, conn_options=conn_options, extra_headers=headers
        )
        response.model = response.model or self.model
        response.provider = response.provider or self.provider
        _validate_response(response, definitions)
        return response

    @abstractmethod
    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
        extra_headers: Mapping[str, str],
    ) -> DecisionResponse: ...

    async def aclose(self) -> None:  # noqa: B027
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
