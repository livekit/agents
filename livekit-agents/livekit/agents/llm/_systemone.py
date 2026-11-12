"""The System One decision wire format: TypeSafe Jev, and every provider the
LiveKit Inference gateway serves through ``/v1/decisions``."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .._exceptions import APIError
from .decisions import (
    Choice,
    ChoiceResult,
    Decision,
    DecisionResult,
    Probability,
    ProbabilityResult,
    ScoreResult,
)


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    probabilities: dict[int, float]
    # TypeSafe keys the legend by level index; the gateway's OpenAI adapter
    # sends it positionally.
    legend: dict[int, str] | list[str]
    confidence: float | None = None

    def legend_by_index(self) -> dict[int, str]:
        if isinstance(self.legend, dict):
            return self.legend
        return dict(enumerate(self.legend))


class RefusalAnswer(BaseModel):
    """The gateway's rendering of a question the model declined."""

    type: Literal["refusal"]


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


Answer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer, Field(discriminator="type")
]
answer_adapter = TypeAdapter[Answer](Answer)


class Response(BaseModel):
    answers: dict[str, Any]
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    id: str = ""


def encode_question(definition: Decision) -> dict[str, Any]:
    question: dict[str, Any] = {"instructions": definition.prompt}
    if isinstance(definition, Probability):
        question["type"] = "noul"
    elif isinstance(definition, Choice):
        question.update(type="choice", criteria=definition.options)
    else:
        question.update(type="score", criteria=definition.levels)
    return question


def decode_answer(answer: Answer) -> DecisionResult:
    if isinstance(answer, RefusalAnswer):
        raise APIError("the model declined this decision", retryable=False)
    if isinstance(answer, NoulAnswer):
        return ProbabilityResult(value=answer.noul)
    if isinstance(answer, ChoiceAnswer):
        return ChoiceResult(
            value=answer.choice,
            probabilities=answer.probabilities,
            provider_data={"confidence": answer.confidence},
        )
    legend = answer.legend_by_index()
    if legend.keys() != answer.probabilities.keys():
        raise APIError("score legend does not match distribution", retryable=False)
    return ScoreResult(
        value=answer.score,
        probabilities=answer.probabilities,
        levels=[legend[index] for index in sorted(legend)],
        provider_data={"confidence": answer.confidence},
    )


def decode_answers(
    answers: dict[str, Any], *, invalid: str = "invalid decision answer"
) -> tuple[dict[str, DecisionResult], dict[str, str]]:
    """Decode a response's ``answers`` into results and per-decision errors."""
    results: dict[str, DecisionResult] = {}
    errors: dict[str, str] = {}
    for name, raw in answers.items():
        try:
            results[name] = decode_answer(answer_adapter.validate_python(raw))
        except ValidationError:
            errors[name] = invalid
        except APIError as error:
            errors[name] = error.message
    return results, errors
