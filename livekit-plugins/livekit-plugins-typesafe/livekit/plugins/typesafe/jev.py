from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Annotated, Any, Literal, cast

import aiohttp
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    utils,
)
from livekit.agents.decisions import (
    Choice,
    ChoiceResult,
    Decision,
    DecisionModel,
    DecisionResponse,
    DecisionResult,
    Probability,
    ProbabilityResult,
    ScoreResult,
)
from livekit.agents.llm import ChatContext
from livekit.agents.types import APIConnectOptions


class _NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class _ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None


class _ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    probabilities: dict[int, float]
    legend: dict[int, str]
    confidence: float | None = None


class _Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


_Answer = Annotated[_NoulAnswer | _ChoiceAnswer | _ScoreAnswer, Field(discriminator="type")]
_answer_adapter = TypeAdapter[_Answer](_Answer)


class _Response(BaseModel):
    answers: dict[str, Any]
    model: str = ""
    usage: _Usage = Field(default_factory=_Usage)
    id: str = ""


class Jev(DecisionModel):
    """Jev via TypeSafe, or via OpenRouter with :meth:`with_openrouter`.

    Args:
        model: TypeSafe model identifier.
        api_key: Defaults to TYPESAFE_API_KEY.
        endpoint: Complete HTTP endpoint for batched decisions.
        http_session: Optional caller-owned session. Otherwise uses the job's HTTP
            session. Outside a job, use ``async with utils.http_context.open()``.
    """

    def __init__(
        self,
        *,
        model: str = "jev-1.13.0",
        api_key: str | None = None,
        endpoint: str = "https://api.typesafe.ai/v1/systemone",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(capabilities=frozenset({"probability", "choice", "score"}))
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("Jev requires api_key or TYPESAFE_API_KEY")
        self._api_key = key
        self._model = model
        self._endpoint = endpoint
        self._session = http_session
        self._provider = "typesafe"

    @classmethod
    def with_openrouter(
        cls,
        *,
        model: str = "typesafe/jev-1.13",
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> Jev:
        """Use OpenRouter's Decisions API with OPENROUTER_API_KEY."""
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Jev.with_openrouter requires api_key or OPENROUTER_API_KEY")
        instance = cls(
            model=model,
            api_key=key,
            endpoint="https://openrouter.ai/api/alpha/decisions",
            http_session=http_session,
        )
        instance._provider = "openrouter"
        return instance

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    async def _evaluate_impl(
        self,
        *,
        chat_ctx: ChatContext,
        decisions: Mapping[str, Decision],
        include_context_events: bool,
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        questions = {name: _encode_question(definition) for name, definition in decisions.items()}
        state = _encode_context(chat_ctx, include_context_events=include_context_events)
        session = self._session or utils.http_context.http_session()
        try:
            async with session.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "state": state, "questions": questions},
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as response:
                if response.status >= 400:
                    raise APIStatusError(
                        "Jev decision request failed",
                        status_code=response.status,
                        request_id=response.headers.get("x-request-id"),
                    )
                parsed = _Response.model_validate(await response.json())
                request_id = parsed.id or response.headers.get("x-request-id", "")
            results: dict[str, DecisionResult] = {}
            errors: dict[str, str] = {}
            for name, raw_answer in parsed.answers.items():
                try:
                    results[name] = _decode_answer(_answer_adapter.validate_python(raw_answer))
                except ValidationError:
                    errors[name] = "invalid Jev decision answer"
                except APIError as error:
                    errors[name] = error.message
            return DecisionResponse(
                results=results,
                errors=errors,
                model=parsed.model,
                provider=self.provider,
                request_id=request_id,
                input_tokens=parsed.usage.input_tokens,
                output_tokens=parsed.usage.output_tokens,
            )
        except asyncio.TimeoutError as exc:
            raise APITimeoutError("Jev request timed out") from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError("Jev connection failed") from exc
        except ValueError as exc:
            raise APIError("invalid Jev decision response", retryable=False) from exc


def _encode_context(chat_ctx: ChatContext, *, include_context_events: bool) -> list[dict[str, Any]]:
    context = ChatContext(
        [
            item
            for item in chat_ctx.items
            if item.type != "message" or item.role in ("user", "assistant")
        ]
    )
    if include_context_events:
        return cast(
            list[dict[str, Any]],
            context.to_dict(exclude_metrics=True, exclude_config_update=True)["items"],
        )
    return [
        {"role": item.role, "content": item.text_content}
        for item in context.items
        if item.type == "message" and item.text_content
    ]


def _encode_question(definition: Decision) -> dict[str, Any]:
    question: dict[str, Any] = {"instructions": definition.instructions}
    if isinstance(definition, Probability):
        question["type"] = "noul"
    elif isinstance(definition, Choice):
        question.update(type="choice", criteria=definition.options)
    else:
        question.update(type="score", criteria=definition.levels)
    return question


def _decode_answer(answer: _Answer) -> DecisionResult:
    if isinstance(answer, _NoulAnswer):
        return ProbabilityResult(value=answer.noul)
    if isinstance(answer, _ChoiceAnswer):
        return ChoiceResult(
            value=answer.choice,
            probabilities=answer.probabilities,
            provider_data={"confidence": answer.confidence},
        )
    if answer.legend.keys() != answer.probabilities.keys():
        raise APIError("score legend does not match distribution", retryable=False)
    return ScoreResult(
        value=answer.score,
        probabilities=answer.probabilities,
        levels=[answer.legend[index] for index in sorted(answer.legend)],
        provider_data={"confidence": answer.confidence},
    )
