from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

import openai
from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    llm,
)
from livekit.agents.llm.decisions import (
    Choice,
    ChoiceResult,
    Decision,
    DecisionCapabilities,
    DecisionResponse,
    DecisionResult,
    Probability,
    ProbabilityResult,
    ScoreResult,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import is_given
from openai.types.decision import (
    Answer,
    AnswerAnswerResourceChoice,
    AnswerAnswerResourcePredicate,
    AnswerAnswerResourceRefusal,
)
from openai.types.decision_create_params import Question

from .models import DecisionModels


class DecisionModel(llm.DecisionModel):
    """OpenAI's Decisions API (``POST /v1/decisions``).

    One request answers a batch of probability, choice, and score questions about the
    same input.

    Example::

        model = openai.DecisionModel()
        response = await model.evaluate(
            input=chat_ctx,
            decisions={"handoff": llm.Probability("The caller requests a human.")},
        )

    Args:
        model: OpenAI decision model identifier.
        api_key: Defaults to OPENAI_API_KEY.
        base_url: Defaults to the OpenAI API.
        client: Optional caller-owned client, used instead of the arguments above.
        safety_identifier: Opaque end-user identifier passed to OpenAI.
        max_retries: Retries inside the OpenAI client. Decisions are retried by
            ``evaluate`` through ``conn_options``, so this defaults to none.
    """

    def __init__(
        self,
        *,
        model: DecisionModels | str = "gpt-6-luna-decisions",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            capabilities=DecisionCapabilities(decision_kinds=["probability", "choice", "score"])
        )
        if is_given(api_key) and not api_key:
            raise ValueError(
                "OpenAI API key is required, either as argument or set"
                " OPENAI_API_KEY environment variable"
            )

        self._model = model
        self._safety_identifier = safety_identifier
        self._owns_client = client is None
        self._client = client or openai.AsyncClient(
            api_key=api_key if is_given(api_key) else None,
            base_url=base_url if is_given(base_url) else None,
            max_retries=max_retries if is_given(max_retries) else 0,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        names = list(decisions)
        try:
            raw = await self._client.decisions.with_raw_response.create(
                model=self._model,
                input=input,
                questions=[_encode_question(name, decisions[name]) for name in names],
                safety_identifier=self._safety_identifier
                if is_given(self._safety_identifier)
                else openai.omit,
                timeout=conn_options.timeout,
            )
        except openai.APITimeoutError:
            raise APITimeoutError("OpenAI decision request timed out") from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message, status_code=e.status_code, request_id=e.request_id, body=e.body
            ) from None
        except openai.APIError as e:
            raise APIConnectionError("OpenAI decision request failed") from e

        try:
            decision = raw.parse()
        except (ValidationError, ValueError) as e:
            raise APIError("invalid OpenAI decision response", retryable=False) from e

        results, errors = _decode_answers(decision.answers, names)
        return DecisionResponse(
            results=results,
            errors=errors,
            model=decision.model,
            provider=self.provider,
            request_id=raw.request_id or "",
            input_tokens=decision.usage.input_tokens,
            output_tokens=decision.usage.output_tokens,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()


def _encode_question(name: str, definition: Decision) -> Question:
    if isinstance(definition, Probability):
        return {"type": "predicate", "name": name, "instructions": definition.prompt}
    if isinstance(definition, Choice):
        return {
            "type": "choice",
            "name": name,
            "instructions": definition.prompt,
            "choices": [
                {"value": value, "description": description}
                for value, description in definition.options.items()
            ],
        }
    return {
        "type": "score",
        "name": name,
        "instructions": definition.prompt,
        "levels": [{"label": level} for level in definition.levels],
    }


def _decode_answers(
    answers: list[Answer], names: list[str]
) -> tuple[dict[str, DecisionResult], dict[str, str]]:
    """Pair answers with the decisions they answer, by name or, unnamed, by question order.

    Question order only identifies an unnamed answer in a whole batch: one dropped answer
    would otherwise shift every later one onto the wrong decision, unnoticed between
    decisions of the same kind.
    """
    results: dict[str, DecisionResult] = {}
    errors: dict[str, str] = {}
    if any(answer.name is None for answer in answers) and (
        len(answers) != len(names)
        or any(answer.name not in (None, names[index]) for index, answer in enumerate(answers))
    ):
        raise APIError("unnamed decision answers do not line up with the batch", retryable=False)

    for index, answer in enumerate(answers):
        name = names[index] if answer.name is None else answer.name
        if not name:
            raise APIError("decision answer does not name a decision", retryable=False)
        if name in results or name in errors:
            raise APIError("decision response answers a decision twice", retryable=False)
        try:
            results[name] = _decode_answer(answer)
        except ValidationError:
            errors[name] = "invalid OpenAI decision answer"
        except APIError as error:
            errors[name] = error.message
    return results, errors


def _decode_answer(answer: Answer) -> DecisionResult:
    if isinstance(answer, AnswerAnswerResourceRefusal):
        raise APIError("the model declined this decision", retryable=False)
    if isinstance(answer, AnswerAnswerResourcePredicate):
        return ProbabilityResult(value=answer.probability)
    if isinstance(answer, AnswerAnswerResourceChoice):
        # Choices are typed; we only ever send strings, so a boolean names no option.
        probabilities = {
            p.value: p.probability for p in answer.probabilities if isinstance(p.value, str)
        }
        if not isinstance(answer.choice, str) or len(probabilities) != len(answer.probabilities):
            raise APIError("choice answer is not one of the requested options", retryable=False)
        return ChoiceResult(
            value=answer.choice,
            probabilities=probabilities,
            provider_data={"confidence": answer.confidence},
        )
    levels = {p.value: p.label for p in answer.probabilities}
    if len(levels) != len(answer.probabilities):
        raise APIError("score distribution repeats a level", retryable=False)
    return ScoreResult(
        value=answer.score,
        probabilities={p.value: p.probability for p in answer.probabilities},
        levels=[levels[index] for index in sorted(levels)],
        provider_data={"confidence": answer.confidence},
    )
