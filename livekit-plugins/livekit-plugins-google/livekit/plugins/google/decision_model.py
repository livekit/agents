from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import ValidationError

from google import genai
from google.genai import _interactions as genai_errors, decisions as genai_decisions, types
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
    Score,
    ScoreResult,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import is_given

from .models import DecisionModels


class DecisionModel(llm.DecisionModel):
    """Gemini's Decisions API (``POST /v1beta/decisions``).

    One request answers a batch of probability, choice, and score questions about the
    same input.

    Example::

        model = google.DecisionModel()
        response = await model.evaluate(
            input=chat_ctx,
            decisions={"handoff": llm.Probability("The caller requests a human.")},
        )

    Args:
        model: Gemini decision model identifier.
        api_key: Gemini API key. Defaults to GOOGLE_API_KEY.
        client: Optional caller-owned client, used instead of ``api_key``. Pass one to
            configure Vertex AI or other HTTP options.
    """

    def __init__(
        self,
        *,
        model: DecisionModels | str = "gemini-decision-1",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        client: genai.Client | None = None,
    ) -> None:
        super().__init__(
            capabilities=DecisionCapabilities(decision_kinds=["probability", "choice", "score"])
        )
        self._model = model
        self._owns_client = client is None
        if client is None:
            gemini_api_key = api_key if is_given(api_key) else os.environ.get("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise ValueError(
                    "Gemini API key is required, either as argument or set"
                    " GOOGLE_API_KEY environment variable"
                )
            # Decisions are retried by ``evaluate`` through ``conn_options``.
            client = genai.Client(
                api_key=gemini_api_key,
                http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
            )
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "google"

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        try:
            raw = await self._client.aio.decisions.with_raw_response.create(
                model=self._model,
                input=input,
                questions={name: _encode_question(d) for name, d in decisions.items()},
                # The SDK defaults to v1, which has no decisions endpoint (a bare 404).
                api_version="v1beta",
                timeout=conn_options.timeout,
            )
            decision = await raw.parse()
        except genai_errors.APITimeoutError:
            raise APITimeoutError("Gemini decision request timed out") from None
        except genai_errors.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.response.status_code,
                request_id=e.response.headers.get("x-request-id"),
                body=e.body,
            ) from None
        except genai_errors.APIResponseValidationError as e:
            raise APIError("invalid Gemini decision response", retryable=False) from e
        except genai_errors.APIError as e:
            raise APIConnectionError("Gemini decision request failed") from e

        if feedback := decision.prompt_feedback:
            if feedback.block_reason:
                reason = feedback.block_reason_message or feedback.block_reason
                raise APIError(f"Gemini blocked the decision input: {reason}", retryable=False)

        results: dict[str, DecisionResult] = {}
        errors: dict[str, str] = {}
        for name, answer in (decision.answers or {}).items():
            if name not in decisions:
                raise APIError("decision response contains unknown decisions", retryable=False)
            try:
                results[name] = _decode_answer(answer, decisions[name])
            except ValidationError:
                errors[name] = "invalid Gemini decision answer"
            except APIError as error:
                errors[name] = error.message

        usage = decision.usage
        return DecisionResponse(
            results=results,
            errors=errors,
            model=self._model,
            provider=self.provider,
            request_id=raw.request_id or "",
            input_tokens=usage.total_input_tokens if usage else None,
            output_tokens=usage.total_output_tokens if usage else None,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aio.aclose()


def _encode_question(definition: Decision) -> genai_decisions.QuestionParam:
    if isinstance(definition, Probability):
        return {"type": "noul", "instructions": definition.prompt}
    if isinstance(definition, Choice):
        return {
            "type": "choice",
            "instructions": definition.prompt,
            "criteria": dict(definition.options),
        }
    return {"type": "score", "instructions": definition.prompt, "criteria": definition.levels}


def _decode_answer(answer: genai_decisions.Answer, definition: Decision) -> DecisionResult:
    if isinstance(answer, genai_decisions.NoulAnswer):
        return ProbabilityResult(value=answer.noul, provider_data={"confidence": answer.confidence})
    if isinstance(answer, genai_decisions.ChoiceAnswer):
        return ChoiceResult(
            value=answer.choice,
            probabilities=answer.probabilities,
            provider_data={"confidence": answer.confidence},
        )
    if isinstance(answer, genai_decisions.ScoreAnswer):
        # Only canonical indices: int() would also take "00" or " 1", merging two keys
        # into one level and silently dropping a probability.
        if any(
            not (level.isascii() and level.isdigit()) or str(int(level)) != level
            for level in answer.probabilities
        ):
            raise APIError("score distribution is not keyed by level", retryable=False)
        probabilities = {int(level): p for level, p in answer.probabilities.items()}
        # The legend only describes levels; the requested levels are the scale. A score
        # answering another kind of decision is rejected by the shared validation.
        return ScoreResult(
            value=answer.score,
            probabilities=probabilities,
            levels=definition.levels if isinstance(definition, Score) else [],
            provider_data={"confidence": answer.confidence},
        )
    raise APIError("unsupported Gemini decision answer", retryable=False)
