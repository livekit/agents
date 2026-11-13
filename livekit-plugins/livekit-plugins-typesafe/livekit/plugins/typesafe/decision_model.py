from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    llm,
    utils,
)
from livekit.agents.llm import _systemone as systemone
from livekit.agents.llm.decisions import Decision, DecisionCapabilities, DecisionResponse
from livekit.agents.types import APIConnectOptions

from .models import DecisionModels, OpenRouterDecisionModels

TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"


class DecisionModel(llm.DecisionModel):
    """TypeSafe Jev, via the TypeSafe API or OpenRouter (:meth:`with_openrouter`).

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
        model: DecisionModels | str = "jev-1.13.0",
        api_key: str | None = None,
        endpoint: str = TYPESAFE_ENDPOINT,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=DecisionCapabilities(decision_kinds=["probability", "choice", "score"])
        )
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("TypeSafe requires api_key or TYPESAFE_API_KEY")
        self._api_key = key
        self._model = model
        self._endpoint = endpoint
        self._session = http_session
        self._provider = "typesafe"

    @classmethod
    def with_openrouter(
        cls,
        *,
        model: OpenRouterDecisionModels | str = "typesafe/jev-1.13",
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> DecisionModel:
        """Use OpenRouter's Decisions API. ``api_key`` defaults to OPENROUTER_API_KEY."""
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("TypeSafe via OpenRouter requires api_key or OPENROUTER_API_KEY")
        instance = cls(
            model=model, api_key=key, endpoint=OPENROUTER_ENDPOINT, http_session=http_session
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
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        questions = {name: systemone.encode_question(d) for name, d in decisions.items()}
        session = self._session or utils.http_context.http_session()
        try:
            async with session.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "state": input, "questions": questions},
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as response:
                if response.status >= 400:
                    raise APIStatusError(
                        "TypeSafe decision request failed",
                        status_code=response.status,
                        request_id=response.headers.get("x-request-id"),
                    )
                parsed = systemone.Response.model_validate(await response.json())
                request_id = parsed.id or response.headers.get("x-request-id", "")
            results, errors = systemone.decode_answers(
                parsed.answers, invalid="invalid Jev decision answer"
            )
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
            raise APITimeoutError("TypeSafe request timed out") from exc
        except aiohttp.ContentTypeError as exc:
            raise APIError("invalid TypeSafe decision response", retryable=False) from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError("TypeSafe connection failed") from exc
        except ValueError as exc:
            raise APIError("invalid TypeSafe decision response", retryable=False) from exc
