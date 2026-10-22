from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Literal

import aiohttp
from pydantic import BaseModel, ValidationError

from .._exceptions import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    create_api_error_from_http,
)
from ..llm import _systemone as systemone
from ..llm.decisions import (
    Decision,
    DecisionCapabilities,
    DecisionModel as BaseDecisionModel,
    DecisionResponse,
)
from ..types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import http_context, is_given
from ._utils import (
    HEADER_REQUEST_ID,
    create_access_token,
    get_default_inference_url,
    get_inference_headers,
    resolve_credentials,
)

DecisionModels = Literal[
    "openai/gpt-6-luna-decisions",
    "typesafe/jev",
    "typesafe/jev-latest",
    "typesafe/jev-preview",
    "livekit/clef-omni",
]


class _GatewayError(BaseModel):
    type: str = ""
    error: str = ""
    hint: str = ""


class DecisionModel(BaseDecisionModel):
    """Decision models served through LiveKit Inference (``POST /v1/decisions``).

    Unlike the BYOK plugins, this calls the inference gateway with LiveKit
    credentials; the gateway calls the provider with LiveKit's key. Every provider
    speaks the same body, so ``model`` is the only thing that changes between them.

    Example::

        model = inference.DecisionModel("openai/gpt-6-luna-decisions")
        response = await model.evaluate(
            input=chat_ctx,
            decisions={"handoff": llm.Probability("The caller requests a human.")},
        )

    Credentials resolve from the arguments, then ``LIVEKIT_INFERENCE_API_KEY`` /
    ``LIVEKIT_INFERENCE_API_SECRET``, then ``LIVEKIT_API_KEY`` / ``LIVEKIT_API_SECRET``.
    A project the endpoint is not enabled for gets a non-retryable 403.
    """

    def __init__(
        self,
        model: DecisionModels | str = "openai/gpt-6-luna-decisions",
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            model: Gateway model id, ``"<provider>/<model>"``.
            base_url: Inference gateway base URL. Defaults to the environment's
                gateway (see ``get_default_inference_url``).
            api_key: Gateway API key (see the class docstring for resolution).
            api_secret: Gateway API secret (see the class docstring).
            http_session: Optional caller-owned session. Otherwise uses the job's
                HTTP session. Outside a job, use ``async with utils.http_context.open()``.
        """
        super().__init__(
            capabilities=DecisionCapabilities(decision_kinds=["probability", "choice", "score"])
        )
        self._model = model
        self._base_url = base_url if is_given(base_url) else get_default_inference_url()
        self._api_key, self._api_secret = resolve_credentials(
            api_key if is_given(api_key) else None,
            api_secret if is_given(api_secret) else None,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "livekit"

    def _headers(self) -> dict[str, str]:
        # Minted per request: a retry must not reuse a token that may have expired.
        return {
            **get_inference_headers(),
            "Authorization": f"Bearer {create_access_token(self._api_key, self._api_secret)}",
            "Content-Type": "application/json",
        }

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        questions = {name: systemone.encode_question(d) for name, d in decisions.items()}
        session = self._session or http_context.http_session()
        url = f"{self._base_url.rstrip('/')}/decisions"
        try:
            async with session.post(
                url,
                headers=self._headers(),
                json={"model": self._model, "state": input, "questions": questions},
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as response:
                request_id = response.headers.get(HEADER_REQUEST_ID)
                if response.status >= 400:
                    raise _error_from_response(
                        await response.text(), status=response.status, request_id=request_id
                    )
                parsed = systemone.Response.model_validate(await response.json())
            results, errors = systemone.decode_answers(parsed.answers)
            return DecisionResponse(
                results=results,
                errors=errors,
                model=parsed.model,
                provider=self.provider,
                request_id=request_id or parsed.id,
                input_tokens=parsed.usage.input_tokens,
                output_tokens=parsed.usage.output_tokens,
            )
        except asyncio.TimeoutError as exc:
            raise APITimeoutError("inference decision request timed out") from exc
        except aiohttp.ContentTypeError as exc:
            raise APIError("invalid inference decision response", retryable=False) from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError("inference gateway connection failed") from exc
        except ValueError as exc:
            raise APIError("invalid inference decision response", retryable=False) from exc


def _error_from_response(text: str, *, status: int, request_id: str | None) -> APIError:
    body: object = text
    message = "inference gateway returned an error"
    try:
        parsed = _GatewayError.model_validate_json(text)
    except ValidationError:
        pass
    else:
        body = parsed.model_dump()
        detail = parsed.error or parsed.type
        if detail:
            message = f"{message}: {detail}"
    return create_api_error_from_http(message, status=status, request_id=request_id, body=body)
