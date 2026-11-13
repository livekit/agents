# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlparse

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

from .models import PerplexityDecisionModels
from .version import __version__

_PERPLEXITY_HOST = "api.perplexity.ai"
_BASE_URL = f"https://{_PERPLEXITY_HOST}/v1"
_ATTRIBUTION_HEADER = {"X-Pplx-Integration": f"livekit-agents/{__version__}"}


class DecisionModel(llm.DecisionModel):
    """Perplexity's Decisions API (``POST /v1/decisions``).

    One request answers a batch of probability, choice, and score questions about the
    same input.

    Example::

        model = perplexity.DecisionModel()
        response = await model.evaluate(
            input=chat_ctx,
            decisions={"handoff": llm.Probability("The caller requests a human.")},
        )

    Args:
        model: Perplexity decision model. ``pplx-decider-flash-v1.3`` is the
            low-latency default for in-turn decisions. ``pplx-decider-v1.3`` is the
            larger, more accurate model for background checks.
        api_key: Required for the default base URL, where it defaults to
            PERPLEXITY_API_KEY. Other base URLs only receive an explicit ``api_key``.
        base_url: Defaults to ``https://api.perplexity.ai/v1``. For Decider for Mac, use
            ``http://127.0.0.1:8080/v1``; it needs no API key.
        mode: ``"web_grounding"`` runs one web search per question and adds the results
            to the input before answering. Each search is billed and adds network
            latency, so prefer it for background decisions, not end-of-turn gating.
        max_staleness: Maximum age in seconds of a fetched page when the input is a
            single URL and ``mode="web_grounding"``.
        http_session: Optional caller-owned session. Otherwise uses the job's HTTP
            session. Outside a job, use ``async with utils.http_context.open()``.
    """

    def __init__(
        self,
        *,
        model: PerplexityDecisionModels | str = "pplx-decider-flash-v1.3",
        api_key: str | None = None,
        base_url: str = _BASE_URL,
        mode: Literal["web_grounding"] | None = None,
        max_staleness: int | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=DecisionCapabilities(decision_kinds=["probability", "choice", "score"])
        )
        hosted = urlparse(base_url).hostname == _PERPLEXITY_HOST
        # The environment key is for Perplexity's API; never send it to another server.
        key = api_key or (os.environ.get("PERPLEXITY_API_KEY") if hosted else None)
        if not key and hosted:
            raise ValueError(
                "PERPLEXITY_API_KEY is required, either as argument or set "
                "PERPLEXITY_API_KEY environmental variable"
            )
        if max_staleness is not None and mode is None:
            raise ValueError("max_staleness requires mode='web_grounding'")
        if max_staleness is not None and max_staleness < 0:
            raise ValueError("max_staleness must be non-negative")

        self._model = model
        self._url = f"{base_url.rstrip('/')}/decisions"
        self._headers = {
            **_ATTRIBUTION_HEADER,
            **({"Authorization": f"Bearer {key}"} if key else {}),
        }
        grounding = {"mode": mode, "max_staleness": max_staleness}
        self._grounding: dict[str, Any] = {k: v for k, v in grounding.items() if v is not None}
        self._session = http_session

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Perplexity"

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        body = {
            "model": self._model,
            "state": input,
            "questions": {name: systemone.encode_question(d) for name, d in decisions.items()},
            **self._grounding,
        }
        session = self._session or utils.http_context.http_session()
        try:
            async with session.post(
                self._url,
                headers=self._headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as response:
                request_id = response.headers.get("x-request-id", "")
                if response.status != 200:
                    raise APIStatusError(
                        "Perplexity decision request failed",
                        status_code=response.status,
                        request_id=request_id or None,
                        body=await response.text(),
                        retryable=response.status in (408, 429) or response.status >= 500,
                    )
                parsed = systemone.Response.model_validate(await response.json())
            results, errors = systemone.decode_answers(
                parsed.answers, invalid="invalid Perplexity decision answer"
            )
            return DecisionResponse(
                results=results,
                errors=errors,
                model=parsed.model,
                provider=self.provider,
                request_id=parsed.id or request_id,
                input_tokens=parsed.usage.input_tokens,
                output_tokens=parsed.usage.output_tokens,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("Perplexity decision request timed out") from None
        except aiohttp.ContentTypeError:
            raise APIError("invalid Perplexity decision response", retryable=False) from None
        except (aiohttp.ClientError, OSError):
            raise APIConnectionError("Perplexity decision request failed") from None
        except ValueError:
            # pydantic.ValidationError and json.JSONDecodeError are both ValueErrors
            raise APIError("invalid Perplexity decision response", retryable=False) from None
