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
from collections.abc import Mapping
from urllib.parse import urlsplit

import aiohttp

from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError, llm, utils
from livekit.agents.llm import _systemone as systemone
from livekit.agents.llm.decisions import Decision, DecisionCapabilities, DecisionResponse
from livekit.agents.types import APIConnectOptions

from ._common import client_headers, gateway_url, resolve_api_key
from .models import DecisionModels


class DecisionModel(llm.DecisionModel):
    """Typed decisions from the VLM Run gateway's System One API.

    Example::

        model = vlmrun.DecisionModel()
        response = await model.evaluate(
            input=chat_ctx,
            decisions={"handoff": llm.Probability("The caller requests a human.")},
        )

    Args:
        model: Gateway model id. Defaults to ``qwen/qwen3.8-27b``.
        api_key: VLM Run API key. Defaults to ``VLMRUN_API_KEY``.
        base_url: Gateway root URL. Defaults to ``VLMRUN_GATEWAY_BASE_URL``, then
            ``VLMRUN_GATEWAY_URL``, then ``https://gateway.vlm.run``.
        http_session: Optional caller-owned aiohttp session. Defaults to the job's HTTP
            session; outside a job, use ``async with utils.http_context.open()``.
    """

    def __init__(
        self,
        *,
        model: DecisionModels | str = "qwen/qwen3.8-27b",
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=DecisionCapabilities(decision_kinds=["probability", "choice", "score"])
        )
        key = resolve_api_key(api_key)
        if not key:
            raise ValueError(
                "VLMRUN_API_KEY is required, either as argument or set VLMRUN_API_KEY environment variable"  # noqa: E501
            )
        self._model = model
        root = gateway_url(base_url)
        self._url = f"{root}/typesafe/v1/systemone"
        # Host and port only: netloc would carry user:password@ into telemetry.
        parts = urlsplit(root)
        self._provider = f"{parts.hostname}:{parts.port}" if parts.port else (parts.hostname or "")
        self._headers = client_headers(key)
        self._session = http_session

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
                self._url,
                headers=self._headers,
                json={"model": self._model, "state": input, "questions": questions},
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
                allow_redirects=False,
            ) as response:
                request_id = response.headers.get("x-request-id", "")
                if response.status != 200:
                    raise APIStatusError(
                        "VLM Run decision request failed",
                        status_code=response.status,
                        request_id=request_id or None,
                        body=await response.text(),
                        retryable=response.status in (408, 429) or 500 <= response.status < 600,
                    )
                parsed = systemone.Response.model_validate(await response.json())
                request_id = parsed.id or request_id
            results, errors = systemone.decode_answers(
                parsed.answers, invalid="invalid VLM Run decision answer"
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
        except asyncio.TimeoutError:
            raise APITimeoutError("VLM Run decision request timed out") from None
        except aiohttp.ContentTypeError:
            raise APIError("invalid VLM Run decision response", retryable=False) from None
        except (aiohttp.ClientError, ConnectionError, OSError):
            raise APIConnectionError("VLM Run decision transport failed") from None
        except ValueError:
            raise APIError("invalid VLM Run decision response", retryable=False) from None
