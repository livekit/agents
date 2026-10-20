# Copyright 2023 LiveKit, Inc.
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
from pathlib import Path
from typing import Literal

import aiohttp

from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError, llm, utils
from livekit.agents.llm import _systemone as systemone
from livekit.agents.llm.decisions import Decision, DecisionCapabilities, DecisionResponse
from livekit.agents.types import APIConnectOptions

from ._http import Configuration, HTTPClient

DecisionModels = Literal["Microsoft-Decision-1"]


class DecisionModel(llm.DecisionModel):
    """Microsoft-Decision-1 through a Microsoft Foundry deployment's System One endpoint.

    Example::

        model = microsoft_ai.DecisionModel()
        response = await model.evaluate(
            input=chat_ctx,
            decisions={"handoff": llm.Probability("The caller requests a human.")},
        )

    Args:
        url: Full POST endpoint, or MICROSOFT_AI_DECISION_URL, e.g.
            ``https://<resource>.services.ai.azure.com/providers/microsoft/v1/systemone``.
        model: Foundry deployment name of Microsoft-Decision-1, or
            MICROSOFT_AI_DECISION_MODEL.
        api_key: Foundry resource key, or MICROSOFT_AI_DECISION_API_KEY, sent using
            auth_header.
        auth_header: api-key (the default) or Authorization (with Bearer prefix, for
            an Entra ID token), or MICROSOFT_AI_DECISION_AUTH_HEADER.
        headers: Explicit authentication headers instead of api_key/auth_header.
        http_session: Optional caller-owned aiohttp session. Defaults to the job's HTTP
            session; outside a job, use ``async with utils.http_context.open()``.
        env_file: Explicit dotenv file, or MICROSOFT_AI_ENV_FILE. Constructor
            arguments override environment variables, which override this file.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        model: DecisionModels | str | None = None,
        api_key: str | None = None,
        auth_header: Literal["Authorization", "api-key"] | None = None,
        headers: Mapping[str, str] | None = None,
        http_session: aiohttp.ClientSession | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        super().__init__(
            capabilities=DecisionCapabilities(decision_kinds=["probability", "choice", "score"])
        )
        config = Configuration(env_file)
        self._model = config.required(model, "MICROSOFT_AI_DECISION_MODEL")
        self._session = http_session
        self._client = HTTPClient(
            config=config,
            service="DECISION",
            url=url,
            api_key=api_key,
            headers=headers,
            http_session=http_session,
            auth_header=auth_header,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Microsoft AI"

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
                self._client.url,
                headers=self._client.headers,
                json={"model": self._model, "state": input, "questions": questions},
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
                allow_redirects=False,
            ) as response:
                request_id = response.headers.get("x-request-id") or response.headers.get(
                    "apim-request-id", ""
                )
                if response.status != 200:
                    raise APIStatusError(
                        "Microsoft AI decision request failed",
                        status_code=response.status,
                        request_id=request_id or None,
                        body=await response.text(),
                        retryable=response.status in (408, 429) or 500 <= response.status < 600,
                    )
                parsed = systemone.Response.model_validate(await response.json())
                request_id = parsed.id or request_id
            results, errors = systemone.decode_answers(
                parsed.answers, invalid="invalid Microsoft AI decision answer"
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
            raise APITimeoutError("Microsoft AI decision request timed out") from None
        except aiohttp.ContentTypeError:
            raise APIError("invalid Microsoft AI decision response", retryable=False) from None
        except (aiohttp.ClientError, ConnectionError, OSError):
            raise APIConnectionError("Microsoft AI decision transport failed") from None
        except ValueError:
            raise APIError("invalid Microsoft AI decision response", retryable=False) from None
