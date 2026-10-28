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

import gzip
import json
import os

import httpx2 as httpx
import msgpack
import openai
from openai.types import ReasoningEffort

from livekit.agents.llm import ToolChoice
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

from .models import CerebrasChatModels


class _PayloadTransport(httpx.AsyncBaseTransport):
    """Re-encodes JSON request bodies as msgpack and/or gzip before they are sent.

    Lives below the SDK client, so every per-request copy the SDK makes (``with_options``)
    shares it. See https://inference-docs.cerebras.ai/payload-optimization
    """

    def __init__(
        self, inner: httpx.AsyncBaseTransport, *, use_msgpack: bool, use_gzip: bool
    ) -> None:
        self._inner = inner
        self._use_msgpack = use_msgpack
        self._use_gzip = use_gzip

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.aread()
            if body:
                headers = request.headers.copy()
                del headers["content-length"]
                if self._use_msgpack:
                    body = msgpack.packb(json.loads(body))
                    headers["content-type"] = "application/vnd.msgpack"
                if self._use_gzip:
                    body = gzip.compress(body, compresslevel=5)
                    headers["content-encoding"] = "gzip"
                request = httpx.Request(
                    request.method,
                    request.url,
                    headers=headers,
                    content=body,
                    extensions=request.extensions,
                )
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | CerebrasChatModels = "gpt-oss-120b",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = "https://api.cerebras.ai/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
        gzip_compression: bool = True,
        msgpack_encoding: bool = True,
    ):
        """
        Create a new instance of Cerebras LLM.

        ``api_key`` must be set to your Cerebras API key, either using the argument or by setting
        the ``CEREBRAS_API_KEY`` environmental variable.

        When ``gzip_compression`` is True (default), request payloads are gzip-compressed,
        which can reduce TTFT for requests with large prompts.

        When ``msgpack_encoding`` is True (default), request payloads are encoded with msgpack
        binary format instead of JSON.
        """

        cerebras_api_key = _get_api_key(api_key)

        created_client = False
        if client is None and (gzip_compression or msgpack_encoding):
            transport = _PayloadTransport(
                httpx.AsyncHTTPTransport(
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=50,
                        keepalive_expiry=120,
                    )
                ),
                use_msgpack=msgpack_encoding,
                use_gzip=gzip_compression,
            )
            client = openai.AsyncClient(
                api_key=cerebras_api_key,
                base_url=base_url if is_given(base_url) else None,
                max_retries=max_retries if is_given(max_retries) else 0,
                http_client=httpx.AsyncClient(
                    timeout=timeout
                    if timeout
                    else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                    follow_redirects=True,
                    transport=transport,
                ),
            )
            created_client = True

        super().__init__(
            model=model,
            api_key=cerebras_api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
            max_retries=max_retries,
            _strict_tool_schema=False,
        )

        if created_client:
            self._owns_client = True


def _get_api_key(key: NotGivenOr[str]) -> str:
    cerebras_api_key = key if is_given(key) else os.environ.get("CEREBRAS_API_KEY")
    if not cerebras_api_key:
        raise ValueError(
            "CEREBRAS_API_KEY is required, either as argument or set "
            "CEREBRAS_API_KEY environmental variable"
        )
    return cerebras_api_key
