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
from typing import Any

import httpx
import msgpack
import openai
from openai._models import FinalRequestOptions
from openai._utils import is_mapping
from openai.types import ReasoningEffort

from livekit.agents.llm import ToolChoice
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

from .models import CerebrasChatModels


class _CerebrasClient(openai.AsyncClient):
    """AsyncClient subclass that compresses request payloads via msgpack and/or gzip.

    Overrides _build_request() to serialize json_data directly to the target
    format, avoiding a JSON->dict->msgpack round-trip when msgpack is enabled.

    See https://inference-docs.cerebras.ai/payload-optimization
    """

    def __init__(
        self,
        *,
        use_msgpack: bool = False,
        use_gzip: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._use_msgpack = use_msgpack
        self._use_gzip = use_gzip

    def _build_request(
        self,
        options: FinalRequestOptions,
        *,
        retries_taken: int = 0,
    ) -> httpx.Request:
        if not (self._use_msgpack or self._use_gzip):
            return super()._build_request(options, retries_taken=retries_taken)

        json_data = options.json_data
        if json_data is not None:
            # merge extra_json (same logic as base class)
            if options.extra_json is not None:
                if is_mapping(json_data):
                    json_data = {**json_data, **options.extra_json}

            if self._use_msgpack:
                body = msgpack.packb(json_data)
                content_type = "application/vnd.msgpack"
            else:
                body = json.dumps(json_data, separators=(",", ":"), ensure_ascii=False).encode()
                content_type = "application/json"

            if self._use_gzip:
                body = gzip.compress(body, compresslevel=5)

            # bypass openapi_dumps() by switching to the content path
            options.json_data = None
            options.extra_json = None
            options.content = body

            existing = (
                dict(options.headers) if is_given(options.headers) and options.headers else {}
            )
            overrides: dict[str, str] = {"Content-Type": content_type}
            if self._use_gzip:
                overrides["Content-Encoding"] = "gzip"
            options.headers = existing | overrides

        return super()._build_request(options, retries_taken=retries_taken)


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | CerebrasChatModels = "llama3.1-8b",
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
            client = _CerebrasClient(
                use_msgpack=msgpack_encoding,
                use_gzip=gzip_compression,
                api_key=cerebras_api_key,
                base_url=base_url if is_given(base_url) else None,
                max_retries=max_retries if is_given(max_retries) else 0,
                http_client=httpx.AsyncClient(
                    timeout=timeout
                    if timeout
                    else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                    follow_redirects=True,
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=50,
                        keepalive_expiry=120,
                    ),
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
