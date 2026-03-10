# Copyright 2025 LiveKit, Inc.
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

import json
import os
import platform
from typing import Any

import httpx
import openai
from openai.types import ReasoningEffort
from openai.types.chat import (
    ChatCompletionToolChoiceOptionParam,
    completion_create_params,
)

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    __version__ as livekit_version,
    llm,
)
from livekit.agents.llm import ChatContext, ToolChoice, utils as llm_utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.openai.llm import LLM as OpenAILLM, LLMStream as OpenAILLMStream

from .models import SarvamLLMModels

SARVAM_LLM_BASE_URL = "https://api.sarvam.ai/v1"
USER_AGENT = f"Livekit/{livekit_version} Python/{platform.python_version()}"


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | SarvamLLMModels = "sarvam-30b",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = SARVAM_LLM_BASE_URL,
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        wiki_grounding: NotGivenOr[bool] = NOT_GIVEN,
        extra_headers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        extra_body: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        Create a new instance of Sarvam LLM.

        ``api_key`` must be set to your Sarvam API key, either using the argument or by setting
        the ``SARVAM_API_KEY`` environment variable.
        """
        sarvam_api_key = _get_api_key(api_key)
        merged_headers = dict(extra_headers) if is_given(extra_headers) else {}
        # Sarvam chat-completions auth is subscription-key based.
        merged_headers.setdefault("api-subscription-key", sarvam_api_key)
        # Keep parity with Sarvam STT/TTS telemetry headers.
        merged_headers.setdefault("User-Agent", USER_AGENT)

        merged_body = dict(extra_body) if is_given(extra_body) else {}
        if is_given(max_tokens):
            merged_body["max_tokens"] = max_tokens
        if is_given(wiki_grounding):
            merged_body["wiki_grounding"] = wiki_grounding

        super().__init__(
            model=model,
            api_key=sarvam_api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            top_p=top_p,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            extra_headers=merged_headers,
            extra_body=merged_body if merged_body else NOT_GIVEN,
            timeout=timeout,
        )

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            completion_create_params.ResponseFormat | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> OpenAILLMStream:
        extra = {}
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.extra_body):
            extra["extra_body"] = self._opts.extra_body

        if is_given(self._opts.extra_headers):
            extra["extra_headers"] = self._opts.extra_headers

        if is_given(self._opts.extra_query):
            extra["extra_query"] = self._opts.extra_query

        if is_given(self._opts.metadata):
            extra["metadata"] = self._opts.metadata

        if is_given(self._opts.user):
            extra["user"] = self._opts.user

        if is_given(self._opts.max_completion_tokens):
            extra["max_completion_tokens"] = self._opts.max_completion_tokens

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        if is_given(self._opts.service_tier):
            extra["service_tier"] = self._opts.service_tier

        if is_given(self._opts.reasoning_effort):
            extra["reasoning_effort"] = self._opts.reasoning_effort

        if is_given(self._opts.safety_identifier):
            extra["safety_identifier"] = self._opts.safety_identifier

        if is_given(self._opts.prompt_cache_key):
            extra["prompt_cache_key"] = self._opts.prompt_cache_key

        if is_given(self._opts.top_p):
            extra["top_p"] = self._opts.top_p

        if is_given(self._opts.verbosity):
            extra["verbosity"] = self._opts.verbosity

        if is_given(self._opts.prompt_cache_retention):
            extra["prompt_cache_retention"] = self._opts.prompt_cache_retention

        parallel_tool_calls = (
            parallel_tool_calls if is_given(parallel_tool_calls) else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice  # type: ignore
        if is_given(tool_choice):
            oai_tool_choice: ChatCompletionToolChoiceOptionParam
            if isinstance(tool_choice, dict):
                oai_tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice["function"]["name"]},
                }
                extra["tool_choice"] = oai_tool_choice
            elif tool_choice in ("auto", "required", "none"):
                oai_tool_choice = tool_choice
                extra["tool_choice"] = oai_tool_choice

        if is_given(response_format):
            extra["response_format"] = llm_utils.to_openai_response_format(response_format)  # type: ignore

        return _SarvamLLMStream(
            self,
            model=self._opts.model,
            provider_fmt=self._provider_fmt,
            strict_tool_schema=self._strict_tool_schema,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Sarvam"


def _get_api_key(key: NotGivenOr[str]) -> str:
    sarvam_api_key = key if is_given(key) else os.environ.get("SARVAM_API_KEY")
    if not sarvam_api_key:
        raise ValueError(
            "SARVAM_API_KEY is required, either as argument or set SARVAM_API_KEY environment variable"
        )
    return sarvam_api_key


def _stringify_raw_error_body(body: object | None) -> str:
    if body is None:
        return "No response body"
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(body)


class _SarvamLLMStream(OpenAILLMStream):
    async def _run(self) -> None:
        try:
            await super()._run()
        except APIStatusError as e:
            raw_error = _stringify_raw_error_body(e.body)
            raise APIStatusError(
                message=f"Sarvam LLM API Error ({e.status_code}): {raw_error}",
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=e.retryable,
            ) from e
