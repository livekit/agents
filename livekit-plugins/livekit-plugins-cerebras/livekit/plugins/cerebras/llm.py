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

import os

from livekit.agents.inference.llm import ReasoningEffort
from livekit.agents.llm import ToolChoice
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

from .models import CerebrasChatModels


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | CerebrasChatModels = "gpt-oss-120b",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = "https://api.cerebras.ai/v1",
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
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
        binary format instead of JSON. See https://inference-docs.cerebras.ai/payload-optimization
        """
        super().__init__(
            model=model,
            api_key=_get_api_key(api_key),
            base_url=base_url,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            _strict_tool_schema=False,
            _gzip_body=gzip_compression,
            _msgpack_body=msgpack_encoding,
        )


def _get_api_key(key: NotGivenOr[str]) -> str:
    cerebras_api_key = key if is_given(key) else os.environ.get("CEREBRAS_API_KEY")
    if not cerebras_api_key:
        raise ValueError(
            "CEREBRAS_API_KEY is required, either as argument or set "
            "CEREBRAS_API_KEY environmental variable"
        )
    return cerebras_api_key
