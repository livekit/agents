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

import os
from typing import Literal

import httpx
import openai
from openai.types import ReasoningEffort

from livekit.agents.llm import ToolChoice
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

LLMModels = Literal[
    "MiniMax-M2.7",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
]


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | LLMModels = "MiniMax-M2.7",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        user: NotGivenOr[str] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        base_url: NotGivenOr[str] = "https://api.minimax.io/v1",
        metadata: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        service_tier: NotGivenOr[str] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
    ):
        """
        Create a new instance of MiniMax LLM.

        ``api_key`` must be set to your MiniMax API key, either using the argument or by setting
        the ``MINIMAX_API_KEY`` environmental variable.
        """

        super().__init__(
            model=model,
            api_key=_get_api_key(api_key),
            base_url=base_url,
            client=client,
            user=user,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata,
            max_completion_tokens=max_completion_tokens,
        )


def _get_api_key(key: NotGivenOr[str]) -> str:
    api_key = key if is_given(key) else os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError(
            "MINIMAX_API_KEY is required, either as argument or set MINIMAX_API_KEY environmental variable"  # noqa: E501
        )
    return api_key
