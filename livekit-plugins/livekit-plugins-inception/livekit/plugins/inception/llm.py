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
from typing import Any

import httpx
import openai

from livekit.agents.llm import ToolChoice
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

from .models import InceptionChatModels, InceptionReasoningEffort


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | InceptionChatModels = "mercury-2",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = "https://api.inceptionlabs.ai/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[InceptionReasoningEffort] = NOT_GIVEN,
        realtime: NotGivenOr[bool] = NOT_GIVEN,
    ):
        """
        Create a new instance of Inception LLM.

        ``api_key`` must be set to your Inception API key, either using the argument or by setting
        the ``INCEPTION_API_KEY`` environmental variable.

        ``reasoning_effort`` controls the depth of reasoning: "instant" is fastest with minimal
        reasoning, "low", "medium", and "high" offer progressively deeper reasoning.

        ``realtime`` reduces wait time for the first diffusion block (TTFT).
        """
        inception_api_key = _get_api_key(api_key)

        # Inception-specific params go in extra_body so reasoning_effort="instant" (not in
        # OpenAI's ReasoningEffort type) and realtime are passed through without type conflicts.
        extra_body: dict[str, Any] = {}
        if is_given(reasoning_effort):
            extra_body["reasoning_effort"] = reasoning_effort
        if is_given(realtime):
            extra_body["realtime"] = realtime

        super().__init__(
            model=model,
            api_key=inception_api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=extra_body if extra_body else NOT_GIVEN,
            _strict_tool_schema=False,
        )


def _get_api_key(key: NotGivenOr[str]) -> str:
    inception_api_key = key if is_given(key) else os.environ.get("INCEPTION_API_KEY")
    if not inception_api_key:
        raise ValueError(
            "INCEPTION_API_KEY is required, either as argument or set "
            "INCEPTION_API_KEY environmental variable"
        )
    return inception_api_key
