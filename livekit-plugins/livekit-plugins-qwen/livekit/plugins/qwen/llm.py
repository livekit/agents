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

"""Chat completions on Model Studio's OpenAI-compatible endpoint.

The wire protocol is plain ``openai.LLM`` against ``/compatible-mode/v1``. This subclass
fills in the region base URL and ``DASHSCOPE_API_KEY``, reports one stable provider label
(the OpenAI plugin reports the base URL host, which on Model Studio can be a per-workspace
domain), and pins Qwen's thinking mode off unless asked for.
"""

from __future__ import annotations

from typing import Any

import httpx
import openai

from livekit.agents.llm import ToolChoice
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

from ._utils import resolve_api_key, resolve_compat_url
from .models import DEFAULT_LLM_MODEL, DEFAULT_REGION, LLMModels, QwenRegion


class LLM(OpenAILLM):
    """Qwen chat completions on Alibaba Cloud Model Studio's OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        model: LLMModels | str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        region: QwenRegion = DEFAULT_REGION,
        base_url: str | None = None,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
        extra_body: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
    ) -> None:
        """Create a Qwen LLM.

        Args:
            model: Model Studio chat model id, for example ``"qwen-plus"`` or ``"qwen3-max"``.
            api_key: Model Studio API key; falls back to ``DASHSCOPE_API_KEY``.
            region: ``"intl"`` (Singapore) or ``"cn"`` (Beijing). API keys are region-bound.
            base_url: OpenAI-compatible base URL, for example a workspace-dedicated domain
                ``https://<WorkspaceId>.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1``.
                Overrides ``region``.
            enable_thinking: Qwen's deep-thinking mode. Off by default: several Qwen models
                enable it on their own, and LiveKit's LLM stream never reads
                ``delta.reasoning_content``, so a thinking turn plays as silence.
            thinking_budget: Maximum reasoning tokens when thinking is enabled.
            extra_body: Extra request fields, merged over the thinking settings.
            temperature, top_p, parallel_tool_calls, tool_choice, max_completion_tokens,
                timeout, max_retries, client: Passed through to ``openai.LLM``.

        Raises:
            ValueError: If no API key is available.
        """
        body: dict[str, Any] = {"enable_thinking": enable_thinking}
        if thinking_budget is not None:
            body["thinking_budget"] = thinking_budget
        if is_given(extra_body):
            body.update(extra_body)

        super().__init__(
            model=model,
            api_key=resolve_api_key(api_key),
            base_url=resolve_compat_url(base_url, region),
            client=client,
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
            max_retries=max_retries,
            extra_body=body,
        )

    @property
    def provider(self) -> str:
        return "Qwen"
