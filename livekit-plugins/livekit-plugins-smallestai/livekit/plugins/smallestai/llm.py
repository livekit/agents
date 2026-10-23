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

"""Smallest AI (Electron) LLM for LiveKit Agents.

Electron is an OpenAI-compatible chat-completions endpoint: same wire format,
streaming, tool calling, and JSON mode as the OpenAI Chat Completions API. This
plugin is a thin preset over :class:`livekit.plugins.openai.LLM` pointed at the
Smallest AI gateway, the same pattern used for other OpenAI-compatible
providers (Groq, Baseten, Cerebras, ...).
"""

from __future__ import annotations

import os

import httpx2 as httpx
import openai
from openai.types import ReasoningEffort

from livekit.agents.llm import ToolChoice
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

from .models import LLMModels
from .version import __version__

SMALLEST_LLM_BASE_URL = "https://api.smallest.ai/waves/v1"


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: LLMModels | str = "electron",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = SMALLEST_LLM_BASE_URL,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
    ) -> None:
        """
        Create a new instance of Smallest AI (Electron) LLM.

        ``api_key`` must be set to your Smallest AI API key, either using the argument
        or by setting the ``SMALLEST_API_KEY`` environment variable.

        ``model`` selects between ``"electron"`` (default) and ``"electron-2.0"``; both
        are served from the same ``base_url`` and accept the same request shape.

        .. note::
            ``"electron-2.0"`` is not yet live on the Smallest AI gateway and currently
            returns a 502 ``upstream model unavailable`` error. The model id is accepted
            here in anticipation of its rollout; use ``"electron"`` until it's available.
        """
        api_key = api_key if is_given(api_key) else os.environ.get("SMALLEST_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Smallest AI API key is required, either as argument or set "
                "SMALLEST_API_KEY environment variable"
            )

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            max_retries=max_retries,
            extra_headers={"X-Source": "livekit", "X-LiveKit-Version": __version__},
        )

    @property
    def provider(self) -> str:
        return "SmallestAI"
