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

import httpx
import openai

from livekit.agents.llm import ToolChoice
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins.openai import LLM as OpenAILLM

from ._utils import ATTRIBUTION_HEADERS, SPRAG_BASE_URL, resolve_api_key
from .models import LLMModels


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | LLMModels = "symphony",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = SPRAG_BASE_URL,
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        Create a new instance of Sprag LLM.

        ``api_key`` must be set to your Sprag API key, either using the argument or by
        setting the ``SPRAG_API_KEY`` environmental variable.
        """
        super().__init__(
            model=model,
            api_key=resolve_api_key(api_key),
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            top_p=top_p,
            timeout=timeout,
            extra_headers=ATTRIBUTION_HEADERS,
        )

    @property
    def provider(self) -> str:
        return "Sprag"
