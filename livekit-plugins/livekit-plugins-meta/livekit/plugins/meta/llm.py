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

import os

import openai

from livekit.agents.llm import ToolChoice
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.openai import LLM as OpenAILLM

from .models import ChatModels


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "Llama-4-Maverick-17B-128E-Instruct-FP8",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str | None = "https://api.llama.com/compat/v1/",
        client: openai.AsyncClient | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Llama LLM.

        ``api_key`` must be set to your Llama API key, either using the argument or by setting
        the ``LLAMA_API_KEY`` environmental variable.

        Args:
            `model` (str | ChatModels): The model to use.
            Defaults to `"Llama-4-Maverick-17B-128E-Instruct-FP8"`.
            `api_key` (str, optional): The Meta Llama API key.
            Defaults to the ``LLAMA_API_KEY`` environment variable.
            `base_url` (str, optional): The base URL for the Llama API. Defaults to None.
            `client` (openai.AsyncClient | None): The async client to use. Defaults to None.
            `top_k` (int, optional): The top K for the Llama API. Defaults to None.
            `max_completion_tokens` (int, optional): The max tokens for the Llama API.
            Defaults to None.
            `temperature` (float, optional): The temperature for the Llama API. Defaults to None.
            `parallel_tool_calls` (bool, optional): Whether to parallelize tool calls.
            Defaults to None.
            `tool_choice` (ToolChoice, optional): The tool choice for the Llama API.
            Defaults to `"auto"`.
        """

        super().__init__(
            model=model,
            api_key=_get_api_key(api_key),
            base_url=base_url,
            client=client,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )


def _get_api_key(key: NotGivenOr[str]) -> str:
    llama_api_key = key if is_given(key) else os.environ.get("LLAMA_API_KEY")
    if not llama_api_key:
        raise ValueError(
            "LLAMA_API_KEY is required, either as argument or set LLAMA_API_KEY environmental variable"  # noqa: E501
        )
    return llama_api_key
