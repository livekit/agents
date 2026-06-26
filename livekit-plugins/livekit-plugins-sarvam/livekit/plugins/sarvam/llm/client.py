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
import platform
from typing import Any

import httpx
import openai
from openai.types import ReasoningEffort

from livekit.agents import __version__ as livekit_version
from livekit.agents.llm import ToolChoice
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.openai.llm import LLM as OpenAILLM

from .models import SarvamLLMModels

SARVAM_LLM_BASE_URL = "https://api.sarvam.ai/v1"
USER_AGENT = f"Livekit/{livekit_version} Python/{platform.python_version()}"
_SUPPORTED_MODELS = {
    "sarvam-m",
    "sarvam-30b",
    "sarvam-30b-16k",  # deprecated by Sarvam, kept for backward compatibility
    "sarvam-105b",
    "sarvam-105b-32k",  # deprecated by Sarvam, kept for backward compatibility
}
_ALLOWED_EXTRA_BODY_PARAMS = {
    "frequency_penalty",
    "max_tokens",
    "n",
    "presence_penalty",
    "seed",
    "stop",
    "wiki_grounding",
}


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
        stop: NotGivenOr[str | list[str]] = NOT_GIVEN,
        n: NotGivenOr[int] = NOT_GIVEN,
        seed: NotGivenOr[int] = NOT_GIVEN,
        frequency_penalty: NotGivenOr[float] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        extra_headers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        extra_body: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        Create a new instance of Sarvam LLM.

        ``api_key`` must be set to your Sarvam API key, either using the argument or by setting
        the ``SARVAM_API_KEY`` environment variable.
        """
        validated_model = _validate_model(model)
        sarvam_api_key = _get_api_key(api_key)
        merged_headers = dict(extra_headers) if is_given(extra_headers) else {}
        # Sarvam chat-completions auth and telemetry headers are always enforced.
        merged_headers["api-subscription-key"] = sarvam_api_key
        merged_headers["User-Agent"] = USER_AGENT

        merged_body = dict(extra_body) if is_given(extra_body) else {}
        if is_given(max_tokens):
            merged_body["max_tokens"] = max_tokens
        if is_given(wiki_grounding):
            merged_body["wiki_grounding"] = wiki_grounding
        if is_given(stop):
            merged_body["stop"] = stop
        if is_given(n):
            merged_body["n"] = n
        if is_given(seed):
            merged_body["seed"] = seed
        if is_given(frequency_penalty):
            merged_body["frequency_penalty"] = frequency_penalty
        if is_given(presence_penalty):
            merged_body["presence_penalty"] = presence_penalty
        filtered_body = _filter_extra_body(merged_body)

        super().__init__(
            model=validated_model,
            api_key=sarvam_api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            top_p=top_p,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            extra_headers=merged_headers,
            extra_body=filtered_body if filtered_body else NOT_GIVEN,
            timeout=timeout,
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


def _validate_model(model: str) -> str:
    if model not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported Sarvam model '{model}'. Supported models: {sorted(_SUPPORTED_MODELS)}"
        )
    return model


def _filter_extra_body(extra_body: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in extra_body.items() if k in _ALLOWED_EXTRA_BODY_PARAMS}
