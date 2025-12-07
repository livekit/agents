# Copyright 2023 LiveKit, Inc.
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
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

import openai
from livekit.agents import llm
from livekit.agents.inference.llm import LLMStream as _LLMStream
from livekit.agents.llm import ToolChoice, utils as llm_utils
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types import ReasoningEffort
from openai.types.chat import ChatCompletionToolChoiceOptionParam, completion_create_params

from .models import (
    CerebrasChatModels,
    ChatModels,
    CometAPIChatModels,
    DeepSeekChatModels,
    NebiusChatModels,
    OctoChatModels,
    OpenRouterProviderPreferences,
    OpenRouterWebPlugin,
    PerplexityChatModels,
    TelnyxChatModels,
    TogetherChatModels,
    XAIChatModels,
    _supports_reasoning_effort,
)
from .utils import AsyncAzureADTokenProvider

lk_oai_debug = int(os.getenv("LK_OPENAI_DEBUG", 0))

Verbosity = Literal["low", "medium", "high"]
PromptCacheRetention = Literal["in_memory", "24h"]


@dataclass
class _LLMOptions:
    model: str | ChatModels
    user: NotGivenOr[str]
    safety_identifier: NotGivenOr[str]
    prompt_cache_key: NotGivenOr[str]
    temperature: NotGivenOr[float]
    top_p: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]
    store: NotGivenOr[bool]
    metadata: NotGivenOr[dict[str, str]]
    max_completion_tokens: NotGivenOr[int]
    service_tier: NotGivenOr[str]
    reasoning_effort: NotGivenOr[ReasoningEffort]
    verbosity: NotGivenOr[Verbosity]
    prompt_cache_retention: NotGivenOr[PromptCacheRetention]
    extra_body: NotGivenOr[dict[str, Any]]
    extra_headers: NotGivenOr[dict[str, str]]
    extra_query: NotGivenOr[dict[str, str]]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4.1",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        store: NotGivenOr[bool] = NOT_GIVEN,
        metadata: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        max_retries: NotGivenOr[int] = NOT_GIVEN,
        service_tier: NotGivenOr[str] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        verbosity: NotGivenOr[Verbosity] = NOT_GIVEN,
        prompt_cache_retention: NotGivenOr[PromptCacheRetention] = NOT_GIVEN,
        extra_body: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        extra_headers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        extra_query: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        _provider_fmt: NotGivenOr[str] = NOT_GIVEN,
        _strict_tool_schema: bool = True,
    ) -> None:
        """
        Create a new instance of OpenAI LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """
        super().__init__()

        if not is_given(reasoning_effort) and _supports_reasoning_effort(model):
            if model == "gpt-5.1":
                reasoning_effort = "none"  # type: ignore[assignment]
            else:
                reasoning_effort = "minimal"

        self._opts = _LLMOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            max_completion_tokens=max_completion_tokens,
            service_tier=service_tier,
            reasoning_effort=reasoning_effort,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            top_p=top_p,
            verbosity=verbosity,
            prompt_cache_retention=prompt_cache_retention,
            extra_body=extra_body,
            extra_headers=extra_headers,
            extra_query=extra_query,
        )
        self._provider_fmt = _provider_fmt or "openai"
        self._strict_tool_schema = _strict_tool_schema
        self._client = client or openai.AsyncClient(
            api_key=api_key if is_given(api_key) else None,
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

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return self._client._base_url.netloc.decode("utf-8")

    @staticmethod
    def with_azure(
        *,
        model: str | ChatModels = "gpt-4o",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        verbosity: NotGivenOr[Verbosity] = NOT_GIVEN,
    ) -> LLM:
        """
        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        """  # noqa: E501

        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout
            if timeout
            else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        )  # type: ignore

        return LLM(
            model=model,
            client=azure_client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            top_p=top_p,
            verbosity=verbosity,
        )

    @staticmethod
    def with_cerebras(
        *,
        model: str | CerebrasChatModels = "llama-4-scout-17b-16e-instruct",
        api_key: str | None = None,
        base_url: str = "https://api.cerebras.ai/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of Cerebras LLM.

        ``api_key`` must be set to your Cerebras API key, either using the argument or by setting
        the ``CEREBRAS_API_KEY`` environment variable.
        """

        api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if api_key is None:
            raise ValueError(
                "Cerebras API key is required, either as argument or set CEREBRAS_API_KEY environment variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
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
            _strict_tool_schema=False,
        )

    @staticmethod
    def with_fireworks(
        *,
        model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
        api_key: str | None = None,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of Fireworks LLM.

        ``api_key`` must be set to your Fireworks API key, either using the argument or by setting
        the ``FIREWORKS_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if api_key is None:
            raise ValueError(
                "Fireworks API key is required, either as argument or set FIREWORKS_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
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
        )

    @staticmethod
    def with_x_ai(
        *,
        model: str | XAIChatModels = "grok-3-fast",
        api_key: str | None = None,
        base_url: str = "https://api.x.ai/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of XAI LLM.

        ``api_key`` must be set to your XAI API key, either using the argument or by setting
        the ``XAI_API_KEY`` environmental variable.
        """
        api_key = api_key or os.environ.get("XAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "XAI API key is required, either as argument or set XAI_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            # TODO(long): add provider fmt for grok
            reasoning_effort=reasoning_effort,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            top_p=top_p,
        )

    @staticmethod
    def with_openrouter(
        *,
        model: str = "auto",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        client: openai.AsyncClient | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        fallback_models: list[str] | None = None,
        provider: OpenRouterProviderPreferences | None = None,
        plugins: list[OpenRouterWebPlugin] | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> LLM:
        """
        Create a new instance of OpenRouter LLM.

        ``api_key`` must be set to your OpenRouter API key, either using the argument or by setting
        the ``OPENROUTER_API_KEY`` environment variable.
        """

        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "OpenRouter API key is required, either as argument or set OPENROUTER_API_KEY environment variable"
            )

        # Set up analytics headers for OpenRouter
        default_headers: dict[str, str] = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name

        # Build OpenRouter-specific request body
        or_body: dict[str, Any] = {}
        if provider:
            or_body["provider"] = provider
        if fallback_models:
            # Set fallback models for routing
            or_body["models"] = [model, *fallback_models]
        if plugins:
            or_body["plugins"] = [
                {k: v for k, v in asdict(p).items() if v is not None} for p in plugins
            ]

        return LLM(
            model=model,
            api_key=api_key,
            client=client,
            base_url=base_url,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            top_p=top_p,
            extra_body=or_body,
            extra_headers=default_headers,
            timeout=timeout,
        )

    @staticmethod
    def with_deepseek(
        *,
        model: str | DeepSeekChatModels = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of DeepSeek LLM.

        ``api_key`` must be set to your DeepSeek API key, either using the argument or by setting
        the ``DEEPSEEK_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError(
                "DeepSeek API key is required, either as argument or set DEEPSEEK_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
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
        )

    @staticmethod
    def with_cometapi(
        *,
        model: str | CometAPIChatModels = "gpt-5-chat-latest",
        api_key: str | None = None,
        base_url: str = "https://api.cometapi.com/v1/",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of CometAPI LLM.

        ``api_key`` must be set to your CometAPI API key, either using the argument or by setting
        the ``COMETAPI_API_KEY`` environmental variable.

        CometAPI provides access to 500+ AI models from multiple providers including OpenAI,
        Anthropic, Google, xAI, DeepSeek, and Qwen through a unified API.

        Get your API key at: https://api.cometapi.com/console/token
        Learn more: https://www.cometapi.com/?utm_source=livekit&utm_campaign=integration&utm_medium=integration&utm_content=integration
        """

        api_key = api_key or os.environ.get("COMETAPI_API_KEY")
        if api_key is None:
            raise ValueError(
                "CometAPI API key is required, either as argument or set COMETAPI_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
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
        )

    @staticmethod
    def with_octo(
        *,
        model: str | OctoChatModels = "llama-2-13b-chat",
        api_key: str | None = None,
        base_url: str = "https://text.octoai.run/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of OctoAI LLM.

        ``api_key`` must be set to your OctoAI API key, either using the argument or by setting
        the ``OCTOAI_TOKEN`` environmental variable.
        """

        api_key = api_key or os.environ.get("OCTOAI_TOKEN")
        if api_key is None:
            raise ValueError(
                "OctoAI API key is required, either as argument or set OCTOAI_TOKEN environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
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
        )

    @staticmethod
    def with_ollama(
        *,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434/v1",
        client: openai.AsyncClient | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of Ollama LLM.
        """

        return LLM(
            model=model,
            api_key="ollama",
            base_url=base_url,
            client=client,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            top_p=top_p,
        )

    @staticmethod
    def with_ovhcloud(
        *,
        model: str = "gpt-oss-120b",
        api_key: str | None = None,
        base_url: str = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of OVHcloud AI Endpoints LLM.

        ``api_key`` must be set to your OVHcloud AI Endpoints API key, either using the argument or by setting
        the ``OVHCLOUD_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("OVHCLOUD_API_KEY")
        if api_key is None:
            raise ValueError(
                "OVHcloud AI Endpoints API key is required, either as argument or set OVHCLOUD_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
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
        )

    @staticmethod
    def with_perplexity(
        *,
        model: str | PerplexityChatModels = "llama-3.1-sonar-small-128k-chat",
        api_key: str | None = None,
        base_url: str = "https://api.perplexity.ai",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of PerplexityAI LLM.

        ``api_key`` must be set to your TogetherAI API key, either using the argument or by setting
        the ``PERPLEXITY_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if api_key is None:
            raise ValueError(
                "Perplexity AI API key is required, either as argument or set PERPLEXITY_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
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
        )

    @staticmethod
    def with_together(
        *,
        model: str | TogetherChatModels = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key: str | None = None,
        base_url: str = "https://api.together.xyz/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of TogetherAI LLM.

        ``api_key`` must be set to your TogetherAI API key, either using the argument or by setting
        the ``TOGETHER_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if api_key is None:
            raise ValueError(
                "Together AI API key is required, either as argument or set TOGETHER_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
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
        )

    @staticmethod
    def with_telnyx(
        *,
        model: str | TelnyxChatModels = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        api_key: str | None = None,
        base_url: str = "https://api.telnyx.com/v2/ai",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of Telnyx LLM.

        ``api_key`` must be set to your Telnyx API key, either using the argument or by setting
        the ``TELNYX_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("TELNYX_API_KEY")
        if api_key is None:
            raise ValueError(
                "Telnyx AI API key is required, either as argument or set TELNYX_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
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
        )

    @staticmethod
    def with_nebius(
        *,
        model: str | NebiusChatModels = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        api_key: str | None = None,
        base_url: str = "https://api.studio.nebius.com/v1/",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        safety_identifier: NotGivenOr[str] = NOT_GIVEN,
        prompt_cache_key: NotGivenOr[str] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of Nebius LLM.

        ``api_key`` must be set to your Nebius API key, either using the argument or by setting
        the ``NEBIUS_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("NEBIUS_API_KEY")
        if api_key is None:
            raise ValueError(
                "Nebius API key is required, either as argument or set NEBIUS_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
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
        )

    @staticmethod
    def with_letta(
        *,
        agent_id: str,
        base_url: str = "https://api.letta.com/v1/chat/completions",
        api_key: str | None = None,
    ) -> LLM:
        """
        Create a new Letta-backed LLM instance connected to the specified Letta agent.

        Args:
            agent_id (str): The Letta agent ID (must be prefixed with 'agent-').
            base_url (str): The URL of the Letta server (e.g., http://localhost:8283/v1/chat/completions for local or https://api.letta.com/v1/chat/completions for cloud).
            api_key (str | None, optional): Optional API key for authentication, required if
                                            the Letta server enforces auth.

        Returns:
            LLM: A configured LLM instance for interacting with the given Letta agent.
        """

        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Invalid URL scheme: '{parsed.scheme}'. Must be 'http' or 'https'.")
        if not parsed.netloc:
            raise ValueError(f"URL '{base_url}' is missing a network location (e.g., domain name).")

        api_key = api_key or os.environ.get("LETTA_API_KEY")
        if api_key is None:
            raise ValueError(
                "Letta API key is required, either as argument or set LETTA_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=agent_id,
            api_key=api_key,
            base_url=base_url,
            client=None,
            user=NOT_GIVEN,
            temperature=NOT_GIVEN,
            parallel_tool_calls=NOT_GIVEN,
            tool_choice=NOT_GIVEN,
        )

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            completion_create_params.ResponseFormat | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
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

        return LLMStream(
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


class LLMStream(_LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        model: str | ChatModels,
        provider_fmt: str,
        strict_tool_schema: bool,
        client: openai.AsyncClient,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            llm,
            model=model,
            provider_fmt=provider_fmt,
            strict_tool_schema=strict_tool_schema,
            client=client,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            extra_kwargs=extra_kwargs,
        )
