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
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from livekit.agents import llm
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins import openai

from .log import logger


@dataclass
class WebPlugin:
    """OpenRouter web search plugin configuration."""
    id: str = "web"
    max_results: int = 5
    search_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {"id": self.id, "max_results": self.max_results}
        if self.search_prompt is not None:
            result["search_prompt"] = self.search_prompt
        return result


@dataclass
class ProviderPreferences:
    """OpenRouter provider routing preferences."""
    order: list[str] | None = None
    allow_fallbacks: bool | None = None  
    require_parameters: bool | None = None
    data_collection: Literal["allow", "deny"] | None = None
    only: list[str] | None = None
    ignore: list[str] | None = None
    quantizations: list[str] | None = None
    sort: Literal["price", "throughput", "latency"] | None = None
    max_price: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class _LLMOptions:
    model: str
    temperature: NotGivenOr[float]
    site_url: NotGivenOr[str]
    app_name: NotGivenOr[str]
    fallback_models: list[str] | None
    provider_preferences: ProviderPreferences | None
    plugins: list[WebPlugin] | None


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str = "auto",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        site_url: NotGivenOr[str] = NOT_GIVEN,
        app_name: NotGivenOr[str] = NOT_GIVEN,
        fallback_models: list[str] | None = None,
        provider_preferences: ProviderPreferences | None = None,
        plugins: list[WebPlugin] | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        Create a new instance of OpenRouter LLM.

        ``api_key`` must be set to your OpenRouter API key, either using the argument
        or by setting the ``OPENROUTER_API_KEY`` environmental variable.

        Args:
            model: OpenRouter model to use (default: "openrouter/auto" for auto-routing)
                   Supports OpenRouter shortcuts like :online, :nitro, :floor
            api_key: OpenRouter API key
            temperature: Sampling temperature
            site_url: Your site URL for OpenRouter analytics
            app_name: Your app name for OpenRouter analytics
            fallback_models: List of fallback models if primary model fails
            provider_preferences: OpenRouter provider routing preferences
            plugins: List of OpenRouter plugins (e.g., WebPlugin for search)
            timeout: HTTP request timeout
        """
        super().__init__()

        # Resolve API key
        resolved_api_key: str
        if api_key is NOT_GIVEN:
            env_key = os.environ.get("OPENROUTER_API_KEY")
            if not env_key:
                raise ValueError(
                    "OpenRouter API key not provided. "
                    "Please provide the api_key argument or set the OPENROUTER_API_KEY environment variable."
                )
            resolved_api_key = env_key
        else:
            resolved_api_key = str(api_key)

        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            site_url=site_url,
            app_name=app_name,
            fallback_models=fallback_models,
            provider_preferences=provider_preferences,
            plugins=plugins,
        )

        # Build OpenAI client with OpenRouter configuration
        import openai as oai_client
        
        # Prepare headers for OpenRouter analytics
        headers = {}
        if site_url and site_url is not NOT_GIVEN:
            headers["HTTP-Referer"] = site_url
        if app_name and app_name is not NOT_GIVEN:
            headers["X-Title"] = app_name
            
        # Create OpenAI client instance with custom headers
        client = oai_client.AsyncOpenAI(
            api_key=resolved_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers,
            timeout=timeout,
        )
        
        # Create LiveKit OpenAI LLM wrapper with the configured client
        self._client = openai.LLM(
            model=model,
            client=client,
            temperature=temperature,
        )

        logger.info(f"Initialized OpenRouter LLM with model: {model}")

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool] | None = None,
        tool_choice: llm.ToolChoice | None = None,
        **kwargs: Any,
    ) -> llm.LLMStream:
        """Forward chat requests to the underlying OpenAI client with OpenRouter routing."""
        # Convert None to NOT_GIVEN for OpenAI plugin compatibility
        actual_tool_choice = tool_choice if tool_choice is not None else NOT_GIVEN
        
        # Handle OpenRouter-specific routing features
        extra_kwargs = kwargs.get("extra_kwargs", {})
        if not isinstance(extra_kwargs, dict):
            extra_kwargs = {}
            
        openrouter_body = {}
        
        # Add provider preferences if specified
        if self._opts.provider_preferences:
            provider_dict = self._opts.provider_preferences.to_dict()
            if provider_dict:
                openrouter_body["provider"] = provider_dict
                logger.info(f"Using OpenRouter provider preferences: {provider_dict}")
        
        # Handle fallback models (legacy support)
        if self._opts.fallback_models:
            # Use OpenRouter's models parameter for fallback routing
            models_list = [self._opts.model] + self._opts.fallback_models
            openrouter_body["models"] = models_list
            logger.info(f"Using OpenRouter models parameter: {models_list}")
        
        # Add plugins if specified
        if self._opts.plugins:
            plugins_list = [plugin.to_dict() for plugin in self._opts.plugins]
            openrouter_body["plugins"] = plugins_list
            logger.info(f"Using OpenRouter plugins: {plugins_list}")
        
        # Add OpenRouter body parameters via extra_body in extra_kwargs
        if openrouter_body:
            extra_kwargs["extra_body"] = openrouter_body
            kwargs["extra_kwargs"] = extra_kwargs
        
        return self._client.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=actual_tool_choice,
            **kwargs,
        )

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._opts.model
    
