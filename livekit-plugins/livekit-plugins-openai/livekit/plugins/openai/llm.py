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

import asyncio
import os
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

import httpx

import openai
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
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
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    completion_create_params,
)
from openai.types.chat.chat_completion_chunk import Choice

from .log import logger
from .models import (
    CerebrasChatModels,
    ChatModels,
    DeepSeekChatModels,
    OctoChatModels,
    PerplexityChatModels,
    TelnyxChatModels,
    TogetherChatModels,
    XAIChatModels,
)
from .utils import AsyncAzureADTokenProvider, to_fnc_ctx

lk_oai_debug = int(os.getenv("LK_OPENAI_DEBUG", 0))


@dataclass
class _LLMOptions:
    model: str | ChatModels
    user: NotGivenOr[str]
    temperature: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]
    store: NotGivenOr[bool]
    metadata: NotGivenOr[dict[str, str]]
    max_completion_tokens: NotGivenOr[int]
    service_tier: NotGivenOr[str]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4o",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        store: NotGivenOr[bool] = NOT_GIVEN,
        metadata: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
        _provider_fmt: NotGivenOr[str] = NOT_GIVEN,
        service_tier: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of OpenAI LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """
        super().__init__()
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
        )
        self._provider_fmt = _provider_fmt or "openai"
        self._client = client or openai.AsyncClient(
            api_key=api_key if is_given(api_key) else None,
            base_url=base_url if is_given(base_url) else None,
            max_retries=0,
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
        """Get the model name for this LLM instance."""
        return self._opts.model

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
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
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
        )

    @staticmethod
    def with_cerebras(
        *,
        model: str | CerebrasChatModels = "llama3.1-8b",
        api_key: str | None = None,
        base_url: str = "https://api.cerebras.ai/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
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
        )

    @staticmethod
    def with_letta(
        *,
        agent_id: str,
        base_url: str = "https://api.letta.com/v1/voice-beta",
        api_key: str | None = None,
    ) -> LLM:
        """
        Create a new Letta-backed LLM instance connected to the specified Letta agent.

        Args:
            agent_id (str): The Letta agent ID (must be prefixed with 'agent-').
            base_url (str): The URL of the Letta server (e.g., from ngrok or Letta Cloud).
            api_key (str | None, optional): Optional API key for authentication, required if
                                            the Letta server enforces auth.

        Returns:
            LLM: A configured LLM instance for interacting with the given Letta agent.
        """

        base_url = f"{base_url}/{agent_id}"
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
            model="letta-fast",
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
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        model: str | ChatModels,
        provider_fmt: str,
        client: openai.AsyncClient,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._provider_fmt = provider_fmt
        self._client = client
        self._llm = llm
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        # current function call that we're waiting for full completion (args are streamed)
        # (defined inside the _run method to make sure the state is reset for each run/attempt)
        self._oai_stream: openai.AsyncStream[ChatCompletionChunk] | None = None
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._tool_index: int | None = None
        retryable = True

        try:
            chat_ctx, _ = self._chat_ctx.to_provider_format(format=self._provider_fmt)
            fnc_ctx = to_fnc_ctx(self._tools) if self._tools else openai.NOT_GIVEN
            if lk_oai_debug:
                tool_choice = self._extra_kwargs.get("tool_choice", NOT_GIVEN)
                logger.debug(
                    "chat.completions.create",
                    extra={
                        "fnc_ctx": fnc_ctx,
                        "tool_choice": tool_choice,
                        "chat_ctx": chat_ctx,
                    },
                )

            self._oai_stream = stream = await self._client.chat.completions.create(
                messages=cast(list[ChatCompletionMessageParam], chat_ctx),
                tools=fnc_ctx,
                model=self._model,
                stream_options={"include_usage": True},
                stream=True,
                timeout=httpx.Timeout(self._conn_options.timeout),
                **self._extra_kwargs,
            )

            thinking = asyncio.Event()
            async with stream:
                async for chunk in stream:
                    for choice in chunk.choices:
                        chat_chunk = self._parse_choice(chunk.id, choice, thinking)
                        if chat_chunk is not None:
                            retryable = False
                            self._event_ch.send_nowait(chat_chunk)

                    if chunk.usage is not None:
                        retryable = False
                        tokens_details = chunk.usage.prompt_tokens_details
                        cached_tokens = tokens_details.cached_tokens if tokens_details else 0
                        chunk = llm.ChatChunk(
                            id=chunk.id,
                            usage=llm.CompletionUsage(
                                completion_tokens=chunk.usage.completion_tokens,
                                prompt_tokens=chunk.usage.prompt_tokens,
                                prompt_cached_tokens=cached_tokens or 0,
                                total_tokens=chunk.usage.total_tokens,
                            ),
                        )
                        self._event_ch.send_nowait(chunk)

        except openai.APITimeoutError:
            raise APITimeoutError(retryable=retryable) from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=retryable,
            ) from None
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _parse_choice(
        self, id: str, choice: Choice, thinking: asyncio.Event
    ) -> llm.ChatChunk | None:
        delta = choice.delta

        # https://github.com/livekit/agents/issues/688
        # the delta can be None when using Azure OpenAI (content filtering)
        if delta is None:
            return None

        if delta.tool_calls:
            for tool in delta.tool_calls:
                if not tool.function:
                    continue

                call_chunk = None
                if self._tool_call_id and tool.id and tool.index != self._tool_index:
                    call_chunk = llm.ChatChunk(
                        id=id,
                        delta=llm.ChoiceDelta(
                            role="assistant",
                            content=delta.content,
                            tool_calls=[
                                llm.FunctionToolCall(
                                    arguments=self._fnc_raw_arguments or "",
                                    name=self._fnc_name or "",
                                    call_id=self._tool_call_id or "",
                                )
                            ],
                        ),
                    )
                    self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None

                if tool.function.name:
                    self._tool_index = tool.index
                    self._tool_call_id = tool.id
                    self._fnc_name = tool.function.name
                    self._fnc_raw_arguments = tool.function.arguments or ""
                elif tool.function.arguments:
                    self._fnc_raw_arguments += tool.function.arguments  # type: ignore

                if call_chunk is not None:
                    return call_chunk

        if choice.finish_reason in ("tool_calls", "stop") and self._tool_call_id:
            call_chunk = llm.ChatChunk(
                id=id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content=delta.content,
                    tool_calls=[
                        llm.FunctionToolCall(
                            arguments=self._fnc_raw_arguments or "",
                            name=self._fnc_name or "",
                            call_id=self._tool_call_id or "",
                        )
                    ],
                ),
            )
            self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
            return call_chunk

        delta.content = llm_utils.strip_thinking_tokens(delta.content, thinking)

        if not delta.content:
            return None

        return llm.ChatChunk(
            id=id,
            delta=llm.ChoiceDelta(content=delta.content, role="assistant"),
        )
