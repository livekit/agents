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
import datetime
import os
from dataclasses import dataclass
from typing import Any, Literal, MutableSet, Union

import aiohttp
import httpx
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    llm,
)
from livekit.agents.llm import ToolChoice, _create_ai_function_info
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice

from ._oai_api import build_oai_function_description
from .log import logger
from .models import (
    CerebrasChatModels,
    ChatModels,
    DeepSeekChatModels,
    GroqChatModels,
    OctoChatModels,
    PerplexityChatModels,
    TelnyxChatModels,
    TogetherChatModels,
    VertexModels,
    XAIChatModels,
)
from .utils import AsyncAzureADTokenProvider, build_oai_message


@dataclass
class LLMOptions:
    model: str | ChatModels
    user: str | None
    temperature: float | None
    parallel_tool_calls: bool | None
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto"
    store: bool | None = None
    metadata: dict[str, str] | None = None


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        user: str | None = None,
        client: openai.AsyncClient | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
        store: bool | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """
        Create a new instance of OpenAI LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """
        super().__init__()
        self._capabilities = llm.LLMCapabilities(supports_choices_on_int=True)

        self._opts = LLMOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
        )
        self._client = client or openai.AsyncClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )
        self._running_fncs: MutableSet[asyncio.Task[Any]] = set()

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
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        """
        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        """

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
        base_url: str | None = "https://api.cerebras.ai/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        """
        Create a new instance of Cerebras LLM.

        ``api_key`` must be set to your Cerebras API key, either using the argument or by setting
        the ``CEREBRAS_API_KEY`` environmental variable.
        @integrations:cerebras:llm
        """

        api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if api_key is None:
            raise ValueError(
                "Cerebras API key is required, either as argument or set CEREBAAS_API_KEY environmental variable"
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
    def with_vertex(
        *,
        model: str | VertexModels = "google/gemini-2.0-flash-exp",
        project_id: str | None = None,
        location: str = "us-central1",
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        """
        Create a new instance of VertexAI LLM.

        `GOOGLE_APPLICATION_CREDENTIALS` environment variable must be set to the path of the service account key file.
        """
        project_id = project_id
        location = location
        _gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if _gac is None:
            logger.warning(
                "`GOOGLE_APPLICATION_CREDENTIALS` environment variable is not set. please set it to the path of the service account key file. Otherwise, use any of the other Google Cloud auth methods."
            )

        try:
            from google.auth._default_async import default_async
            from google.auth.transport._aiohttp_requests import Request
        except ImportError:
            raise ImportError(
                "Google Auth dependencies not found. Please install with: `pip install livekit-plugins-openai[vertex]`"
            )

        class AuthTokenRefresher(openai.AsyncClient):
            def __init__(self, **kwargs: Any) -> None:
                self.creds, self.project = default_async(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                project = project_id or self.project
                base_url = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/endpoints/openapi"
                kwargs.update({"base_url": base_url})
                super().__init__(api_key="DUMMY", **kwargs)
                self.refresh_threshold = 600  # 10 minutes

            def _token_needs_refresh(self) -> bool:
                if not self.creds or not self.creds.valid:
                    return True
                expiry = self.creds.expiry
                if expiry is None:
                    return True
                remaining = (expiry - datetime.datetime.utcnow()).total_seconds()
                return remaining < self.refresh_threshold

            async def _refresh_credentials(self) -> None:
                if self.creds and self.creds.valid and not self._token_needs_refresh():
                    return
                async with aiohttp.ClientSession(auto_decompress=False) as session:
                    auth_req = Request(session=session)
                    await self.creds.refresh(auth_req)
                self.api_key = self.creds.token

        client = AuthTokenRefresher(
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

        vertex_llm = LLM(
            model=model,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )
        vertex_llm._capabilities = llm.LLMCapabilities(supports_choices_on_int=False)
        return vertex_llm

    @staticmethod
    def with_fireworks(
        *,
        model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
        api_key: str | None = None,
        base_url: str | None = "https://api.fireworks.ai/inference/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        """
        Create a new instance of Fireworks LLM.

        ``api_key`` must be set to your Fireworks API key, either using the argument or by setting
        the ``FIREWORKS_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if api_key is None:
            raise ValueError(
                "Fireworks API key is required, either as argument or set FIREWORKS_API_KEY environmental variable"
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
        model: str | XAIChatModels = "grok-2-public",
        api_key: str | None = None,
        base_url: str | None = "https://api.x.ai/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ):
        """
        Create a new instance of XAI LLM.

        ``api_key`` must be set to your XAI API key, either using the argument or by setting
        the ``XAI_API_KEY`` environmental variable.
        """
        api_key = api_key or os.environ.get("XAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "XAI API key is required, either as argument or set XAI_API_KEY environmental variable"
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
    def with_groq(
        *,
        model: str | GroqChatModels = "llama3-8b-8192",
        api_key: str | None = None,
        base_url: str | None = "https://api.groq.com/openai/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        """
        Create a new instance of Groq LLM.

        ``api_key`` must be set to your Groq API key, either using the argument or by setting
        the ``GROQ_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if api_key is None:
            raise ValueError(
                "Groq API key is required, either as argument or set GROQ_API_KEY environmental variable"
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
    def with_deepseek(
        *,
        model: str | DeepSeekChatModels = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = "https://api.deepseek.com/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        """
        Create a new instance of DeepSeek LLM.

        ``api_key`` must be set to your DeepSeek API key, either using the argument or by setting
        the ``DEEPSEEK_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError(
                "DeepSeek API key is required, either as argument or set DEEPSEEK_API_KEY environmental variable"
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
        base_url: str | None = "https://text.octoai.run/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        """
        Create a new instance of OctoAI LLM.

        ``api_key`` must be set to your OctoAI API key, either using the argument or by setting
        the ``OCTOAI_TOKEN`` environmental variable.
        """

        api_key = api_key or os.environ.get("OCTOAI_TOKEN")
        if api_key is None:
            raise ValueError(
                "OctoAI API key is required, either as argument or set OCTOAI_TOKEN environmental variable"
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
        base_url: str | None = "http://localhost:11434/v1",
        client: openai.AsyncClient | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
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
        base_url: str | None = "https://api.perplexity.ai",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        """
        Create a new instance of PerplexityAI LLM.

        ``api_key`` must be set to your TogetherAI API key, either using the argument or by setting
        the ``PERPLEXITY_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if api_key is None:
            raise ValueError(
                "Perplexity AI API key is required, either as argument or set PERPLEXITY_API_KEY environmental variable"
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
        base_url: str | None = "https://api.together.xyz/v1",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        """
        Create a new instance of TogetherAI LLM.

        ``api_key`` must be set to your TogetherAI API key, either using the argument or by setting
        the ``TOGETHER_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if api_key is None:
            raise ValueError(
                "Together AI API key is required, either as argument or set TOGETHER_API_KEY environmental variable"
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
        base_url: str | None = "https://api.telnyx.com/v2/ai",
        client: openai.AsyncClient | None = None,
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        """
        Create a new instance of Telnyx LLM.

        ``api_key`` must be set to your Telnyx API key, either using the argument or by setting
        the ``TELNYX_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("TELNYX_API_KEY")
        if api_key is None:
            raise ValueError(
                "Telnyx AI API key is required, either as argument or set TELNYX_API_KEY environmental variable"
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
    def create_azure_client(
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
        user: str | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = "auto",
    ) -> LLM:
        logger.warning("This alias is deprecated. Use LLM.with_azure() instead")
        return LLM.with_azure(
            model=model,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]]
        | None = None,
    ) -> "LLMStream":
        if parallel_tool_calls is None:
            parallel_tool_calls = self._opts.parallel_tool_calls

        if tool_choice is None:
            tool_choice = self._opts.tool_choice

        if temperature is None:
            temperature = self._opts.temperature

        return LLMStream(
            self,
            client=self._client,
            model=self._opts.model,
            user=self._opts.user,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
            n=n,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        client: openai.AsyncClient,
        model: str | ChatModels,
        user: str | None,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        fnc_ctx: llm.FunctionContext | None,
        temperature: float | None,
        n: int | None,
        parallel_tool_calls: bool | None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]],
    ) -> None:
        super().__init__(
            llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options
        )
        self._client = client
        self._model = model
        self._llm: LLM = llm

        self._user = user
        self._temperature = temperature
        self._n = n
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice

    async def _run(self) -> None:
        if hasattr(self._llm._client, "_refresh_credentials"):
            await self._llm._client._refresh_credentials()

        # current function call that we're waiting for full completion (args are streamed)
        # (defined inside the _run method to make sure the state is reset for each run/attempt)
        self._oai_stream: openai.AsyncStream[ChatCompletionChunk] | None = None
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._tool_index: int | None = None
        retryable = True

        try:
            opts: dict[str, Any] = dict()
            if self._fnc_ctx and len(self._fnc_ctx.ai_functions) > 0:
                fncs_desc = []
                for fnc in self._fnc_ctx.ai_functions.values():
                    fncs_desc.append(
                        build_oai_function_description(fnc, self._llm._capabilities)
                    )

                opts["tools"] = fncs_desc
                if self._parallel_tool_calls is not None:
                    opts["parallel_tool_calls"] = self._parallel_tool_calls

                if self._tool_choice is not None:
                    if isinstance(self._tool_choice, ToolChoice):
                        # specific function
                        opts["tool_choice"] = {
                            "type": "function",
                            "function": {"name": self._tool_choice.name},
                        }
                    else:
                        opts["tool_choice"] = self._tool_choice

            if self._llm._opts.metadata is not None:
                # some OpenAI-like API doesn't support having a `metadata` field. (Even None)
                opts["metadata"] = self._llm._opts.metadata

            if self._llm._opts.store is not None:
                opts["store"] = self._llm._opts.store

            user = self._user or openai.NOT_GIVEN
            messages = _build_oai_context(self._chat_ctx, id(self))
            stream = await self._client.chat.completions.create(
                messages=messages,
                model=self._model,
                n=self._n,
                temperature=self._temperature,
                stream_options={"include_usage": True},
                stream=True,
                user=user,
                **opts,
            )

            async with stream:
                async for chunk in stream:
                    for choice in chunk.choices:
                        chat_chunk = self._parse_choice(chunk.id, choice)
                        if chat_chunk is not None:
                            retryable = False
                            self._event_ch.send_nowait(chat_chunk)

                    if chunk.usage is not None:
                        usage = chunk.usage
                        self._event_ch.send_nowait(
                            llm.ChatChunk(
                                request_id=chunk.id,
                                usage=llm.CompletionUsage(
                                    completion_tokens=usage.completion_tokens,
                                    prompt_tokens=usage.prompt_tokens,
                                    total_tokens=usage.total_tokens,
                                ),
                            )
                        )

        except openai.APITimeoutError:
            raise APITimeoutError(retryable=retryable)
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
            )
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _parse_choice(self, id: str, choice: Choice) -> llm.ChatChunk | None:
        delta = choice.delta

        # https://github.com/livekit/agents/issues/688
        # the delta can be None when using Azure OpenAI using content filtering
        if delta is None:
            return None

        if delta.tool_calls:
            # check if we have functions to calls
            for tool in delta.tool_calls:
                if not tool.function:
                    continue  # oai may add other tools in the future

                call_chunk = None
                if self._tool_call_id and tool.id and tool.index != self._tool_index:
                    call_chunk = self._try_build_function(id, choice)

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
            # we're done with the tool calls, run the last one
            return self._try_build_function(id, choice)

        return llm.ChatChunk(
            request_id=id,
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(content=delta.content, role="assistant"),
                    index=choice.index,
                )
            ],
        )

    def _try_build_function(self, id: str, choice: Choice) -> llm.ChatChunk | None:
        if not self._fnc_ctx:
            logger.warning("oai stream tried to run function without function context")
            return None

        if self._tool_call_id is None:
            logger.warning(
                "oai stream tried to run function but tool_call_id is not set"
            )
            return None

        if self._fnc_name is None or self._fnc_raw_arguments is None:
            logger.warning(
                "oai stream tried to call a function but raw_arguments and fnc_name are not set"
            )
            return None

        fnc_info = _create_ai_function_info(
            self._fnc_ctx, self._tool_call_id, self._fnc_name, self._fnc_raw_arguments
        )

        self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
        self._function_calls_info.append(fnc_info)

        return llm.ChatChunk(
            request_id=id,
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        tool_calls=[fnc_info],
                        content=choice.delta.content,
                    ),
                    index=choice.index,
                )
            ],
        )


def _build_oai_context(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> list[ChatCompletionMessageParam]:
    return [build_oai_message(msg, cache_key) for msg in chat_ctx.messages]  # type: ignore
