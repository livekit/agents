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
from dataclasses import dataclass
from typing import Any, Awaitable, MutableSet

import httpx
from livekit.agents import llm

import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice

from .log import logger
from .models import ChatModels
from .utils import AsyncAzureADTokenProvider, get_base_url, build_oai_message

DEFAULT_MODEL = "gpt-4o"


@dataclass
class LLMOptions:
    model: str | ChatModels


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = DEFAULT_MODEL,
        base_url: str | None = None,
        client: openai.AsyncClient | None = None,
    ) -> None:
        self._opts = LLMOptions(model=model)
        self._client = client or openai.AsyncClient(
            base_url=get_base_url(base_url),
            http_client=httpx.AsyncClient(
                timeout=5.0,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                    keepalive_expiry=120,
                ),
            ),
        )
        self._running_fncs: MutableSet[asyncio.Task[Any]] = set()

    @staticmethod
    def create_azure_client(
        *,
        model: str | ChatModels = DEFAULT_MODEL,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
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

        return LLM(model=model, client=azure_client)

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
    ) -> "LLMStream":
        opts: dict[str, Any] = dict()
        if fnc_ctx and len(fnc_ctx.ai_functions) > 0:
            fncs_desc = []
            for fnc in fnc_ctx.ai_functions.values():
                fncs_desc.append(llm._oai_api.build_oai_function_description(fnc))

            opts["tools"] = fncs_desc

            if fnc_ctx and parallel_tool_calls is not None:
                opts["parallel_tool_calls"] = parallel_tool_calls

        messages = _build_oai_context(chat_ctx, id(self))
        cmp = self._client.chat.completions.create(
            messages=messages,
            model=self._opts.model,
            n=n,
            temperature=temperature,
            stream=True,
            **opts,
        )

        return LLMStream(oai_stream=cmp, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        *,
        oai_stream: Awaitable[openai.AsyncStream[ChatCompletionChunk]],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._awaitable_oai_stream = oai_stream
        self._oai_stream: openai.AsyncStream[ChatCompletionChunk] | None = None

        # current function call that we're waiting for full completion (args are streamed)
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None

    async def aclose(self) -> None:
        if self._oai_stream:
            await self._oai_stream.close()

        return await super().aclose()

    async def __anext__(self):
        if not self._oai_stream:
            self._oai_stream = await self._awaitable_oai_stream

        async for chunk in self._oai_stream:
            for choice in chunk.choices:
                chat_chunk = self._parse_choice(choice)
                if chat_chunk is not None:
                    return chat_chunk

        raise StopAsyncIteration

    def _parse_choice(self, choice: Choice) -> llm.ChatChunk | None:
        delta = choice.delta

        if delta.tool_calls:
            # check if we have functions to calls
            for tool in delta.tool_calls:
                if not tool.function:
                    continue  # oai may add other tools in the future

                call_chunk = None
                if self._tool_call_id and tool.id and tool.id != self._tool_call_id:
                    call_chunk = self._try_run_function(choice)

                if tool.function.name:
                    self._tool_call_id = tool.id
                    self._fnc_name = tool.function.name
                    self._fnc_raw_arguments = tool.function.arguments or ""
                elif tool.function.arguments:
                    self._fnc_raw_arguments += tool.function.arguments  # type: ignore

                if call_chunk is not None:
                    return call_chunk

        if choice.finish_reason == "tool_calls":
            # we're done with the tool calls, run the last one
            return self._try_run_function(choice)

        return llm.ChatChunk(
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(content=delta.content, role="assistant"),
                    index=choice.index,
                )
            ]
        )

    def _try_run_function(self, choice: Choice) -> llm.ChatChunk | None:
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

        fnc_info = llm._oai_api.create_ai_function_info(
            self._fnc_ctx, self._tool_call_id, self._fnc_name, self._fnc_raw_arguments
        )
        self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
        self._function_calls_info.append(fnc_info)

        return llm.ChatChunk(
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(role="assistant", tool_calls=[fnc_info]),
                    index=choice.index,
                )
            ]
        )


def _build_oai_context(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> list[ChatCompletionMessageParam]:
    return [build_oai_message(msg, cache_key) for msg in chat_ctx.messages]  # type: ignore
