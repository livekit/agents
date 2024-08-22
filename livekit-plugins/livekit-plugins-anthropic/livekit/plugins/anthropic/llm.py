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
from typing import Any, Awaitable, MutableSet, List, Dict, Tuple

import httpx
from livekit.agents import llm

import anthropic

from .log import logger
from .models import (
    ChatModels,
)


@dataclass
class LLMOptions:
    model: str | ChatModels
    user: str | None


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        user: str | None = None,
        client: anthropic.AsyncClient | None = None,
    ) -> None:
        self._opts = LLMOptions(model=model, user=user)
        self._client = client or anthropic.AsyncClient(
            api_key=api_key,
            base_url=base_url,
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
            fncs_desc: list[dict[str, Any]] = []
            for fnc in fnc_ctx.ai_functions.values():
                fncs_desc.append(llm._oai_api.build_oai_function_description(fnc))

            opts["tools"] = fncs_desc

            if fnc_ctx and parallel_tool_calls is not None:
                opts["parallel_tool_calls"] = parallel_tool_calls

        messages = _build_oai_context(chat_ctx, id(self))
        stream = await self._client.messages.create(
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
        anthropic_stream: Awaitable[anthropic.AsyncStream[anthropic.types.message.]],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._awaitable_anthropic_stream = anthropic_stream
        self._anthropic_stream: anthropic.AsyncStream[ChatCompletionChunk] | None = None

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


def _collapse_message(chat_ctx: llm.ChatContext, cache_key: Any) -> Tuple[llm.ChatMessage, List[llm.ChatMessage]]:
    """
    Returns:
        Tuple[llm.ChatMessage, List[[llm.ChatMessage]]]: returns the latest system message and a list of combined messages (because anthropic enforces alternating roles) 
    """
    # Anthropic enforces alternating messages
    combined_messages: list[llm.ChatMessage] = []
    latest_system_message: llm.ChatMessage = llm.ChatMessage("system", content="")
    for m in chat_ctx.messages:
        if m.role == "system":
            latest_system_message = m
            continue

        if len(combined_messages) == 0 or  m.role != combined_messages[-1].role:
            combined_messages.append(llm.ChatMessage(m.role, content=""))
            continue

        last_message = combined_messages[-1]
        if isinstance(last_message.content, str):
            if isinstance(m.content, str):
                last_message.content += " " + m.content
            elif isinstance(m.content, list):
                new_text = " ".join([c for c in m.content if isinstance(c, str)])
                content: List[str | llm.ChatImage] = [last_message.content + " " + new_text]
                content.extend([c for c in m.content if not isinstance(c, str)])
                last_message.content = content
        elif isinstance(last_message.content, list):
            if isinstance(m.content, str):
                old_text = " ".join([c for c in last_message.content if isinstance(c, str)])
                content: List[str | llm.ChatImage] = [old_text + " " + m.content]
                content.extend([c for c in last_message.content if not isinstance(c, str)])
                last_message.content = content
            elif isinstance(m.content, list):
                new_text = " ".join([c for c in m.content if isinstance(c, str)])
                old_text = " ".join([c for c in last_message.content if isinstance(c, str)])
                content: List[str | llm.ChatImage] = [old_text + " " + new_text]
                content.extend([c for c in last_message.content if not isinstance(c, str)])
                content.extend([c for c in m.content if not isinstance(c, str)])
                last_message.content = content

    return latest_system_message, combined_messages

        
