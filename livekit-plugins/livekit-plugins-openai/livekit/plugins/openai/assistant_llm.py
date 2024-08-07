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
from dataclasses import dataclass, field
from typing import Any, MutableSet

import httpx
from livekit.agents import llm

from openai import AsyncAssistantEventHandler, AsyncClient
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import ToolCall, ToolCallDelta

from .models import AssistantTools, ChatModels
from .utils import build_oai_message

DEFAULT_MODEL = "gpt-4o"
MESSAGE_THREAD_KEY = "__openai_message_thread__"


@dataclass
class LLMOptions:
    model: str | ChatModels


@dataclass
class AssistantCreateOptions:
    name: str
    instructions: str
    model: ChatModels
    tools: list[AssistantTools] = field(default_factory=list)


@dataclass
class AssistantLoadOptions:
    assistant_id: str
    thread_id: str


class AssistantLLM(llm.LLM):
    def __init__(
        self,
        *,
        assistant_opts: AssistantCreateOptions | AssistantLoadOptions,
        client: AsyncClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._client = client or AsyncClient(
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
        self._assistant_opts = assistant_opts
        self._running_fncs: MutableSet[asyncio.Task[Any]] = set()

        self._sync_openai_task = asyncio.create_task(self._sync_openai())

    async def _sync_openai(self) -> AssistantLoadOptions:
        if isinstance(self._assistant_opts, AssistantCreateOptions):
            assistant = await self._client.beta.assistants.create(
                model=self._assistant_opts.model,
                name=self._assistant_opts.name,
                instructions=self._assistant_opts.instructions,
                tools=[{"type": t} for t in self._assistant_opts.tools],
            )
            thread = await self._client.beta.threads.create()
            return AssistantLoadOptions(assistant_id=assistant.id, thread_id=thread.id)
        else:
            return self._assistant_opts

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
    ):
        return AssistantLLMStream(
            cache_key=id(self),
            sync_openai_task=self._sync_openai_task,
            client=self._client,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
        )


class AssistantLLMStream(llm.LLMStream):
    class EventHandler(AsyncAssistantEventHandler):
        def __init__(
            self, output_queue: asyncio.Queue[llm.ChatChunk | Exception | None]
        ):
            super().__init__()
            self._output_queue = output_queue

        async def on_text_delta(self, delta: TextDelta, snapshot: Text):
            self._output_queue.put_nowait(
                llm.ChatChunk(
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(role="assistant", content=delta.value)
                        )
                    ]
                )
            )
            print("NEIL text delta", delta.value, snapshot.value)

        async def on_tool_call_created(self, tool_call: ToolCall):
            print(f"\nassistant > {tool_call.type}\n", flush=True)

        async def on_tool_call_delta(self, delta: ToolCallDelta, snapshot: ToolCall):
            if delta.type == "code_interpreter":
                pass

    def __init__(
        self,
        *,
        cache_key: int,
        client: AsyncClient,
        sync_openai_task: asyncio.Task[AssistantLoadOptions],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._cache_key = cache_key
        self._client = client
        self._thread_id: str | None = None
        self._assistant_id: str | None = None

        # current function call that we're waiting for full completion (args are streamed)
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._output_queue = asyncio.Queue[llm.ChatChunk | Exception | None]()
        self._create_stream_task = asyncio.create_task(self._create_stream())
        self._sync_openai_task = sync_openai_task

    async def _sync_thread(self) -> None:
        load_options = await self._sync_openai_task
        for msg in self._chat_ctx.messages:
            if msg.role == "assistant":
                continue
            if MESSAGE_THREAD_KEY not in msg._metadata:
                msg._metadata[MESSAGE_THREAD_KEY] = set[str]()
            if self._thread_id not in msg._metadata[MESSAGE_THREAD_KEY]:
                converted_msg = build_oai_message(msg, self._cache_key)
                await self._client.beta.threads.messages.create(
                    thread_id=load_options.thread_id,
                    role=msg.role,
                    content=converted_msg["content"],
                )
                msg._metadata[MESSAGE_THREAD_KEY].add(self._thread_id)

    async def _create_stream(self) -> None:
        try:
            await self._sync_thread()
            load_options = await self._sync_openai_task

            async with self._client.beta.threads.runs.stream(
                thread_id=load_options.thread_id,
                assistant_id=load_options.assistant_id,
                event_handler=AssistantLLMStream.EventHandler(self._output_queue),
            ) as stream:
                await stream.until_done()
        except Exception as e:
            await self._output_queue.put(e)

    async def aclose(self) -> None:
        pass

    async def __anext__(self):
        item = await self._output_queue.get()
        if item is None:
            raise StopAsyncIteration

        if isinstance(item, Exception):
            raise item

        return item
