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
from typing import Any, MutableSet

from livekit.agents import llm

from openai import AssistantEventHandler, AsyncClient
from openai.types.beta.assistant import Assistant
from openai.types.beta.thread import Thread
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import ToolCall, ToolCallDelta

from .models import ChatModels
from .utils import build_oai_message

DEFAULT_MODEL = "gpt-4o"
MESSAGE_THREAD_KEY = "__openai_message_thread__"


@dataclass
class LLMOptions:
    model: str | ChatModels


class AssistantLLM(llm.LLM):
    def __init__(
        self,
        *,
        client: AsyncClient,
        assistant: Assistant,
        thread: Thread,
        instructions: str | None = None,
    ) -> None:
        self._client = client
        self._assistant = assistant
        self._thread = thread
        self._instructions = instructions
        self._running_fncs: MutableSet[asyncio.Task[Any]] = set()

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
            client=self._client,
            thread=self._thread,
            assistant=self._assistant,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            instructions=self._instructions,
        )


class AssistantLLMStream(llm.LLMStream):
    class EventHandler(AssistantEventHandler):
        def on_text_created(self, text: Text) -> None:
            print("NEIL text", text.value)

        def on_text_delta(self, delta: TextDelta, snapshot: Text):
            print("NEIL text delta", delta.value, snapshot.value)

        def on_tool_call_created(self, tool_call: ToolCall):
            print(f"\nassistant > {tool_call.type}\n", flush=True)

        def on_tool_call_delta(self, delta: ToolCallDelta, snapshot: ToolCall):
            if delta.type == "code_interpreter":
                pass

    def __init__(
        self,
        *,
        client: AsyncClient,
        thread: Thread,
        assistant: Assistant,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        instructions: str | None,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._instructions = instructions
        self._client = client
        self._thread = thread
        self._assistant = assistant

        # current function call that we're waiting for full completion (args are streamed)
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._create_stream_task = asyncio.create_task(self._create_stream())
        self._output_queue = asyncio.Queue[llm.ChatChunk | Exception | None]()

    async def _sync_thread(self) -> None:
        for msg in self._chat_ctx.messages:
            if MESSAGE_THREAD_KEY not in msg._metadata:
                msg._metadata[MESSAGE_THREAD_KEY] = set[str]()
            if self._thread.id not in msg._metadata[MESSAGE_THREAD_KEY]:
                converted_msg = build_oai_message(msg)
                await self._client.beta.threads.messages.create(
                    thread_id=self._thread.id,
                    role=msg.role,
                    content=converted_msg["content"],
                )
                msg._metadata[MESSAGE_THREAD_KEY].add(self._thread.id)

    async def _create_stream(self) -> None:
        try:
            await self._sync_thread()
            async with self._client.beta.threads.runs.stream(
                thread_id=self._thread.id,
                assistant_id=self._assistant.id,
                instructions=self._instructions,
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
