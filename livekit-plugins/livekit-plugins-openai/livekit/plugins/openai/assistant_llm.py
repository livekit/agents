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
import uuid
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
MESSAGE_ID_KEY = "__openai_message_id__"
MESSAGES_ADDED_KEY = "__openai_messages_added__"


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
            self,
            output_queue: asyncio.Queue[llm.ChatChunk | Exception | None],
            chat_ctx: llm.ChatContext,
        ):
            super().__init__()
            self._chat_ctx = chat_ctx
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

        # Running stream is used to ensure that we only have one stream running at a time
        self._running_stream: asyncio.Future[None] = asyncio.Future()
        self._running_stream.set_result(None)

    async def _create_stream(self) -> None:
        await self._running_stream
        self._running_stream = asyncio.Future[None]()
        try:
            load_options = await self._sync_openai_task
            self._thread_id = load_options.thread_id
            self._assistant_id = load_options.assistant_id

            # At the chat_ctx level, create a map of thread_id to message_ids
            # This is used to keep track of which messages have been added to the thread
            # and which we may need to delete from OpenAI
            if MESSAGES_ADDED_KEY not in self._chat_ctx._metadata:
                self._chat_ctx._metadata[MESSAGES_ADDED_KEY] = dict[str, set[str]()]()

            if self._thread_id not in self._chat_ctx._metadata[MESSAGES_ADDED_KEY]:
                self._chat_ctx._metadata[MESSAGES_ADDED_KEY][self._thread_id] = set()

            added_messages_set: set[str] = self._chat_ctx._metadata[MESSAGES_ADDED_KEY][
                self._thread_id
            ]
            # Note: this will add latency unfortunately. Usually it's just one message so we loop it but
            # it will create an extra round trip to OpenAI before being able to run inference.
            for msg in self._chat_ctx.messages:
                msg_id = msg._metadata.get(MESSAGE_ID_KEY)
                if msg_id and msg_id not in added_messages_set:
                    await self._client.beta.threads.messages.delete(
                        thread_id=load_options.thread_id,
                        message_id=msg_id,
                    )
                    added_messages_set.remove(msg_id)

            for msg in self._chat_ctx.messages:
                if msg.role != "user":
                    continue

                msg_id = str(uuid.uuid4())
                if MESSAGE_ID_KEY not in msg._metadata:
                    converted_msg = build_oai_message(msg, self._cache_key)
                    converted_msg["private_message_id"] = msg_id
                    new_msg = await self._client.beta.threads.messages.create(
                        thread_id=self._thread_id,
                        role="user",
                        content=converted_msg["content"],
                    )
                    msg._metadata[MESSAGE_ID_KEY] = new_msg.id
                    self._chat_ctx._metadata[MESSAGES_ADDED_KEY][self._thread_id].add(
                        new_msg.id
                    )

            eh = AssistantLLMStream.EventHandler(
                self._output_queue, chat_ctx=self._chat_ctx
            )
            async with self._client.beta.threads.runs.stream(
                thread_id=load_options.thread_id,
                assistant_id=load_options.assistant_id,
                event_handler=eh,
            ) as stream:
                await stream.until_done()

            await self._output_queue.put(None)
        except Exception as e:
            await self._output_queue.put(e)
        finally:
            self._running_stream.set_result(None)

    async def aclose(self) -> None:
        pass

    async def __anext__(self):
        item = await self._output_queue.get()
        if item is None:
            raise StopAsyncIteration

        if isinstance(item, Exception):
            raise item

        return item
