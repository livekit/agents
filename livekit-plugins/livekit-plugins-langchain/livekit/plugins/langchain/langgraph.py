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

from typing import Any

from langchain_core.messages import AIMessage, BaseMessageChunk, HumanMessage, SystemMessage
from langgraph.pregel import PregelProtocol, RunnableConfig

from livekit.agents import llm, utils
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)


class LLMAdapter(llm.LLM):
    def __init__(
        self,
        graph: PregelProtocol,
        *,
        config: RunnableConfig | None = None,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._config = config

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        # these are unused, since tool execution takes place in langgraph
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LangGraphStream:
        return LangGraphStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools,
            graph=self._graph,
            conn_options=conn_options,
            config=self._config,
        )


class LangGraphStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLMAdapter,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
        graph: PregelProtocol,
        config: RunnableConfig | None = None,
    ):
        super().__init__(
            llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._graph = graph
        self._config = config

    async def _run(self):
        state = self._chat_ctx_to_state()

        async for message_chunk, _ in self._graph.astream(
            state,
            self._config,
            stream_mode="messages",
        ):
            chat_chunk = _to_chat_chunk(message_chunk)
            if chat_chunk:
                self._event_ch.send_nowait(chat_chunk)

    def _chat_ctx_to_state(self) -> dict[str, Any]:
        """Convert chat context to langgraph input"""
        print(f"chat_ctx: {self._chat_ctx.items}")
        messages: list[AIMessage | HumanMessage | SystemMessage] = []
        for item in self._chat_ctx.items:
            # only support chat messages, ignoring tool calls
            if isinstance(item, ChatMessage):
                content = item.text_content
                if content:
                    if item.role == "assistant":
                        messages.append(AIMessage(content=content))
                    elif item.role == "user":
                        messages.append(HumanMessage(content=content))
                    elif item.role in ["system", "developer"]:
                        messages.append(SystemMessage(content=content))

        return {
            "messages": messages,
        }


def _to_chat_chunk(msg: BaseMessageChunk | None) -> llm.ChatChunk | None:
    message_id = utils.shortuuid("LC_")
    content: str | None = None

    if isinstance(msg, str):
        content = msg
    elif isinstance(msg, BaseMessageChunk):
        content = msg.text()
        if msg.id:
            message_id = msg.id

    if not content:
        return None

    return llm.ChatChunk(
        id=message_id,
        delta=llm.ChoiceDelta(
            role="assistant",
            content=content,
        ),
    )
