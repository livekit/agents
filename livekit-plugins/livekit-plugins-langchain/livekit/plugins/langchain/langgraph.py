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

from typing import Any, Generic

from langchain_core.messages import AIMessage, BaseMessageChunk, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.pregel.protocol import PregelProtocol
from langgraph.typing import ContextT

from livekit.agents import llm, utils
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)


class LLMAdapter(llm.LLM, Generic[ContextT]):
    def __init__(
        self,
        graph: PregelProtocol[Any, ContextT, Any, Any],
        *,
        config: RunnableConfig | None = None,
        context: ContextT | None = None,
        subgraphs: bool = False,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._config = config
        self._context = context
        self._subgraphs = subgraphs

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "LangChain"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        # these are unused, since tool execution takes place in langgraph
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LangGraphStream[ContextT]:
        return LangGraphStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            graph=self._graph,
            conn_options=conn_options,
            config=self._config,
            context=self._context,
            subgraphs=self._subgraphs,
        )


class LangGraphStream(llm.LLMStream, Generic[ContextT]):
    def __init__(
        self,
        llm: LLMAdapter[ContextT],
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        graph: PregelProtocol[Any, ContextT, Any, Any],
        config: RunnableConfig | None = None,
        context: ContextT | None = None,
        subgraphs: bool = False,
    ):
        super().__init__(
            llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._graph = graph
        self._config = config
        self._context = context
        self._subgraphs = subgraphs

    async def _run(self) -> None:
        state = self._chat_ctx_to_state()

        # Some LangGraph versions don't accept the `subgraphs` or `context` kwargs yet.
        # Try with them first; fall back gracefully if unsupported.
        try:
            aiter = self._graph.astream(
                state,
                self._config,
                context=self._context,
                stream_mode="messages",
                subgraphs=self._subgraphs,
            )
        except TypeError:
            aiter = self._graph.astream(
                state,
                self._config,
                stream_mode="messages",
            )

        async for item in aiter:
            token_like = _extract_message_chunk(item)
            if token_like is None:
                continue

            chat_chunk = _to_chat_chunk(token_like)
            if chat_chunk:
                self._event_ch.send_nowait(chat_chunk)

    def _chat_ctx_to_state(self) -> dict[str, Any]:
        """Convert chat context to langgraph input"""

        messages: list[AIMessage | HumanMessage | SystemMessage] = []
        for item in self._chat_ctx.items:
            # only support chat messages, ignoring tool calls
            if isinstance(item, ChatMessage):
                content = item.text_content
                if content:
                    if item.role == "assistant":
                        messages.append(AIMessage(content=content, id=item.id))
                    elif item.role == "user":
                        messages.append(HumanMessage(content=content, id=item.id))
                    elif item.role in ["system", "developer"]:
                        messages.append(SystemMessage(content=content, id=item.id))

        return {"messages": messages}


def _extract_message_chunk(item: Any) -> BaseMessageChunk | str | None:
    """
    Normalize outputs from graph.astream(..., stream_mode='messages', [subgraphs]).

    Expected shapes:
      - (token, meta)
      - (namespace, (token, meta))                  # with subgraphs=True
      - (mode, (token, meta))                       # future-friendly
      - (namespace, mode, (token, meta))            # future-friendly
    Also tolerate direct token-like values for robustness.
    """
    # Already a token-like thing?
    if isinstance(item, (BaseMessageChunk, str)):
        return item

    if not isinstance(item, tuple):
        return None

    # token is usually BaseMessageChunk, but could be a str
    # (token, meta)
    if len(item) == 2 and not isinstance(item[1], tuple):
        token, _meta = item
        return token  # type: ignore

    # (namespace, (token, meta))  OR  (mode, (token, meta))
    if len(item) == 2 and isinstance(item[1], tuple):
        inner = item[1]
        if len(inner) == 2:
            token, _meta = inner
            return token  # type: ignore

    # (namespace, mode, (token, meta))
    if len(item) == 3 and isinstance(item[2], tuple):
        inner = item[2]
        if len(inner) == 2:
            token, _meta = inner
            return token  # type: ignore

    return None


def _to_chat_chunk(msg: str | Any) -> llm.ChatChunk | None:
    message_id = utils.shortuuid("LC_")
    content: str | None = None

    if isinstance(msg, str):
        content = msg
    elif isinstance(msg, BaseMessageChunk):
        content = msg.text()
        if getattr(msg, "id", None):
            message_id = msg.id  # type: ignore

    if not content:
        return None

    return llm.ChatChunk(
        id=message_id,
        delta=llm.ChoiceDelta(
            role="assistant",
            content=content,
        ),
    )
