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
from langgraph.types import StreamMode
from langgraph.typing import ContextT

from livekit.agents import llm, utils
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

_SUPPORTED_MODES: set[StreamMode] = {"messages", "custom"}


class LLMAdapter(llm.LLM, Generic[ContextT]):
    def __init__(
        self,
        graph: PregelProtocol[Any, ContextT, Any, Any],
        *,
        config: RunnableConfig | None = None,
        context: ContextT | None = None,
        subgraphs: bool = False,
        stream_mode: StreamMode | list[StreamMode] = "messages",
    ) -> None:
        super().__init__()
        modes = {stream_mode} if isinstance(stream_mode, str) else set(stream_mode)
        unsupported = modes - _SUPPORTED_MODES
        if unsupported:
            raise ValueError(
                f"Unsupported stream mode(s): {unsupported}. Only {_SUPPORTED_MODES} are supported."
            )
        self._graph = graph
        self._config = config
        self._context = context
        self._subgraphs = subgraphs
        self._stream_mode = stream_mode

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
            stream_mode=self._stream_mode,
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
        stream_mode: StreamMode | list[StreamMode] = "messages",
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
        self._stream_mode = stream_mode

    async def _run(self) -> None:
        state = self._chat_ctx_to_state()
        is_multi_mode = isinstance(self._stream_mode, list)

        # Some LangGraph versions don't accept the `subgraphs` or `context` kwargs yet.
        # Try with them first; fall back gracefully if unsupported.
        try:
            aiter = self._graph.astream(
                state,
                self._config,
                context=self._context,
                stream_mode=self._stream_mode,
                subgraphs=self._subgraphs,
            )
        except TypeError:
            aiter = self._graph.astream(
                state,
                self._config,
                stream_mode=self._stream_mode,
            )

        async for item in aiter:
            # Multi-mode: item is (mode, data) tuple wrapper
            if is_multi_mode and isinstance(item, tuple) and len(item) == 2:
                mode, data = item
                if isinstance(mode, str):
                    if mode == "custom":
                        # data = payload (str, dict, object)
                        chat_chunk = _to_chat_chunk(data)
                        if chat_chunk:
                            self._event_ch.send_nowait(chat_chunk)
                        continue
                    elif mode == "messages":
                        # data = (token, metadata)
                        token_like = _extract_message_chunk(data)
                        if token_like is None:
                            continue
                        chat_chunk = _to_chat_chunk(token_like)
                        if chat_chunk:
                            self._event_ch.send_nowait(chat_chunk)
                        continue

            # Single-mode: item is data directly (no tuple wrapper)
            if self._stream_mode == "custom":
                # item = payload (str, dict, object)
                chat_chunk = _to_chat_chunk(item)
                if chat_chunk:
                    self._event_ch.send_nowait(chat_chunk)
            elif self._stream_mode == "messages":
                # item = (token, metadata)
                token_like = _extract_message_chunk(item)
                if token_like is None:
                    continue
                chat_chunk = _to_chat_chunk(token_like)
                if chat_chunk:
                    self._event_ch.send_nowait(chat_chunk)

    def _chat_ctx_to_state(self) -> dict[str, Any]:
        """Convert chat context to langgraph input"""

        messages: list[AIMessage | HumanMessage | SystemMessage] = []
        for msg in self._chat_ctx.messages():
            content = msg.text_content
            if content:
                if msg.role == "assistant":
                    messages.append(AIMessage(content=content, id=msg.id))
                elif msg.role == "user":
                    messages.append(HumanMessage(content=content, id=msg.id))
                elif msg.role in ["system", "developer"]:
                    messages.append(SystemMessage(content=content, id=msg.id))

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
        content = msg.text
        if getattr(msg, "id", None):
            message_id = msg.id  # type: ignore
    elif isinstance(msg, dict):
        raw = msg.get("content")
        if isinstance(raw, str):
            content = raw
    elif hasattr(msg, "content"):
        raw = msg.content
        if isinstance(raw, str):
            content = raw

    if not content:
        return None

    return llm.ChatChunk(
        id=message_id,
        delta=llm.ChoiceDelta(
            role="assistant",
            content=content,
        ),
    )
