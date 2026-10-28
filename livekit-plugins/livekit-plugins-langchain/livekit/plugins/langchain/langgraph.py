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

from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import aclosing
from typing import Generic, Never, Protocol, TypeVar, cast, runtime_checkable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

# The SDK ships annotated namespace modules without a recognized typing marker.
from langgraph.pregel.protocol import PregelProtocol  # pyright: ignore[reportMissingTypeStubs]
from langgraph.types import StreamMode
from langgraph.typing import ContextT, InputT, OutputT, StateT

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
_ContextT_contra = TypeVar("_ContextT_contra", contravariant=True)


class _StreamingGraph(Protocol[_ContextT_contra]):
    def astream(
        self,
        input: dict[str, object],
        config: RunnableConfig | None = None,
        *,
        context: _ContextT_contra | None = None,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        subgraphs: bool = False,
    ) -> AsyncIterator[object]: ...


@runtime_checkable
class _Content(Protocol):
    @property
    def content(self) -> object: ...


class LLMAdapter(llm.LLM[Never], Generic[ContextT]):
    def __init__(
        self,
        graph: PregelProtocol[StateT, ContextT, InputT, OutputT],
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
        # LangGraph accepts partial state updates; its input type describes the full state.
        self._graph = cast(_StreamingGraph[ContextT], graph)
        self._config = config
        self._context = context
        self._subgraphs = subgraphs
        self._stream_mode: StreamMode | list[StreamMode] = stream_mode

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
        extra_kwargs: NotGivenOr[dict[str, object]] = NOT_GIVEN,
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
        graph: _StreamingGraph[ContextT],
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
        self._stream_mode: StreamMode | list[StreamMode] = stream_mode

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

        # LangGraph's astream is an async generator, annotated as the narrower AsyncIterator.
        async with aclosing(cast(AsyncGenerator[object, None], aiter)) as events:
            async for item in events:
                # Multi-mode: item is (mode, data) tuple wrapper
                if is_multi_mode:
                    if not isinstance(item, tuple):
                        continue
                    values = cast(tuple[object, ...], item)
                    if len(values) != 2:
                        continue
                    mode, data = values
                    if isinstance(mode, str):
                        if mode == "custom":
                            # data = payload (str, dict, object)
                            chat_chunk = _to_chat_chunk(data)
                            if chat_chunk:
                                self.send(chat_chunk)
                            continue
                        elif mode == "messages":
                            # data = (token, metadata)
                            token_like = _extract_message_chunk(data)
                            if token_like is None:
                                continue
                            chat_chunk = _to_chat_chunk(token_like)
                            if chat_chunk:
                                self.send(chat_chunk)
                            continue

                    continue

                # Single-mode: item is data directly (no tuple wrapper)
                if self._stream_mode == "custom":
                    # item = payload (str, dict, object)
                    chat_chunk = _to_chat_chunk(item)
                    if chat_chunk:
                        self.send(chat_chunk)
                elif self._stream_mode == "messages":
                    # item = (token, metadata)
                    token_like = _extract_message_chunk(item)
                    if token_like is None:
                        continue
                    chat_chunk = _to_chat_chunk(token_like)
                    if chat_chunk:
                        self.send(chat_chunk)

    def _chat_ctx_to_state(self) -> dict[str, object]:
        """Convert chat context to langgraph input"""

        messages: list[AIMessage | HumanMessage | SystemMessage] = []
        for msg in self._chat_ctx.messages():
            content = msg.raw_text_content
            if content:
                if msg.role == "assistant":
                    messages.append(AIMessage(content=content, id=msg.id))
                elif msg.role == "user":
                    messages.append(HumanMessage(content=content, id=msg.id))
                elif msg.role in ["system", "developer"]:
                    messages.append(SystemMessage(content=content, id=msg.id))

        return {"messages": messages}


def _extract_message_chunk(item: object) -> BaseMessage | str | None:
    """Extract a message from LangGraph's optional namespace and mode wrappers."""
    if isinstance(item, (BaseMessage, str)):
        return item
    if not isinstance(item, tuple):
        return None

    values = cast(tuple[object, ...], item)
    if len(values) not in (2, 3):
        return None
    if len(values) == 3 and not isinstance(values[-1], tuple):
        return None
    token: object = values[0]
    if isinstance(values[-1], tuple):
        inner = cast(tuple[object, ...], values[-1])
        if len(inner) != 2:
            return None
        token = inner[0]
    return token if isinstance(token, (BaseMessage, str)) else None


def _to_chat_chunk(msg: object) -> llm.ChatChunk | None:
    message_id = utils.shortuuid("LC_")
    content: str | None = None

    if isinstance(msg, str):
        content = msg
    elif isinstance(msg, BaseMessage):
        content = msg.text
        if msg.id:
            message_id = msg.id
    elif isinstance(msg, dict):
        raw = cast(Mapping[object, object], msg).get("content")
        if isinstance(raw, str):
            content = raw
    elif isinstance(msg, _Content):
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
