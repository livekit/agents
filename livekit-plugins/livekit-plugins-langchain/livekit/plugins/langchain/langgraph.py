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

from collections.abc import AsyncIterable
from typing import Any, Generic

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.pregel.protocol import PregelProtocol
from langgraph.types import StreamMode
from langgraph.typing import ContextT

from livekit.agents import llm, utils
from livekit.agents.llm import ChatChunk, ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    FlushSentinel,
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

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[ChatChunk]) -> None:
        async def _filtered(aiter: AsyncIterable) -> AsyncIterable[ChatChunk]:
            async for ev in aiter:
                if isinstance(ev, ChatChunk):
                    yield ev

        await super()._metrics_monitor_task(_filtered(event_aiter))

    async def _run(self) -> None:
        state = self._chat_ctx_to_state()

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

        multi_mode = isinstance(self._stream_mode, list)

        async for item in aiter:
            # Strip subgraph namespace prefix when present.
            # With subgraphs=True, items are prefixed with a namespace tuple:
            #   single-mode: (ns, data) -> data
            #   multi-mode:  (ns, mode, data) -> (mode, data)
            if self._subgraphs and isinstance(item, tuple) and isinstance(item[0], tuple):
                item = item[1:]
                if len(item) == 1:
                    item = item[0]

            # Extract mode tag in multi-mode; infer in single-mode.
            if multi_mode and isinstance(item, tuple) and len(item) == 2:
                mode, data = item
            else:
                mode = self._stream_mode if isinstance(self._stream_mode, str) else None
                data = item

            if mode == "messages":
                self._send_message(data)
            elif mode == "custom":
                self._send_custom(data)

    def _send_custom(self, data: Any) -> None:
        """Handle custom stream mode items from StreamWriter.

        Custom mode emits raw values written by StreamWriter nodes — strings,
        dicts, BaseMessages, or arbitrary objects (e.g. FlushSentinel).
        FlushSentinel is forwarded directly to trigger immediate TTS playback.
        We extract text content where possible; non-text values are silently
        skipped since ChatChunk only carries text.
        """
        if isinstance(data, FlushSentinel):
            self._event_ch.send_nowait(data)  # type: ignore[arg-type]
            return

        content = _extract_custom_content(data)
        if content:
            chunk = _to_chat_chunk(content)
            if chunk:
                self._event_ch.send_nowait(chunk)

    def _send_message(self, data: Any) -> None:
        """Handle messages stream mode items.

        Messages mode yields (message, metadata) tuples where the message is
        typically a BaseMessageChunk from LLM streaming.
        Also tolerates bare token-like values for robustness.
        """
        if isinstance(data, (BaseMessageChunk, str)):
            token = data
        elif isinstance(data, tuple) and len(data) == 2:
            token, _meta = data
        else:
            return
        chunk = _to_chat_chunk(token)
        if chunk:
            self._event_ch.send_nowait(chunk)

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


def _extract_custom_content(value: Any) -> str | None:
    """Extract text from a custom stream value.

    StreamWriter can emit arbitrary types. We recognize common text-carrying
    shapes (str, BaseMessage, dict with "content" key) and return the text.
    Returns None for non-text values (e.g. FlushSentinel, control objects)
    so the caller can skip them — ChatChunk only carries text content.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, BaseMessage) and isinstance(value.content, str):
        return value.content
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]  # type: ignore[no-any-return]
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

    if not content:
        return None

    return llm.ChatChunk(
        id=message_id,
        delta=llm.ChoiceDelta(
            role="assistant",
            content=content,
        ),
    )
