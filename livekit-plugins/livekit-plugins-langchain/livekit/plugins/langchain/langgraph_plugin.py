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

"""LiveKit LangGraph Plugin.

This plugin provides seamless integration between LiveKit voice agents and
LangGraph workflows. It preserves backwards-compatibility with ``CompiledStateGraph``
(which streams ``BaseMessageChunk`` instances) **and** adds support for
``RemoteGraph`` which streams ``dict`` payloads.

A universal filtering mechanism ensures that only user-facing conversational
responses are spoken by the voice agent.  Tool calls, intermediate workflow
outputs, and other non-conversational chunks are filtered out.  The behaviour
can be toggled with :pyattr:`filter_messages` – enabled by default.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.pregel.protocol import PregelProtocol

from livekit.agents import llm, utils
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)

# ---------------------------------------------------------------------------
# High-level LLM adapter
# ---------------------------------------------------------------------------


class LLMAdapter(llm.LLM):
    """LLM wrapper that executes a LangGraph workflow.

    Parameters
    ----------
    graph
        A :class:`~langgraph.pregel.protocol.PregelProtocol` instance.  Can be a
        ``CompiledStateGraph`` or a ``RemoteGraph``.
    config
        Optional ``RunnableConfig`` forwarded to ``graph.astream``.
    langgraph_node
        If provided, only chunks originating from the specified node(s) will be
        spoken.  Accepts a single ``str`` or a ``list[str]``.
    filter_messages
        Enable/disable universal filtering.  Defaults to ``True``.
    """

    def __init__(
        self,
        graph: PregelProtocol,
        *,
        config: RunnableConfig | None = None,
        langgraph_node: str | list[str] | None = None,
        filter_messages: bool = True,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._config = config
        self._langgraph_node = langgraph_node
        self._filter_messages = filter_messages

    # ---------------------------------------------------------------------
    # Public LLM interface
    # ---------------------------------------------------------------------

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        # these are unused; tool execution happens inside LangGraph.
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "LangGraphStream":
        return LangGraphStream(
            llm_adapter=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            graph=self._graph,
            config=self._config,
            conn_options=conn_options,
            langgraph_node=self._langgraph_node,
            filter_messages=self._filter_messages,
        )


# ---------------------------------------------------------------------------
# Streaming wrapper
# ---------------------------------------------------------------------------


class LangGraphStream(llm.LLMStream):
    """Async stream that relays LangGraph outputs through LiveKit's LLMStream."""

    def __init__(
        self,
        *,
        llm_adapter: LLMAdapter,
        chat_ctx: ChatContext,
        tools: Sequence[FunctionTool | RawFunctionTool],
        graph: PregelProtocol,
        config: RunnableConfig | None,
        conn_options: APIConnectOptions,
        langgraph_node: str | list[str] | None,
        filter_messages: bool,
    ) -> None:
        super().__init__(
            llm=llm_adapter,
            chat_ctx=chat_ctx,
            tools=list(tools),
            conn_options=conn_options,
        )
        self._graph = graph
        self._config = config
        self._langgraph_node = langgraph_node
        self._filter_messages = filter_messages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat_ctx_to_state(self) -> dict[str, Any]:
        """Convert :class:`ChatContext` into LangGraph expected state."""

        messages: list[AIMessage | HumanMessage | SystemMessage] = []
        for item in self._chat_ctx.items:
            if isinstance(item, ChatMessage):
                content = item.text_content
                if not content:
                    continue
                if item.role == "assistant":
                    messages.append(AIMessage(content=content))
                elif item.role == "user":
                    messages.append(HumanMessage(content=content))
                elif item.role in {"system", "developer"}:
                    messages.append(SystemMessage(content=content))
        return {"messages": messages}

    # ------------------------------------------------------------------
    # Core streaming logic
    # ------------------------------------------------------------------

    async def _run(self) -> None:  # pragma: no cover – not used synchronously
        raise NotImplementedError(
            "LangGraphStream is designed for async iteration; use 'async for'."
        )

    async def __aiter__(self):  # noqa: D401 – simple iterator
        state = self._chat_ctx_to_state()

        async for chunk, run_id in self._graph.astream(
            state,
            self._config,
            stream_mode="messages",
        ):
            chat_chunk = _to_chat_chunk(
                chunk,
                langgraph_node=self._langgraph_node,
                run_id=run_id,
                filter_messages=self._filter_messages,
            )
            if chat_chunk is not None:
                yield chat_chunk


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _to_chat_chunk(
    msg: str | Any,
    *,
    langgraph_node: str | list[str] | None,
    run_id: dict | None,
    filter_messages: bool,
) -> llm.ChatChunk | None:
    """Convert LangGraph output into LiveKit :class:`ChatChunk`.

    The function supports three payload shapes:

    * ``str`` – emitted by some custom nodes
    * :class:`langchain_core.messages.BaseMessageChunk` – emitted by
      ``CompiledStateGraph``
    * ``dict`` – emitted by ``RemoteGraph``
    """

    # Handle dict payloads (from RemoteGraph)
    if isinstance(msg, dict):
        # If filtering is enabled, check if we should emit this chunk
        if filter_messages:
            if not _should_process_chunk(msg, langgraph_node, run_id):
                return None

        # Treat dicts similarly to BaseMessageChunk after optional filtering
        content = msg.get("content", "")
        if not content.strip():
            return None
        role = "assistant"
        if msg.get("type") == "HumanMessageChunk":
            role = "user"
        elif msg.get("type") == "SystemMessageChunk":
            role = "system"
        return llm.ChatChunk(
            id=msg.get("id", utils.shortuuid("LC_")),
            delta=llm.ChoiceDelta(content=content, role=role),
        )

    # For BaseMessageChunk / str we fall back to raw handling.
    return _raw_to_chat_chunk(msg)


def _raw_to_chat_chunk(msg: str | BaseMessageChunk | Any) -> llm.ChatChunk | None:
    """Convert raw message (str/BaseMessageChunk) to ChatChunk without filtering."""

    message_id = utils.shortuuid("LC_")
    content: str | None = None

    if isinstance(msg, str):
        content = msg
    elif isinstance(msg, BaseMessageChunk):
        content = msg.content if hasattr(msg, "content") else msg.text()
        if getattr(msg, "id", None):
            message_id = msg.id  # type: ignore[attr-defined]

    if content is None or not content.strip():
        return None

    return llm.ChatChunk(
        id=message_id,
        delta=llm.ChoiceDelta(content=content, role="assistant"),
    )


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def _should_process_chunk(
    chunk: dict,
    allowed_langgraph_nodes: str | list[str] | None,
    run_id: dict | None,
) -> bool:
    """Return *True* if chunk should be spoken; *False* otherwise."""

    # Node-based filtering – if enabled, drop chunks from other nodes.
    if allowed_langgraph_nodes is not None:
        allowed: List[str] = (
            [allowed_langgraph_nodes]
            if isinstance(allowed_langgraph_nodes, str)
            else list(allowed_langgraph_nodes)
        )
        node = None
        if run_id and isinstance(run_id, dict):
            node = run_id.get("langgraph_node")
        node = (
            node
            or chunk.get("response_metadata", {}).get("langgraph_node")
            or chunk.get("langgraph_node")
        )
        if node and node not in allowed:
            return False

    # Universal tool call filtering.
    if (
        chunk.get("tool_calls")
        or chunk.get("tool_call_chunks")
        or chunk.get("invalid_tool_calls")
    ):
        return False

    # Type-based heuristics.
    chunk_type = str(chunk.get("type", ""))
    if "tool" in chunk_type.lower():
        return False

    # Additional kwargs may contain tool metadata.
    additional = chunk.get("additional_kwargs", {})
    if any(
        key in additional for key in ("tool_calls", "function_call", "tool_call_id")
    ):
        return False

    return True
