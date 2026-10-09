from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterable, Callable
from typing import Any

from livekit.agents import AgentSession
from livekit.agents.llm import (
    FunctionCall,
    FunctionToolCall,
    LLMStream,
    Tool,
    ToolContext,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice.events import ToolCallEnded, ToolExecutionUpdatedEvent
from livekit.agents.voice.generation import perform_tool_executions
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream


def response(input_text: str, *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(
        input=input_text,
        content="",
        ttft=0,
        duration=0,
        tool_calls=list(calls),
    )


class DelayedBatchFakeLLM(FakeLLM):
    def __init__(self, *, first_call: FunctionToolCall, second_call: FunctionToolCall) -> None:
        super().__init__(fake_responses=[])
        self.first_call = first_call
        self.second_call = second_call
        self.emit_second = asyncio.Event()
        self.close_stream = asyncio.Event()
        self.stream_eof = asyncio.Event()
        self.progress_seen_by_model = asyncio.Event()

    def chat(
        self,
        *,
        chat_ctx: Any,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        if any(
            item.type == "function_call_output"
            and item.call_id == self.first_call.call_id
            and "room reservation is pending" in str(item.output)
            for item in chat_ctx.items
        ):
            self.progress_seen_by_model.set()
        return _DelayedBatchFakeLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class _DelayedBatchFakeLLMStream(FakeLLMStream):
    _llm: DelayedBatchFakeLLM

    async def _run(self) -> None:
        if self._get_index_text() != "batch":
            return
        self._send_chunk(tool_calls=[self._llm.first_call])
        await self._llm.emit_second.wait()
        self._send_chunk(tool_calls=[self._llm.second_call])
        await self._llm.close_stream.wait()
        self._llm.stream_eof.set()


def collect_terminals(session: AgentSession) -> defaultdict[str, list[ToolCallEnded]]:
    terminals: defaultdict[str, list[ToolCallEnded]] = defaultdict(list)

    def on_update(event: ToolExecutionUpdatedEvent) -> None:
        if isinstance(event.update, ToolCallEnded):
            terminals[event.update.call_id].append(event.update)

    session.on("tool_execution_updated", on_update)
    return terminals


def dispatch_tool_stream(
    session: AgentSession,
    tools: list[Tool],
    function_stream: AsyncIterable[FunctionCall],
    *,
    tool_execution_started_cb: Callable[[FunctionCall], Any] | None = None,
    tool_execution_completed_cb: Callable[[Any], Any] | None = None,
) -> tuple[asyncio.Task[None], Any]:
    return perform_tool_executions(
        session=session,
        speech_handle=SpeechHandle.create(),
        tool_ctx=ToolContext(tools),
        tool_choice="auto",
        function_stream=function_stream,
        tool_execution_started_cb=tool_execution_started_cb or (lambda _: None),
        tool_execution_completed_cb=tool_execution_completed_cb or (lambda _: None),
    )


async def wait(event: asyncio.Event, *, timeout: float = 5) -> None:
    await asyncio.wait_for(event.wait(), timeout=timeout)


async def close(session: AgentSession) -> None:
    await asyncio.wait_for(session.aclose(), timeout=5)
