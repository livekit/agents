from __future__ import annotations

import asyncio
import contextlib
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
from .fake_realtime import generation


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


def await_chain(task: asyncio.Task[Any]) -> str:
    chain: list[str] = []
    awaitable: Any = task.get_coro()
    seen: set[int] = set()
    while awaitable is not None and id(awaitable) not in seen:
        seen.add(id(awaitable))
        name = getattr(awaitable, "__qualname__", type(awaitable).__name__)
        frame = getattr(awaitable, "cr_frame", None)
        if frame is not None:
            name = f"{name} ({frame.f_code.co_filename}:{frame.f_lineno})"
        chain.append(name)
        awaitable = getattr(awaitable, "cr_await", None) or getattr(awaitable, "gi_yieldfrom", None)
    return " -> ".join(chain)


async def close(session: AgentSession) -> None:
    try:
        await asyncio.wait_for(session.aclose(), timeout=5)
    except asyncio.TimeoutError:
        print("live tasks at natural session close timeout:")
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task() and not task.done():
                print(f"  {task.get_name()}: {await_chain(task)}")
        activity = session._activity
        if activity is not None:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(activity._tool_executor.cancel_all(), timeout=2)
            speech = activity.current_speech
            if speech is not None:
                for task in speech._tasks:
                    task.cancel()
                speech._mark_done()
        dangling = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and (
                task.get_name().startswith("tool_dependency_")
                or task.get_name() in {"execute_tools_task", "tool_dependency_ready"}
            )
        ]
        for task in dangling:
            task.cancel()
        if dangling:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(asyncio.gather(*dangling, return_exceptions=True), timeout=2)
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(session.aclose(), timeout=2)
        raise


async def resolve_fake_realtime_replies(
    realtime_session: Any,
    stop: asyncio.Event,
) -> None:
    index = 1
    while not stop.is_set():
        while index < len(realtime_session._reply_futs):
            future = realtime_session._reply_futs[index]
            if not future.done():
                future.set_result(
                    generation(
                        response_id=f"followup-{index}",
                        text="progress acknowledged",
                        audio_duration=0.01,
                    )
                )
            index += 1

        if stop.is_set():
            return
        realtime_session.reply_created.clear()
        stop_wait = asyncio.create_task(stop.wait())
        reply_wait = asyncio.create_task(realtime_session.reply_created.wait())
        done, pending = await asyncio.wait(
            {stop_wait, reply_wait}, timeout=5, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if not done:
            raise asyncio.TimeoutError("fake realtime reply future was not created")
        if stop_wait in done:
            return


def item_snapshot(chat_ctx: Any) -> list[tuple[str, str, str | None, str | None]]:
    return [
        (item.id, item.type, getattr(item, "call_id", None), getattr(item, "output", None))
        for item in chat_ctx.items
    ]
