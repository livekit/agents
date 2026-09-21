from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import (
    FunctionToolCall,
    LLMStream,
    Tool,
    ToolChoice,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice.events import ToolCallEnded, ToolExecutionUpdatedEvent

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class _DelayedEofProgressLLM(FakeLLM):
    def __init__(self, *, root_call: FunctionToolCall, dependent_call: FunctionToolCall) -> None:
        super().__init__()
        self.root_call = root_call
        self.dependent_call = dependent_call
        self.emit_dependent = asyncio.Event()
        self.close_stream = asyncio.Event()
        self.stream_eof = asyncio.Event()
        self.progress_seen_by_model = asyncio.Event()
        self._initial_stream = True

    def chat(
        self,
        *,
        chat_ctx: Any,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        if any(
            item.type == "function_call_output"
            and item.call_id == "room"
            and "room reservation is pending" in str(item.output)
            for item in chat_ctx.items
        ):
            self.progress_seen_by_model.set()

        if self._initial_stream:
            self._initial_stream = False
            return _DelayedEofProgressStream(
                self,
                chat_ctx=chat_ctx,
                tools=tools or [],
                conn_options=conn_options,
            )

        return FakeLLMStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _DelayedEofProgressStream(FakeLLMStream):
    _llm: _DelayedEofProgressLLM

    async def _run(self) -> None:
        self._send_chunk(tool_calls=[self._llm.root_call])
        await self._llm.emit_dependent.wait()
        self._send_chunk(tool_calls=[self._llm.dependent_call])
        await self._llm.close_stream.wait()
        self._llm.stream_eof.set()


async def _close(session: AgentSession) -> None:
    await asyncio.wait_for(session.aclose(), timeout=5)


@pytest.mark.asyncio
async def test_delayed_eof_exposes_root_progress_before_dependency_admission() -> None:
    root_started = asyncio.Event()
    root_progress_sent = asyncio.Event()
    release_root = asyncio.Event()
    dependent_started = asyncio.Event()

    @function_tool(name="save_room")
    async def save_room(ctx: RunContext) -> str:
        root_started.set()
        await ctx.update("room reservation is pending")
        root_progress_sent.set()
        await release_root.wait()
        return "room reservation completed"

    @function_tool(name="save_meal", after=("save_room",))
    async def save_meal(ctx: RunContext) -> str:
        dependent_started.set()
        return "meal saved"

    llm = _DelayedEofProgressLLM(
        root_call=FunctionToolCall(name="save_room", arguments="{}", call_id="room"),
        dependent_call=FunctionToolCall(name="save_meal", arguments="{}", call_id="meal"),
    )
    session = AgentSession(
        llm=llm,
        stt=None,
        vad=None,
        tts=None,
        turn_handling={"turn_detection": None},
        tool_handling={"on_dependency_error": "skip"},
    )
    await session.start(Agent(instructions="booking", tools=[save_room, save_meal]))
    try:
        session.generate_reply(user_input="batch")
        await asyncio.wait_for(root_started.wait(), timeout=5)
        await asyncio.wait_for(root_progress_sent.wait(), timeout=5)

        llm.emit_dependent.set()
        llm.close_stream.set()
        await asyncio.wait_for(llm.stream_eof.wait(), timeout=5)

        await asyncio.wait_for(
            llm.progress_seen_by_model.wait(),
            timeout=1,
        )

        assert llm.stream_eof.is_set()
        assert not release_root.is_set()
        assert not dependent_started.is_set()

        release_root.set()
        await asyncio.wait_for(dependent_started.wait(), timeout=5)
    finally:
        release_root.set()
        llm.emit_dependent.set()
        llm.close_stream.set()
        await asyncio.wait_for(llm.stream_eof.wait(), timeout=5)
        await _close(session)


@pytest.mark.asyncio
async def test_duplicate_dependent_id_keeps_refusal_and_final_correlated() -> None:
    root_started = asyncio.Event()
    release_root = asyncio.Event()
    duplicate_refused = asyncio.Event()
    dependent_started = asyncio.Event()
    final_committed = asyncio.Event()
    dependent_calls = 0

    @function_tool(name="root")
    async def root(ctx: RunContext) -> str:
        root_started.set()
        await release_root.wait()
        return "root done"

    @function_tool(name="dependent", after=("root",))
    async def dependent(ctx: RunContext) -> str:
        nonlocal dependent_calls
        dependent_calls += 1
        dependent_started.set()
        return "dependent final"

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="duplicate-dependent",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="dependent", arguments="{}", call_id="same"),
                    FunctionToolCall(name="dependent", arguments="{}", call_id="same"),
                    FunctionToolCall(name="root", arguments="{}", call_id="root"),
                ],
            )
        ]
    )
    session = AgentSession(
        llm=llm,
        stt=None,
        vad=None,
        tts=None,
        turn_handling={"turn_detection": None},
        tool_handling={"on_dependency_error": "run"},
    )
    await session.start(Agent(instructions="duplicate", tools=[root, dependent]))

    def observe(event: ToolExecutionUpdatedEvent) -> None:
        update = event.update
        if (
            isinstance(update, ToolCallEnded)
            and update.call_id == "same"
            and "duplicate function call id" in str(update.message)
        ):
            duplicate_refused.set()

    session.on("tool_execution_updated", observe)
    history_insert = session.history.insert

    def observe_history(items: Any) -> None:
        history_insert(items)

        def contains_final(item: Any) -> bool:
            if hasattr(item, "type"):
                return (
                    item.type == "function_call_output"
                    and item.name == "dependent"
                    and item.output == "dependent final"
                )
            if isinstance(item, (list, tuple)):
                return any(contains_final(child) for child in item)
            return False

        if contains_final(items):
            final_committed.set()

    session.history.insert = observe_history
    try:
        session.generate_reply(user_input="duplicate-dependent")
        await asyncio.wait_for(root_started.wait(), timeout=5)
        await asyncio.wait_for(duplicate_refused.wait(), timeout=5)

        release_root.set()
        await asyncio.wait_for(dependent_started.wait(), timeout=5)
        await asyncio.wait_for(final_committed.wait(), timeout=5)

        outputs = [
            item
            for item in session.current_agent.chat_ctx.items
            if item.type == "function_call_output" and item.name == "dependent"
        ]
        errors = [
            item
            for item in outputs
            if item.is_error and "duplicate function call id" in str(item.output)
        ]
        finals = [item for item in outputs if item.output == "dependent final"]
        assert dependent_calls == 1
        assert len(errors) == 1
        assert len(finals) == 1
        assert errors[0].call_id != finals[0].call_id
    finally:
        release_root.set()
        await _close(session)
