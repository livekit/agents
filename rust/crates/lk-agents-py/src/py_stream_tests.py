"""Host pump ownership and cleanup, against the freshly linked native package."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import pytest

from livekit.agents import _bindings
from livekit.agents.llm.mcp import _Streams
from livekit.agents.utils import aio

pytestmark = pytest.mark.unit


async def test_mcp_sdk_context_enters_and_exits_in_one_task_across_host_calls() -> None:
    tasks: list[asyncio.Task[object] | None] = []

    @asynccontextmanager
    async def context() -> AsyncIterator[tuple[object, object]]:
        async with anyio.create_task_group():
            tasks.append(asyncio.current_task())
            try:
                yield object(), object()
            finally:
                tasks.append(asyncio.current_task())

    streams = _Streams(context())
    async with asyncio.timeout(5):
        await asyncio.create_task(streams._core_start(None))
        await asyncio.create_task(streams._core_close(None))
    assert len(tasks) == 2 and tasks[0] is tasks[1]


async def test_mcp_cancelled_start_settles_sdk_entry() -> None:
    entering, settled = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def context() -> AsyncIterator[tuple[object, object]]:
        try:
            entering.set()
            await asyncio.Event().wait()
            yield object(), object()
        finally:
            settled.set()

    streams = _Streams(context())
    async with asyncio.timeout(5):
        opening = asyncio.create_task(streams._core_start(None))
        await entering.wait()
        opening.cancel()
        with pytest.raises(asyncio.CancelledError):
            await opening
        assert settled.is_set()
        await streams._core_close(None)


async def test_mcp_close_before_start_never_enters_sdk() -> None:
    @asynccontextmanager
    async def context() -> AsyncIterator[tuple[object, object]]:
        pytest.fail("closed transport entered SDK")
        yield object(), object()

    streams = _Streams(context())
    await streams._core_close(None)
    with pytest.raises(RuntimeError, match="closed"):
        await streams._core_start(None)


async def test_mcp_cancelled_close_waiter_can_rejoin_sdk_exit() -> None:
    exiting, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def context() -> AsyncIterator[tuple[object, object]]:
        async with anyio.create_task_group():
            try:
                yield object(), object()
            finally:
                exiting.set()
                await release.wait()
                settled.set()

    streams = _Streams(context())
    async with asyncio.timeout(5):
        await streams._core_start(None)
        closing = asyncio.create_task(streams._core_close(None))
        try:
            await exiting.wait()
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert not settled.is_set()
        finally:
            release.set()
            await streams._core_close(None)
        assert settled.is_set()


class _Native:
    def __init__(self) -> None:
        self.output: asyncio.Queue[int | None] = asyncio.Queue()
        self.closing = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.allow_close.set()
        self.closed = asyncio.Event()
        self.close_count = 0

    def push(self, item: int) -> None:
        self.output.put_nowait(item)

    def end_input(self) -> None:
        self.output.put_nowait(None)

    async def recv(self) -> int | None:
        return await self.output.get()

    async def aclose(self) -> None:
        self.close_count += 1
        self.closing.set()
        await self.allow_close.wait()
        self.closed.set()


async def test_pump_stops_input_when_native_output_ends_first() -> None:
    native = _Native()
    entered, settled = asyncio.Event(), asyncio.Event()

    async def items() -> AsyncIterator[int]:
        try:
            entered.set()
            native.output.put_nowait(None)
            await asyncio.Event().wait()
            yield 1
        finally:
            settled.set()

    async with asyncio.timeout(5):
        await _bindings.pump(native, items(), native.push, lambda _: None)
    assert entered.is_set() and settled.is_set()
    assert native.close_count == 1 and native.closed.is_set()


async def test_pump_drains_final_output_after_input_finishes() -> None:
    native, output = _Native(), []

    async def items() -> AsyncIterator[int]:
        yield 1
        yield 2

    async with asyncio.timeout(5):
        await _bindings.pump(native, items(), native.push, output.append)
    assert output == [1, 2]
    assert native.close_count == 1 and native.closed.is_set()


async def test_pump_preserves_input_failure_and_closes_waiting_native_reader() -> None:
    native = _Native()
    failure = ValueError("input failed")

    async def items() -> AsyncIterator[int]:
        yield 1
        raise failure

    async with asyncio.timeout(5):
        with pytest.raises(ValueError) as raised:
            await _bindings.pump(native, items(), native.push, lambda _: None)
    assert raised.value is failure
    assert native.close_count == 1 and native.closed.is_set()


async def test_pump_closes_native_even_if_input_cleanup_is_still_waiting() -> None:
    native = _Native()
    entered, cleaning, release, settled = (asyncio.Event() for _ in range(4))

    async def items() -> AsyncIterator[int]:
        try:
            entered.set()
            await asyncio.Event().wait()
            yield 1
        finally:
            cleaning.set()
            await release.wait()
            settled.set()

    async with asyncio.timeout(5):
        pump = asyncio.create_task(_bindings.pump(native, items(), native.push, lambda _: None))
        try:
            await entered.wait()
            pump.cancel()
            await cleaning.wait()
            await native.closing.wait()
            assert not settled.is_set()
        finally:
            release.set()
            await asyncio.gather(pump, return_exceptions=True)
    assert settled.is_set() and native.closed.is_set() and native.close_count == 1


async def test_cancelled_input_is_terminal_even_when_native_output_is_waiting() -> None:
    native = _Native()

    async def items() -> AsyncIterator[int]:
        raise asyncio.CancelledError
        yield 1

    async with asyncio.timeout(5):
        with pytest.raises(asyncio.CancelledError):
            await _bindings.pump(native, items(), native.push, lambda _: None)
    assert native.close_count == 1 and native.closed.is_set()


async def test_rejoining_bounded_cleanup_does_not_cancel_an_authored_finally_again() -> None:
    baseline = aio.pending_cleanup_count()
    entered, cleaning, release, settled = (asyncio.Event() for _ in range(4))

    async def run() -> None:
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            settled.set()

    async with asyncio.timeout(5):
        task = asyncio.create_task(run())
        try:
            await entered.wait()
            assert not await aio.cancel_and_wait(task, timeout=0)
            await cleaning.wait()
            assert aio.pending_cleanup_count() == baseline + 1
            assert not await aio.cancel_and_wait(task, timeout=0)
            assert not settled.is_set()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        assert settled.is_set()
        while aio.pending_cleanup_count() != baseline:
            await asyncio.sleep(0)


async def test_cancelled_pump_waiter_retains_and_settles_both_cleanup_branches() -> None:
    native = _Native()
    native.allow_close.clear()
    baseline = aio.pending_cleanup_count()
    entered, cleaning, release, settled = (asyncio.Event() for _ in range(4))

    async def items() -> AsyncIterator[int]:
        try:
            entered.set()
            await asyncio.Event().wait()
            yield 1
        finally:
            cleaning.set()
            await release.wait()
            settled.set()

    async with asyncio.timeout(5):
        pump = asyncio.create_task(_bindings.pump(native, items(), native.push, lambda _: None))
        try:
            await entered.wait()
            pump.cancel()
            await cleaning.wait()
            await native.closing.wait()
            pump.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pump
            assert not settled.is_set() and not native.closed.is_set()
            assert aio.pending_cleanup_count() > baseline
        finally:
            release.set()
            native.allow_close.set()
            await asyncio.gather(pump, return_exceptions=True)
        await settled.wait()
        await native.closed.wait()
        while aio.pending_cleanup_count() != baseline:
            await asyncio.sleep(0)
    assert native.close_count == 1


@pytest.mark.parametrize("text_session", [False, True])
async def test_failed_stream_keeps_correlated_input_metadata_on_the_original_failure(
    text_session: bool,
) -> None:
    from livekit.agents._generated import agent_run as records, text_session as sessions
    from livekit.agents._generated.storage_turn import PendingCall
    from livekit.agents.voice.run_stream import RunStream, _Started

    requirement = records.RunInputRequirement(
        invocation_id="run-1",
        call=PendingCall(name="ask", call_id="call-1", answer_tool="answer_city"),
        question="Which city?",
        verbatim=True,
    )
    failure = records.RunErrorToolRetriesExceeded(tool="retry", retries=0)
    outcome = (
        sessions.TextSessionUpdateFailed(
            error=sessions.TextSessionErrorRun(error=failure), input_requirements=[requirement]
        )
        if text_session
        else records.RunUpdateFailed(error=failure, input_requirements=[requirement])
    )

    class NativeResult:
        done = True
        cancelled = False

        def begin_result(self) -> int:
            return 0

        async def result(self, consumer: int) -> object:
            return outcome

        def end_result(self, consumer: int, terminal: bool) -> None:
            assert terminal

    async def start(native: object) -> _Started[object]:
        raise AssertionError("the result owner is already started")

    error = RuntimeError("tool retry budget exceeded")
    stream = RunStream(start)
    stream._core = NativeResult()
    stream._started = _Started(
        settle=lambda _: pytest.fail("failure became success"), fail=lambda _: error
    )
    with pytest.raises(RuntimeError) as raised:
        await stream.result()
    assert raised.value is error
    assert raised.value.input_requirements == [requirement]
    with pytest.raises(RuntimeError) as repeated:
        await stream.result()
    assert repeated.value is error


async def test_voice_context_updates_use_the_captured_tool_invocation() -> None:
    from types import SimpleNamespace

    from livekit.agents._generated.agent_run import Invocation
    from livekit.agents._generated.tool_report import ToolOutcomeUpdate, ToolReport
    from livekit.agents.llm import FunctionCall
    from livekit.agents.voice.events import RunContext

    delivered: list[tuple[str, int, str, ToolReport]] = []

    class Port:
        def tool_update(
            self, invocation_id: str, operation: int, call_id: str, report: ToolReport
        ) -> None:
            delivered.append((invocation_id, operation, call_id, report))

    call = FunctionCall(call_id="reused", arguments="{}", name="lookup")
    ctx = RunContext(function_call=call, session=SimpleNamespace(_core=Port()))
    ctx._invocation = Invocation(
        invocation_id="captured-run",
        operation=17,
        agent_name="agent",
        execution_id=None,
        frame_id=None,
        context_scope=None,
        application_context=None,
        history=[],
        function_call=call,
        approval=None,
    )
    from livekit.agents._generated.voice_session import VoiceCommandOriginTool

    assert ctx._origin == VoiceCommandOriginTool(
        invocation_id="captured-run", operation=17, call_id="reused"
    )
    await ctx.update("still working")
    assert len(delivered) == 1
    invocation_id, operation, call_id, report = delivered[0]
    assert (invocation_id, operation, call_id) == ("captured-run", 17, "reused")
    assert isinstance(report.outcome, ToolOutcomeUpdate)
    assert report.outcome.raw == "still working"
