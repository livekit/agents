"""``RunStream``: an invocation's incremental view (agents.md "Streaming runs").

The stream pulls the core's updates on demand: the runner's bounded queue is the backpressure,
so nothing is dropped and nothing is buffered here. One consumer at a time (``stream_text``,
``stream_events``, or ``result()`` draining what nobody reads); exhausting an iterator means
the invocation settled.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

from pydantic import JsonValue

from .._generated import agent_run as records, text_session as session_records
from ..log import logger

if TYPE_CHECKING:
    from .. import _core

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, kw_only=True)
class _Event:
    sequence: int
    """Increasing occurrence number within this stream; not a durable reconnect cursor."""
    agent_name: str
    invocation_id: str
    frame_id: str | None = None
    """The backend session frame, when there is one; ``None`` on a direct run."""


@dataclass(frozen=True, kw_only=True)
class TextDeltaEvent(_Event):
    """User-visible assistant text added to ``message_id``."""

    message_id: str
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True, kw_only=True)
class TextEndEvent(_Event):
    """The assistant message ended; the run may continue."""

    message_id: str
    interrupted: bool
    type: Literal["text_end"] = "text_end"


@dataclass(frozen=True, kw_only=True)
class ToolCallEvent(_Event):
    """A selected call with its validated, complete arguments."""

    call_id: str
    tool_name: str
    arguments: dict[str, JsonValue]
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, kw_only=True)
class ToolResultEvent(_Event):
    """The recorded outcome of a call; a failure has an ``error`` and no output."""

    call_id: str
    tool_name: str
    outcome: Literal["completed", "failed", "cancelled"]
    output: JsonValue
    error: str | None
    type: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True, kw_only=True)
class ProgressEvent(_Event):
    """An application progress update, such as ``ctx.update(...)``."""

    message: str
    call_id: str | None
    tool_name: str | None
    type: Literal["progress"] = "progress"


StreamEvent = TextDeltaEvent | TextEndEvent | ToolCallEvent | ToolResultEvent | ProgressEvent


class RunStream(Generic[ResultT]):
    """The handle ``async with agent.run_stream(...)`` returns; applications do not build it."""

    def __init__(
        self,
        start: Callable[[], Awaitable[_Started[ResultT]]],
        *,
        agent_name: str,
    ) -> None:
        self._start = start
        self._agent_name = agent_name
        self._entered = False
        self._started: _Started[ResultT] | None = None
        self._sequence = 0
        self._viewed = False  # an iterator was started: the stream has its one view
        self._iterating = False
        self._settled: ResultT | None = None
        self._error: BaseException | None = None
        self._closing: asyncio.Task[None] | None = None
        self._presented: dict[str, str] = {}

    async def __aenter__(self) -> RunStream[ResultT]:
        if self._entered:
            raise RuntimeError("a RunStream is entered once and cannot be reused")
        self._entered = True
        self._started = await self._start()
        if self._started.updates is None:
            self._settled = self._started.settle(None)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._done():
            # leaving early cancels; the owned cleanup is awaited, the body's exception stays
            self.cancel()
            await self._close()

    def stream_text(self, *, delta: bool = False) -> AsyncIterator[str]:
        """User-visible assistant text: each fragment with ``delta=True``, else the accumulated
        text of the current message."""
        self._claim()
        return self._text(delta)

    def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Every ``StreamEvent`` of this run, in order."""
        self._claim()
        return self._events()

    async def result(self) -> ResultT:
        """The settled result; drains the notifications nobody reads. Raises what the run
        raised, ``asyncio.CancelledError`` once a cancellation settled."""
        if self._iterating:
            raise RuntimeError("result() while an iterator consumes the stream")
        if not self._entered:
            raise RuntimeError("enter the RunStream before reading its result")
        while not self._done():
            if await self._next() is None:
                break
        return self._outcome()

    def cancel(self) -> None:
        """Requests cancellation of unfinished work; idempotent, a no-op once settled."""
        if self._done() or self._closing is not None or self._started is None:
            return
        updates = self._started.updates
        assert updates is not None
        self._closing = asyncio.ensure_future(updates.aclose())

    def _claim(self) -> None:
        if not self._entered:
            raise RuntimeError("enter the RunStream before iterating it")
        if self._viewed:
            raise RuntimeError("the stream already has an iterator")
        self._viewed = self._iterating = True

    async def _text(self, delta: bool) -> AsyncIterator[str]:
        message_id, text = None, ""
        async for event in self._events():
            if event.type != "text_delta":
                continue
            if event.message_id != message_id:
                message_id, text = event.message_id, ""
            text += event.delta
            yield event.delta if delta else text
            self._presented[event.message_id] = (
                self._presented.get(event.message_id, "") + event.delta
            )

    async def _events(self) -> AsyncIterator[StreamEvent]:
        try:
            while not self._done():
                event = await self._next()
                if event is None:
                    break
                yield event
            self._outcome()  # a failure raises through iteration
        finally:
            self._iterating = False

    async def _next(self) -> StreamEvent | None:
        """The next event; ``None`` once the run settled."""
        started = self._started
        assert started is not None and started.updates is not None
        while True:
            update = await started.updates.recv()
            if update is None:
                # closed while pulling: the cancellation settles
                await self._close()
                if self._error is None and self._settled is None:
                    self._error = asyncio.CancelledError()
                return None
            match update:
                case records.RunUpdateStream(event=event):
                    self._sequence += 1
                    return _event(event, self._sequence, self._agent_name, started.id)
                case session_records.TextSessionUpdateStream():
                    return _event(
                        update.event,
                        update.sequence,
                        update.agent_name,
                        update.invocation_id,
                        update.frame_id.value,
                    )
                case session_records.TextSessionUpdateInvocationSettled(
                    invocation_id=invocation_id
                ):
                    assert started.release is not None
                    started.release(
                        session_records.TextSessionDelivery(
                            invocation_id=invocation_id, messages=self._presented
                        )
                    )
                    self._presented = {}
                case session_records.TextSessionUpdateParked(usage=usage):
                    if started.on_usage is not None:
                        started.on_usage(usage)
                case (
                    records.RunUpdateFinished(settled=settled)
                    | session_records.TextSessionUpdateFinished(settled=settled)
                ):
                    await started.cleanup()
                    self._settled = started.settle(settled)
                    return None
                case (
                    records.RunUpdateFailed(error=error)
                    | session_records.TextSessionUpdateFailed(error=error)
                ):
                    await started.cleanup()
                    self._error = started.fail(error)
                    return None
                case (
                    records.RunUpdateIncompleteCleanup(pending=pending)
                    | session_records.TextSessionUpdateIncompleteCleanup(pending=pending)
                ):
                    logger.error(
                        "the run's cleanup missed its deadline", extra={"pending": pending}
                    )

    async def _close(self) -> None:
        if self._closing is None:
            self.cancel()
        if self._closing is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._closing)
        if self._started is not None:
            await self._started.cleanup()
        if self._settled is None and self._error is None:
            self._error = asyncio.CancelledError()

    def _done(self) -> bool:
        return self._settled is not None or self._error is not None

    def _outcome(self) -> ResultT:
        if self._error is not None:
            raise self._error
        if self._settled is None:
            raise asyncio.CancelledError()
        return self._settled


async def _nothing() -> None:
    pass


@dataclass
class _Started(Generic[ResultT]):
    """A started invocation: its id, its updates (``None`` when it yielded before starting),
    how its final updates become the result or the exception, and ``cleanup``, the release of
    what it holds for its lifetime (its toolsets), awaited once it settled; idempotent."""

    id: str
    updates: _core.AgentRunUpdates | _core.TextSessionUpdates | None
    settle: Callable[[Any], ResultT]
    fail: Callable[[Any], BaseException]
    cleanup: Callable[[], Awaitable[None]] = _nothing
    release: Callable[[session_records.TextSessionDelivery], None] | None = None
    on_usage: Callable[[records.RunUsage], None] | None = None


def _event(
    event: records.StreamEvent,
    sequence: int,
    agent_name: str,
    invocation_id: str,
    frame_id: str | None = None,
) -> StreamEvent:
    common: dict[str, Any] = {
        "sequence": sequence,
        "agent_name": agent_name,
        "invocation_id": invocation_id,
        "frame_id": frame_id,
    }
    match event:
        case records.StreamEventTextDelta():
            return TextDeltaEvent(message_id=event.message_id, delta=event.delta, **common)
        case records.StreamEventTextEnd():
            return TextEndEvent(
                message_id=event.message_id, interrupted=event.interrupted, **common
            )
        case records.StreamEventToolCall():
            return ToolCallEvent(
                call_id=event.call_id,
                tool_name=event.tool_name,
                arguments=cast("dict[str, Any]", event.arguments or {}),
                **common,
            )
        case records.StreamEventToolResult():
            failed = event.status == "failed"
            return ToolResultEvent(
                call_id=event.call_id,
                tool_name=event.tool_name,
                outcome=event.status,
                output=None if failed else event.output,
                error=event.output if failed else None,
                **common,
            )
        case records.StreamEventProgress():
            return ProgressEvent(
                message=event.message,
                call_id=event.call_id,
                tool_name=event.tool_name,
                **common,
            )


__all__ = [
    "ProgressEvent",
    "RunStream",
    "StreamEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ToolCallEvent",
    "ToolResultEvent",
]
