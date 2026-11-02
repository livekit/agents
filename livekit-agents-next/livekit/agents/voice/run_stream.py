"""Typed Python result hydration over the native request stream owner."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Generic, Literal as Literal, TypeVar, cast

from pydantic import JsonValue as JsonValue

from .. import _core
from .._generated import agent_run as records, text_session as session_records
from .._generated.run_stream_events import (
    ProgressEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .._generated.usage import Usage

ResultT = TypeVar("ResultT")
ItemT = TypeVar("ItemT")
StreamEvent = TextDeltaEvent | TextEndEvent | ToolCallEvent | ToolResultEvent | ProgressEvent


class RunStream(Generic[ResultT]):
    """The handle ``async with agent.run_stream(...)`` returns; applications do not build it."""

    def __init__(
        self, start: Callable[[_core._core_RunStream], Awaitable[_Started[ResultT]]]
    ) -> None:
        self._core = _core._core_RunStream()
        self._start: Callable[[_core._core_RunStream], Awaitable[_Started[ResultT]]] | None = start
        self._started: _Started[ResultT] | None = None
        self._settled: ResultT | None = None
        self._error: BaseException | None = None

    @property
    def usage(self) -> Usage:
        """Latest immutable usage; reading it never advances the run."""
        return self._core.usage

    async def __aenter__(self) -> RunStream[ResultT]:
        self._core.enter()
        start, self._start = self._start, None
        assert start is not None
        try:
            self._started = await start(self._core)
        except BaseException:
            await self._core.rollback()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if not self._core.done:
                await self._core.aclose()
            if self._core.cancelled:
                self._error = self._error or asyncio.CancelledError()
                self._started = None
        except BaseException as error:
            self._capture(error)
            if exc is None:
                raise

    def stream_text(self, *, delta: bool = False) -> AsyncIterator[str]:
        """User-visible assistant text, as fragments or snapshots of the current message."""
        return _Iterator(self, self._core.text(delta))

    def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Typed semantic events from this run's single presentation view."""
        return _Iterator(self, self._core.events())

    async def result(self) -> ResultT:
        """Finish unconsumed work and hydrate the settled result once."""
        if self._error is not None:
            raise self._error
        if self._settled is not None:
            return self._settled
        consumer = self._core.begin_result()
        try:
            if self._error is not None:
                raise self._error
            if self._settled is None:
                try:
                    outcome = await self._core.result(consumer)
                    assert self._started is not None
                    if isinstance(
                        outcome,
                        (records.RunUpdateFinished, session_records.TextSessionUpdateFinished),
                    ):
                        self._settled = self._started.settle(outcome.settled)
                    else:
                        assert isinstance(
                            outcome,
                            (records.RunUpdateFailed, session_records.TextSessionUpdateFailed),
                        )
                        self._error = self._started.fail(outcome.error)
                        self._error.__dict__["input_requirements"] = list(
                            outcome.input_requirements
                        )
                        raise self._error
                except BaseException as error:
                    self._capture(error)
                    raise
                finally:
                    if self._settled is not None or self._error is not None:
                        self._started = None
            assert self._settled is not None
            return self._settled
        finally:
            self._core.end_result(consumer, self._settled is not None or self._error is not None)

    def cancel(self) -> None:
        """Request cancellation; the native owner retains cleanup through settlement."""
        self._core.cancel()

    def _capture(self, error: BaseException) -> None:
        # A cancelled Python waiter is distinct from cancellation of native execution.
        if self._core.done and (
            not isinstance(error, asyncio.CancelledError) or self._core.cancelled
        ):
            self._error = error
            self._started = None


class _Iterator(Generic[ItemT]):
    """Python exceptions/results around a native cursor; the cursor owns admission and reads."""

    def __init__(
        self,
        owner: RunStream[Any],
        cursor: _core._core_StreamEvents | _core._core_StreamText,
    ) -> None:
        self._owner = owner
        self._cursor = cursor

    def __aiter__(self) -> _Iterator[ItemT]:
        return self

    async def __anext__(self) -> ItemT:
        if not self._cursor.begin_read():
            raise StopAsyncIteration
        try:
            item = await self._cursor.recv()
        except BaseException as error:
            self._cursor.end_read(False)
            self._owner._capture(error)
            raise
        self._cursor.end_read(item is not None)
        if item is None:
            if self._owner._core.done:
                await self._owner.result()
            raise StopAsyncIteration
        return cast(ItemT, item)

    async def aclose(self) -> None:
        await self._cursor.aclose()


@dataclass
class _Started(Generic[ResultT]):
    """Only language-specific result and exception hydration; Rust owns the operation."""

    settle: Callable[[Any], ResultT]
    fail: Callable[[Any], BaseException]


__all__ = [
    "ProgressEvent",
    "RunStream",
    "StreamEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ToolCallEvent",
    "ToolResultEvent",
]
