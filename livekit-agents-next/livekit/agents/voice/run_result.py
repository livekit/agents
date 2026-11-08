from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NoReturn,
    TypeVar,
    cast,
    overload,
)

from pydantic import JsonValue as JsonValue

from .. import _core, llm
from .._generated import (
    agent_run as records,
    public_voice_run_result as _public,
    text_session as session_records,
)
from .._generated.agent_run import (
    EventExpectation,
    EventExpectationFunctionCall,
    EventExpectationFunctionCallOutput,
    EventExpectationMessage,
    EventType,
)
from .._generated.public_voice_run_result import *  # noqa: F403
from .._generated.usage import Usage
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given

if TYPE_CHECKING:
    from . import Agent
    from .agent_session import VoiceSession

RunEvent = ChatMessageEvent | FunctionCallEvent | FunctionCallOutputEvent


def _events(items: list[llm.ChatItem]) -> list[RunEvent]:
    """The events of an invocation's recorded items: its assistant messages, calls and outputs."""
    events: list[RunEvent] = []
    for item in items:
        if item.type == "message" and item.role == "assistant":
            events.append(ChatMessageEvent(item=item))
        elif item.type == "function_call":
            events.append(FunctionCallEvent(item=item))
        elif item.type == "function_call_output":
            events.append(FunctionCallOutputEvent(item=item))
    return events


def _expectation(
    type: EventType,
    *,
    role: NotGivenOr[llm.ChatRole] = NOT_GIVEN,
    name: NotGivenOr[str] = NOT_GIVEN,
    arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    output: NotGivenOr[str] = NOT_GIVEN,
    is_error: NotGivenOr[bool] = NOT_GIVEN,
) -> EventExpectation:
    if type == "message":
        return EventExpectationMessage(role=role if is_given(role) else None)
    if type == "function_call":
        return EventExpectationFunctionCall(
            name=name if is_given(name) else None,
            arguments=arguments if is_given(arguments) else None,
        )
    return EventExpectationFunctionCallOutput(
        output=output if is_given(output) else None,
        is_error=is_error if is_given(is_error) else None,
    )


class RunAssert:
    """``result.expect``: assertions over the events of the run, in order (testing.md)."""

    def __init__(self, events: list[RunEvent]) -> None:
        self._events = events
        self._core = _core.RunEvents([e.item for e in events])

    @overload
    def __getitem__(self, key: int) -> EventAssert: ...

    @overload
    def __getitem__(self, key: slice) -> EventRangeAssert: ...

    def __getitem__(self, key: int | slice) -> EventAssert | EventRangeAssert:
        """The event at an index (negative from the end), or a range for ``contains_*``."""
        __tracebackhide__ = True
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._events))
            if step != 1:
                raise ValueError("an event range takes no step")
            return EventRangeAssert(self, start, stop)
        return EventAssert(self, self._core.at(key, None))

    @overload
    def next_event(self, *, type: None = None) -> EventAssert: ...

    @overload
    def next_event(self, *, type: Literal["message"]) -> ChatMessageAssert: ...

    @overload
    def next_event(self, *, type: Literal["function_call"]) -> FunctionCallAssert: ...

    @overload
    def next_event(self, *, type: Literal["function_call_output"]) -> FunctionCallOutputAssert: ...

    def next_event(self, *, type: EventType | None = None) -> EventAssert | _Matched[Any]:
        """Advance to the next event, or to the next one of ``type``."""
        __tracebackhide__ = True
        if type is None:
            return EventAssert(self, self._core.next_event(None))
        return _MATCHED[type](self, self._core.next_event(_expectation(type)))

    @overload
    def skip_next_event_if(
        self, *, type: Literal["message"], role: NotGivenOr[llm.ChatRole] = NOT_GIVEN
    ) -> ChatMessageAssert | None: ...

    @overload
    def skip_next_event_if(
        self,
        *,
        type: Literal["function_call"],
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> FunctionCallAssert | None: ...

    @overload
    def skip_next_event_if(
        self,
        *,
        type: Literal["function_call_output"],
        output: NotGivenOr[str] = NOT_GIVEN,
        is_error: NotGivenOr[bool] = NOT_GIVEN,
    ) -> FunctionCallOutputAssert | None: ...

    def skip_next_event_if(self, *, type: EventType, **expected: Any) -> _Matched[Any] | None:
        """Skip the next event only if it matches; its assertion, or ``None``."""
        index = self._core.skip_next_event_if(_expectation(type, **expected))
        return None if index is None else _MATCHED[type](self, index)

    def skip_next(self, count: int = 1) -> RunAssert:
        """Advance past ``count`` events without asserting them."""
        __tracebackhide__ = True
        self._core.skip_next(count)
        return self

    def no_more_events(self) -> None:
        """Nothing is left after the current position."""
        __tracebackhide__ = True
        self._core.no_more_events()

    def contains_function_call(
        self,
        *,
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> FunctionCallAssert:
        __tracebackhide__ = True
        return self[:].contains_function_call(name=name, arguments=arguments)

    def contains_message(self, *, role: NotGivenOr[llm.ChatRole] = NOT_GIVEN) -> ChatMessageAssert:
        __tracebackhide__ = True
        return self[:].contains_message(role=role)

    def contains_function_call_output(
        self, *, output: NotGivenOr[str] = NOT_GIVEN, is_error: NotGivenOr[bool] = NOT_GIVEN
    ) -> FunctionCallOutputAssert:
        __tracebackhide__ = True
        return self[:].contains_function_call_output(output=output, is_error=is_error)


class EventAssert:
    """One event of the run, before its kind is asserted."""

    def __init__(self, parent: RunAssert, index: int) -> None:
        self._parent = parent
        self._index = index

    def event(self) -> RunEvent:
        return self._parent._events[self._index]

    def _check(self, type: EventType, **expected: Any) -> int:
        __tracebackhide__ = True
        return self._parent._core.at(self._index, _expectation(type, **expected))

    def is_function_call(
        self,
        *,
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> FunctionCallAssert:
        """A call of ``name`` whose arguments include ``arguments``."""
        __tracebackhide__ = True
        return FunctionCallAssert(
            self._parent, self._check("function_call", name=name, arguments=arguments)
        )

    def is_function_call_output(
        self, *, output: NotGivenOr[str] = NOT_GIVEN, is_error: NotGivenOr[bool] = NOT_GIVEN
    ) -> FunctionCallOutputAssert:
        __tracebackhide__ = True
        return FunctionCallOutputAssert(
            self._parent, self._check("function_call_output", output=output, is_error=is_error)
        )

    def is_message(self, *, role: NotGivenOr[llm.ChatRole] = NOT_GIVEN) -> ChatMessageAssert:
        __tracebackhide__ = True
        return ChatMessageAssert(self._parent, self._check("message", role=role))


class EventRangeAssert:
    """``result.expect[i:j]``: the ``contains_*`` checks over a range of events."""

    def __init__(self, parent: RunAssert, start: int, stop: int) -> None:
        self._parent = parent
        self._start = start
        self._stop = stop

    def _find(self, type: EventType, **expected: Any) -> int:
        __tracebackhide__ = True
        expectation = _expectation(type, **expected)
        return self._parent._core.contains(self._start, self._stop, expectation)

    def contains_function_call(
        self,
        *,
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> FunctionCallAssert:
        __tracebackhide__ = True
        return FunctionCallAssert(
            self._parent, self._find("function_call", name=name, arguments=arguments)
        )

    def contains_message(self, *, role: NotGivenOr[llm.ChatRole] = NOT_GIVEN) -> ChatMessageAssert:
        __tracebackhide__ = True
        return ChatMessageAssert(self._parent, self._find("message", role=role))

    def contains_function_call_output(
        self, *, output: NotGivenOr[str] = NOT_GIVEN, is_error: NotGivenOr[bool] = NOT_GIVEN
    ) -> FunctionCallOutputAssert:
        __tracebackhide__ = True
        return FunctionCallOutputAssert(
            self._parent, self._find("function_call_output", output=output, is_error=is_error)
        )


_Event_T = TypeVar("_Event_T", ChatMessageEvent, FunctionCallEvent, FunctionCallOutputEvent)


class _Matched(Generic[_Event_T]):
    """An event whose kind the assertion established."""

    def __init__(self, parent: RunAssert, index: int) -> None:
        self._parent = parent
        self._index = index

    def event(self) -> _Event_T:
        return cast(_Event_T, self._parent._events[self._index])


class FunctionCallAssert(_Matched[FunctionCallEvent]):
    pass


class FunctionCallOutputAssert(_Matched[FunctionCallOutputEvent]):
    pass


class ChatMessageAssert(_Matched[ChatMessageEvent]):
    def _fail(self, message: str) -> NoReturn:
        __tracebackhide__ = True
        raise AssertionError(self._parent._core.failure(message, self._index))

    async def judge(self, llm_v: llm.LLM[Any], *, intent: str) -> ChatMessageAssert:
        """
        Evaluate whether the message fulfills the given intent.

        Args:
            llm_v (llm.LLM): LLM instance for judgment.
            intent (str): Description of the expected intent.

        Returns:
            ChatMessageAssert: Self for chaining further assertions.

        Example:
            >>> await msg_assert.judge(llm, intent="should ask for size")
        """
        __tracebackhide__ = True

        msg_content = self.event().item.text_content

        from ..evals import _judge_intent

        try:
            judged = await _judge_intent(llm_v, intent, msg_content or "")
        except AssertionError as e:
            self._fail(str(e))

        if not judged.success:
            self._fail(f"Judgement failed: {judged.reason}")
        return self


_MATCHED: dict[EventType, type[_Matched[Any]]] = {
    "message": ChatMessageAssert,
    "function_call": FunctionCallAssert,
    "function_call_output": FunctionCallOutputAssert,
}

if TYPE_CHECKING:
    MockTools = dict[Agent, dict[str, Callable[..., Any]]]
_MockToolsContextVar = contextvars.ContextVar["MockTools"]("agents_mock_tools")
_SessionMockTools: weakref.WeakKeyDictionary[VoiceSession, MockTools] = weakref.WeakKeyDictionary()


def mock_tools(
    agent: Agent, mocks: dict[str, Callable[..., object]]
) -> contextlib.AbstractContextManager[None]:
    """Replace the bodies of ``agent``'s tools by mocks, sync or async, keyed by tool name, for
    the runs started inside the block and the calls they make, and for no other concurrent run::

        with mock_tools(scheduler, {"book": fake_book}):
            result = await scheduler.run("Book Monday at two.")

    Mocks intercept tool *execution* only; the model keeps seeing the real tool schemas. A mock
    declares the subset of the real tool's parameters it wants, by name.
    """
    return _mock_tools_ctx(agent, mocks)


def _mock_session_tools(
    session: VoiceSession, agent: Agent, mocks: dict[str, Callable[..., Any]]
) -> None:
    """The mocks of ``agent``'s tools on that call, for its lifetime; ``{}`` removes them. A
    ``mock_tools`` block's mocks take precedence."""
    _SessionMockTools.setdefault(session, {})[agent] = dict(mocks)


@contextmanager
def _mock_tools_ctx(
    agent: Agent, mocks: dict[str, Callable[..., Any]]
) -> Generator[None, None, None]:
    token = _MockToolsContextVar.set({**_MockToolsContextVar.get({}), agent: dict(mocks)})
    try:
        yield
    finally:
        _MockToolsContextVar.reset(token)


def _tool_body(
    agent: Agent, name: str, tool: Callable[..., Any], session: VoiceSession | None = None
) -> Callable[..., Any]:
    """What a call of ``agent``'s tool ``name`` runs: the tool, or the mock selected for it."""
    mock = _MockToolsContextVar.get({}).get(agent, {}).get(name)
    if mock is None and session is not None:
        mock = _SessionMockTools.get(session, {}).get(agent, {}).get(name)
    if mock is None:
        return tool

    signature = inspect.signature(tool)
    accepted = inspect.signature(mock).parameters
    takes_all = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())

    def call(*args: Any, **kwargs: Any) -> Any:
        named = signature.bind(*args, **kwargs).arguments
        return mock(**{k: v for k, v in named.items() if takes_all or k in accepted})

    return call


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
                        self._error = outcome.provider_error or outcome.error
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
    """Only language-specific result hydration; Rust owns the operation."""

    settle: Callable[[Any], ResultT]


__all__ = [
    "RunStream",
    "StreamEvent",
]
__all__ += _public.__all__
