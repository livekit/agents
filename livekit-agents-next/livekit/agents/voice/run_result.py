from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import json
import weakref
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
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

from opentelemetry import trace
from typing_extensions import TypeAliasType, TypeVar as TypeVarDefault

from .. import _core, llm
from .._core_view import to_core_item
from .._generated.agent_run import (
    EventExpectation,
    EventExpectationFunctionCall,
    EventExpectationFunctionCallOutput,
    EventExpectationMessage,
)
from ..telemetry import trace_types, tracer
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given

if TYPE_CHECKING:
    from ..metrics import AgentSessionUsage
    from .agent import Agent
    from .agent_session import VoiceSession


Output_T = TypeVarDefault("Output_T", default=Any)


@dataclass
class RunUsage:
    """Token usage of one or more runs; pass one to ``run(usage=)`` to keep adding to it."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def _add(self, other: RunUsage) -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.tool_calls += other.tool_calls

    @classmethod
    def _from_session_usage(cls, usage: AgentSessionUsage) -> RunUsage:
        result = cls()
        for model_usage in usage.model_usage:
            if model_usage.type != "llm_usage":
                continue
            result.input_tokens += model_usage.input_tokens
            result.output_tokens += model_usage.output_tokens
        return result


class _Settled:
    """What both branches of a settled invocation expose: its history and usage."""

    def __init__(
        self,
        *,
        new_messages: list[llm.ChatItem],
        all_messages: list[llm.ChatItem],
        usage: RunUsage,
        verbatim: str | None,
    ) -> None:
        self._new_messages = new_messages
        self._all_messages = all_messages
        self.usage = usage
        """This invocation's usage alone."""
        self.verbatim = verbatim
        """The part of the answer, or of the question, that reached the person exactly as
        written; ``None`` when nothing did (tools.md "Verbatim")."""

    def new_messages(self) -> list[llm.ChatItem]:
        """This invocation's input and everything it recorded, tool calls and results included."""
        return list(self._new_messages)

    def all_messages(self) -> list[llm.ChatItem]:
        """The supplied history plus this invocation, without instructions: the next direct
        run's ``message_history``."""
        return list(self._all_messages)

    @functools.cached_property
    def expect(self) -> RunAssert:
        """Assertions over this invocation's events, in order (testing.md)."""
        return RunAssert(_events(self._new_messages))


class RunCompleted(_Settled, Generic[Output_T]):
    """``RunResult`` with ``status == "completed"``: ``output`` is the accepted value."""

    status: Literal["completed"] = "completed"

    def __init__(self, *, output: Output_T, **settled: Any) -> None:
        super().__init__(**settled)
        self.output = output

    def __repr__(self) -> str:
        return f"RunCompleted(output={self.output!r})"


class RunInputRequired(_Settled):
    """``RunResult`` with ``status == "input_required"``: the invocation asks ``question``."""

    status: Literal["input_required"] = "input_required"

    def __init__(self, *, question: str, _answer_tool: str | None = None, **settled: Any) -> None:
        super().__init__(**settled)
        self.question = question
        # the tool the person's next message is handed to (`InputRequired(tool=)`)
        self._answer_tool = _answer_tool

    def __repr__(self) -> str:
        return f"RunInputRequired(question={self.question!r})"


RunResult = TypeAliasType(
    "RunResult", RunCompleted[Output_T] | RunInputRequired, type_params=(Output_T,)
)
"""One agent invocation (agents.md "Running a turn"): narrow on ``status`` for ``output`` or
``question``; both branches expose ``new_messages()``, ``all_messages()`` and ``usage``."""


@dataclass
class ChatMessageEvent:
    item: llm.ChatMessage
    type: Literal["message"] = "message"


@dataclass
class FunctionCallEvent:
    item: llm.FunctionCall
    type: Literal["function_call"] = "function_call"


@dataclass
class FunctionCallOutputEvent:
    item: llm.FunctionCallOutput
    type: Literal["function_call_output"] = "function_call_output"


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


class _VoiceRun:
    """One reply on a call to typed text (the remote session's text input), recorded until it
    settles. Once awaited it reads as a ``RunResult``: ``status`` is ``"completed"`` with
    ``output``, or ``"input_required"`` with ``question``."""

    def __init__(self, *, session: VoiceSession) -> None:
        self._done_fut = asyncio.Future[None]()
        self._session = session
        self._recorded_items: list[RunEvent] = []
        self._final_output: Any = None
        self._question: dict[str, Any] | None = None

    @property
    def status(self) -> Literal["completed", "input_required"]:
        """``"input_required"`` when the run ended on a question for the person."""
        return "completed" if self._question is None else "input_required"

    @property
    def output(self) -> Any:
        """The accepted output of a completed run: what a tool completed it with, else the last
        reply's text. ``None`` when the run asks the person."""
        if self._question is not None:
            return None
        if self._final_output is not None:
            return self._final_output
        for event in reversed(self._recorded_items):
            if isinstance(event, ChatMessageEvent) and event.item.role == "assistant":
                return event.item.text_content
        return None

    @property
    def question(self) -> str | None:
        """What an ``"input_required"`` run asks the person."""
        return None if self._question is None else str(self._question["question"])

    @property
    def verbatim(self) -> str | None:
        """The question, when it reaches the person exactly as written."""
        if self._question is None or not self._question["verbatim"]:
            return None
        return str(self._question["question"])

    def new_messages(self) -> list[llm.ChatItem]:
        """What this run added to the history."""
        return [event.item for event in self._recorded_items]

    def all_messages(self) -> list[llm.ChatItem]:
        """The history after the run, without the instructions."""
        return list(self._session.history.items)

    @functools.cached_property
    def expect(self) -> RunAssert:
        """Assertions over the events this run recorded, in order (testing.md)."""
        return RunAssert(self._recorded_items)

    def __await__(self) -> Generator[None, None, _VoiceRun]:
        async def _await_impl() -> _VoiceRun:
            await asyncio.shield(self._done_fut)
            return self

        return _await_impl().__await__()

    def _item_added(self, item: llm.ChatItem) -> None:
        if self._done_fut.done():
            return
        # an item told again changed in place (a call approval marked)
        for i, known in enumerate(self._recorded_items):
            if known.item.id == item.id and known.type == item.type:
                self._recorded_items[i] = replace(known, item=item)
                return

        event: RunEvent | None = None
        if item.type == "message":
            event = ChatMessageEvent(item=item)
        elif item.type == "function_call":
            event = FunctionCallEvent(item=item)
        elif item.type == "function_call_output":
            event = FunctionCallOutputEvent(item=item)

        if event is not None:
            index = self._find_insertion_index(created_at=event.item.created_at)
            self._recorded_items.insert(index, event)

    def _settle(
        self, *, output: Any, error: BaseException | None, question: dict[str, Any] | None = None
    ) -> None:
        """The core settled the run: ``output`` is what a tool completed it with, if any;
        ``question`` what the run asks the person instead."""
        if self._done_fut.done():
            return
        if error is not None:
            self._done_fut.set_exception(error)
        elif question is not None:
            self._question = question
            self._done_fut.set_result(None)
        else:
            self._final_output = output
            self._done_fut.set_result(None)

    def _find_insertion_index(self, *, created_at: float) -> int:
        """
        Returns the index to insert an item by creation time.

        Iterates in reverse, assuming items are sorted by `created_at`.
        Finds the position after the last item with `created_at <=` the given timestamp.
        """
        for i in reversed(range(len(self._recorded_items))):
            if self._recorded_items[i].item.created_at <= created_at:
                return i + 1

        return 0


EventType = Literal["message", "function_call", "function_call_output"]


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
        self._core = _core.RunEvents([to_core_item(e.item) for e in events])

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

    @tracer.start_as_current_span("judge_evaluation")
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

        current_span = trace.get_current_span()
        msg_content = self.event().item.text_content

        current_span.set_attribute(trace_types.ATTR_GEN_AI_OPERATION_NAME, "judge")
        current_span.set_attribute(trace_types.ATTR_GEN_AI_REQUEST_MODEL, llm_v.model)
        current_span.set_attribute(trace_types.ATTR_FUNCTION_TOOL_NAME, "judge_evaluation")
        current_span.set_attribute(
            trace_types.ATTR_FUNCTION_TOOL_ARGS,
            json.dumps({"intent": intent, "message": msg_content}),
        )

        from ..evals import _judge_intent

        try:
            judged = await _judge_intent(llm_v, intent, msg_content or "")
        except AssertionError as e:
            self._fail(str(e))

        current_span.set_attribute(trace_types.ATTR_FUNCTION_TOOL_IS_ERROR, not judged.success)
        current_span.set_attribute(trace_types.ATTR_FUNCTION_TOOL_OUTPUT, judged.reason)

        if judged.usage is not None:
            usage = cast("dict[str, int]", judged.usage)
            current_span.set_attributes(
                {
                    trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS: usage["prompt_tokens"],
                    trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS: usage["completion_tokens"],
                    trace_types.ATTR_GEN_AI_USAGE_INPUT_TEXT_TOKENS: usage["prompt_tokens"],
                    trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TEXT_TOKENS: usage["completion_tokens"],
                    trace_types.ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS: usage[
                        "prompt_cached_tokens"
                    ],
                }
            )

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
    agent: Agent, mocks: dict[str, Callable[..., Any]]
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
