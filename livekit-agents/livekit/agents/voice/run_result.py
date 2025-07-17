from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    Union,
    overload,
)

from .. import llm
from ..llm import function_tool, utils as llm_utils
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent import Agent


lk_evals_verbose = int(os.getenv("LIVEKIT_EVALS_VERBOSE", 0))

Run_T = TypeVar("Run_T")


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


@dataclass
class AgentHandoffEvent:
    old_agent: Agent | None
    new_agent: Agent
    type: Literal["agent_handoff"] = "agent_handoff"


RunEvent = Union[ChatMessageEvent, FunctionCallEvent, FunctionCallOutputEvent, AgentHandoffEvent]


class RunResult(Generic[Run_T]):
    def __init__(self, *, user_input: str | None = None, output_type: type[Run_T] | None) -> None:
        self._handles: set[SpeechHandle | asyncio.Task] = set()

        self._done_fut = asyncio.Future[None]()
        self._user_input = user_input
        self._output_type = output_type
        self._recorded_items: list[RunEvent] = []
        self._final_output: Run_T | None = None

        self.__last_speech_handle: SpeechHandle | None = None

    @property
    def events(self) -> list[RunEvent]:
        """List of all recorded events generated during the run."""
        return self._recorded_items

    @functools.cached_property
    def expect(self) -> RunAssert:
        """
        Provides an assertion helper for verifying the run events.

        Returns:
            RunAssert: Assertion interface for run events.
        """
        # TODO(theomonnom): probably not the best place to log
        if lk_evals_verbose:
            events_str = "\n    ".join(_format_events(self.events))
            print(
                "\n+ RunResult(\n"
                f"   user_input=`{self._user_input}`\n"
                f"   events:\n    {events_str}\n"
                ")"
            )

        return RunAssert(self)

    @property
    def final_output(self) -> Run_T:
        """
        Returns the final output of the run after completion.

        Raises:
            RuntimeError: If the run is not complete or no output is set.

        Returns:
            Run_T: The final result output.
        """
        if not self._done_fut.done():
            raise RuntimeError("cannot retrieve final_output, RunResult is not done")

        if not self._final_output:
            raise RuntimeError("no final output")

        return self._final_output

    def done(self) -> bool:
        """Indicates whether the run has finished processing all events."""
        return self._done_fut.done()

    def __await__(self) -> Generator[None, None, RunResult[Run_T]]:
        async def _await_impl() -> RunResult[Run_T]:
            await asyncio.shield(self._done_fut)
            return self

        return _await_impl().__await__()

    def _agent_handoff(self, *, old_agent: Agent | None, new_agent: Agent) -> None:
        self._recorded_items.append(AgentHandoffEvent(old_agent=old_agent, new_agent=new_agent))

    def _item_added(self, item: llm.ChatItem) -> None:
        if self._done_fut.done():
            return

        if item.type == "message":
            self._recorded_items.append(ChatMessageEvent(item=item))
        elif item.type == "function_call":
            self._recorded_items.append(FunctionCallEvent(item=item))
        elif item.type == "function_call_output":
            self._recorded_items.append(FunctionCallOutputEvent(item=item))

    def _watch_handle(self, handle: SpeechHandle | asyncio.Task) -> None:
        self._handles.add(handle)

        if isinstance(handle, SpeechHandle):
            handle._add_item_added_callback(self._item_added)

        handle.add_done_callback(self._mark_done_if_needed)

    def _unwatch_handle(self, handle: SpeechHandle | asyncio.Task) -> None:
        self._handles.discard(handle)
        handle.remove_done_callback(self._mark_done_if_needed)

        if isinstance(handle, SpeechHandle):
            handle._remove_item_added_callback(self._item_added)

    def _mark_done_if_needed(self, handle: SpeechHandle | asyncio.Task | None) -> None:
        if isinstance(handle, SpeechHandle):
            self.__last_speech_handle = handle

        if all(handle.done() for handle in self._handles):
            self._mark_done()

    def _mark_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            if self.__last_speech_handle is None:
                self._done_fut.set_result(None)
                return

            final_output = self.__last_speech_handle._maybe_run_final_output
            if not isinstance(final_output, BaseException):
                if self._output_type and not isinstance(final_output, self._output_type):
                    self._done_fut.set_exception(
                        RuntimeError(
                            f"Expected output of type {self._output_type.__name__}, "
                            f"got {type(self._final_output).__name__}"
                        )
                    )
                else:
                    self._final_output = final_output
                    self._done_fut.set_result(None)
            else:
                self._done_fut.set_exception(final_output)


class RunAssert:
    def __init__(self, run_result: RunResult):
        self._events_list = run_result.events
        self._current_index = 0

    @overload
    def __getitem__(self, index: int) -> EventAssert: ...
    @overload
    def __getitem__(self, s: slice) -> EventRangeAssert: ...

    def __getitem__(self, key: [int, slice]) -> EventAssert | EventRangeAssert:  # type: ignore
        """
        Access a specific event or range for assertions.

        Args:
            key (int | slice): Index or slice of events.

        Returns:
            EventAssert: Assertion for a single event when key is int.
            EventRangeAssert: Assertion for a span of events when key is slice.

        Raises:
            TypeError: If key is not an int or slice.
            AssertionError: If index is out of range.

        Examples:
            # Single event access
            >>> result.expect[0].is_message(role="user")
            >>> result.expect[-1].is_message(role="assistant")

            # Full range access
            >>> result.expect[:].contains_function_call(name="foo")

            # Partial range access
            >>> result.expect[0:2].contains_message(role="assistant")
        """
        if isinstance(key, slice):
            events = self._events_list[key]
            return EventRangeAssert(events, self, key)
        if isinstance(key, int):
            if key < 0:
                key += len(self._events_list)

            if not (0 <= key < len(self._events_list)):
                self._raise_with_debug_info(
                    f"nth({key}) out of range (total events: {len(self._events_list)})",
                    index=key,
                )
            return EventAssert(self._events_list[key], self, key)

        raise TypeError(
            f"{type(self).__name__} indices must be int or slice, not {type(key).__name__}"
        )

    def _current_event(self) -> EventAssert:
        __tracebackhide__ = True

        if self._current_index >= len(self._events_list):
            self._raise_with_debug_info("Expected another event, but none left.")

        event = self[self._current_index]
        return event

    def _raise_with_debug_info(self, message: str, index: int | None = None) -> None:
        __tracebackhide__ = True

        marker_index = self._current_index if index is None else index
        events_str = "\n".join(_format_events(self._events_list, selected_index=marker_index))
        raise AssertionError(f"{message}\nContext around failure:\n" + events_str)

    @overload
    def next_event(self, *, type: None = None) -> EventAssert: ...

    @overload
    def next_event(self, *, type: Literal["message"]) -> ChatMessageAssert: ...

    @overload
    def next_event(self, *, type: Literal["function_call"]) -> FunctionCallAssert: ...

    @overload
    def next_event(self, *, type: Literal["function_call_output"]) -> FunctionCallOutputAssert: ...

    @overload
    def next_event(self, *, type: Literal["agent_handoff"]) -> AgentHandoffAssert: ...

    def next_event(
        self,
        *,
        type: Literal["message", "function_call", "function_call_output", "agent_handoff"]
        | None = None,
    ) -> (
        EventAssert
        | ChatMessageAssert
        | FunctionCallAssert
        | FunctionCallOutputAssert
        | AgentHandoffAssert
    ):
        """
        Advance to the next event, optionally filtering by type.

        Args:
            type (str, optional): Event type to match.

        Returns:
            EventAssert or subclass: Assertion object for the matched event.

        Example:
            >>> result.expect.next_event(type="function_call").is_function_call(name="foo")
        """
        __tracebackhide__ = True

        while True:
            ev_assert = self._current_event()
            self._current_index += 1

            if type is None or ev_assert.event().type == type:
                break

        if type == "message":
            return ev_assert.is_message()
        elif type == "function_call":
            return ev_assert.is_function_call()
        elif type == "function_call_output":
            return ev_assert.is_function_call_output()
        elif type == "agent_handoff":
            return ev_assert.is_agent_handoff()

        return ev_assert

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

    @overload
    def skip_next_event_if(
        self, *, type: Literal["agent_handoff"], new_agent_type: NotGivenOr[type[Agent]] = NOT_GIVEN
    ) -> AgentHandoffAssert | None: ...

    def skip_next_event_if(
        self,
        *,
        type: Literal["message", "function_call", "function_call_output", "agent_handoff"],
        role: NotGivenOr[llm.ChatRole] = NOT_GIVEN,
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        output: NotGivenOr[str] = NOT_GIVEN,
        is_error: NotGivenOr[bool] = NOT_GIVEN,
        new_agent_type: NotGivenOr[type[Agent]] = NOT_GIVEN,
    ) -> (
        ChatMessageAssert
        | AgentHandoffAssert
        | FunctionCallAssert
        | FunctionCallOutputAssert
        | None
    ):
        """
        Conditionally skip the next event if it matches criteria.

        Args:
            type (str): Type of event to check.
            role (ChatRole, optional): Required role for message events.
            name (str, optional): Required function name for calls.
            arguments (dict, optional): Required args for function calls.
            output (str, optional): Required output for function call outputs.
            is_error (bool, optional): Required error flag for call outputs.
            new_agent_type (type, optional): Required agent class for handoffs.

        Returns:
            EventAssert or None: The skipped event assertion if matched.

        Example:
            >>> skipped = result.expect.skip_next_event_if(type="message", role="assistant")
        """
        __tracebackhide__ = True
        try:
            ev: (
                ChatMessageAssert
                | FunctionCallAssert
                | FunctionCallOutputAssert
                | AgentHandoffAssert
                | None
            ) = None
            if type == "message":
                ev = self._current_event().is_message(role=role)
            elif type == "function_call":
                ev = self._current_event().is_function_call(name=name, arguments=arguments)
            elif type == "function_call_output":
                ev = self._current_event().is_function_call_output(output=output, is_error=is_error)
            elif type == "agent_handoff":
                ev = self._current_event().is_agent_handoff(new_agent_type=new_agent_type)

            self._current_index += 1
            return ev
        except AssertionError:
            return None

        raise RuntimeError("unknown event type")

    def skip_next(self, count: int = 1) -> RunAssert:
        """
        Skip a specified number of upcoming events without assertions.

        Args:
            count (int): Number of events to skip.

        Returns:
            RunAssert: Self for chaining.

        Example:
            >>> result.expect.skip_next(2)
        """

        __tracebackhide__ = True

        for i in range(count):
            if self._current_index >= len(self._events_list):
                self._raise_with_debug_info(
                    f"Tried to skip {count} event(s), but only {i} were available."
                )
            self._current_index += 1
        return self

    def no_more_events(self) -> None:
        """
        Assert that there are no further events.

        Raises:
            AssertionError: If unexpected events remain.

        Example:
            >>> result.expect.no_more_events()
        """
        __tracebackhide__ = True

        if self._current_index < len(self._events_list):
            event = self._events_list[self._current_index]
            self._raise_with_debug_info(
                f"Expected no more events, but found: {type(event).__name__}"
            )

    def contains_function_call(
        self,
        *,
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> FunctionCallAssert:
        """
        Assert existence of a function call event matching criteria.

        Args:
            name (str, optional): Function name to match.
            arguments (dict, optional): Arguments to match.

        Returns:
            FunctionCallAssert: Assertion for the matching call.

        Example:
            >>> result.expect.contains_function_call(name="foo")
        """
        __tracebackhide__ = True
        return self[:].contains_function_call(name=name, arguments=arguments)

    def contains_message(
        self,
        *,
        role: NotGivenOr[llm.ChatRole] = NOT_GIVEN,
    ) -> ChatMessageAssert:
        """
        Assert existence of a message event matching criteria.

        Args:
            role (ChatRole, optional): Role to match.

        Returns:
            ChatMessageAssert: Assertion for the matching message.

        Example:
            >>> result.expect.contains_message(role="user")
        """
        __tracebackhide__ = True
        return self[:].contains_message(role=role)

    def contains_function_call_output(
        self,
        *,
        output: NotGivenOr[str] = NOT_GIVEN,
        is_error: NotGivenOr[bool] = NOT_GIVEN,
    ) -> FunctionCallOutputAssert:
        """
        Assert existence of a function call output event matching criteria.

        Args:
            output (str, optional): Output string to match.
            is_error (bool, optional): Error flag to match.

        Returns:
            FunctionCallOutputAssert: Assertion for the matching output.

        Example:
            >>> result.expect.contains_function_call_output(is_error=True)
        """
        __tracebackhide__ = True
        return self[:].contains_function_call_output(output=output, is_error=is_error)

    def contains_agent_handoff(
        self, *, new_agent_type: NotGivenOr[type[Agent]] = NOT_GIVEN
    ) -> AgentHandoffAssert:
        """
        Assert existence of an agent handoff event matching criteria.

        Args:
            new_agent_type (type, optional): Expected new agent class.

        Returns:
            AgentHandoffAssert: Assertion for the matching handoff.

        Example:
            >>> result.expect.contains_agent_handoff(new_agent_type=MyAgent)
        """
        __tracebackhide__ = True
        return self[:].contains_agent_handoff(new_agent_type=new_agent_type)


class EventAssert:
    def __init__(self, event: RunEvent, parent: RunAssert, index: int = -1):
        self._event = event
        self._parent = parent
        self._index = index

    def _raise(self, message: str) -> None:
        __tracebackhide__ = True
        self._parent._raise_with_debug_info(message, index=self._index)

    def event(self) -> RunEvent:
        return self._event

    def is_function_call(
        self,
        *,
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> FunctionCallAssert:
        """
        Verify this event is a function call with matching details.

        Args:
            name (str, optional): Expected function name.
            arguments (dict, optional): Expected call arguments.

        Returns:
            FunctionCallAssert: Assertion for the function call.

        Raises:
            AssertionError: If the event is not a function call or details mismatch.

        Example:
            >>> ev_assert.is_function_call(name="foo", arguments={"x": 1})
        """
        __tracebackhide__ = True

        if not isinstance(self._event, FunctionCallEvent):
            self._raise("Expected FunctionCallEvent")

        assert isinstance(self._event, FunctionCallEvent)  # type check

        if is_given(name) and self._event.item.name != name:
            self._raise(f"Expected call name '{name}', got '{self._event.item.name}'")
        if is_given(arguments):
            actual = json.loads(self._event.item.arguments)
            for key, value in arguments.items():
                if key not in actual or actual[key] != value:
                    self._raise(f"For key '{key}', expected {value}, got {actual.get(key)}")

        return FunctionCallAssert(self._event, self._parent, self._index)

    def is_function_call_output(
        self, *, output: NotGivenOr[str] = NOT_GIVEN, is_error: NotGivenOr[bool] = NOT_GIVEN
    ) -> FunctionCallOutputAssert:
        """
        Verify this event is a function call output with matching details.

        Args:
            output (str, optional): Expected output text.
            is_error (bool, optional): Expected error flag.

        Returns:
            FunctionCallOutputAssert: Assertion for the output.

        Raises:
            AssertionError: If the event is not function output or details mismatch.

        Example:
            >>> ev_assert.is_function_call_output(output="OK", is_error=False)
        """
        __tracebackhide__ = True

        if not isinstance(self._event, FunctionCallOutputEvent):
            self._raise("Expected FunctionCallOutputEvent")

        assert isinstance(self._event, FunctionCallOutputEvent)  # type check

        if is_given(output) and self._event.item.output != output:
            self._raise(f"Expected output '{output}', got '{self._event.item.output}'")
        if is_given(is_error) and self._event.item.is_error != is_error:
            self._raise(f"Expected is_error={is_error}, got {self._event.item.is_error}")
        return FunctionCallOutputAssert(self._event, self._parent, self._index)

    def is_message(self, *, role: NotGivenOr[llm.ChatRole] = NOT_GIVEN) -> ChatMessageAssert:
        """
        Verify this event is a message from the given role.

        Args:
            role (ChatRole, optional): Expected sender role.

        Returns:
            ChatMessageAssert: Assertion for the message.

        Raises:
            AssertionError: If the event is not a message or role mismatch.

        Example:
            >>> ev_assert.is_message(role="assistant")
        """
        __tracebackhide__ = True

        if not isinstance(self._event, ChatMessageEvent):
            self._raise("Expected ChatMessageEvent")

        assert isinstance(self._event, ChatMessageEvent)  # type check

        if is_given(role) and self._event.item.role != role:
            self._raise(f"Expected role '{role}', got '{self._event.item.role}'")
        return ChatMessageAssert(self._event, self._parent, self._index)

    def is_agent_handoff(
        self, *, new_agent_type: NotGivenOr[type[Agent]] = NOT_GIVEN
    ) -> AgentHandoffAssert:
        """
        Verify this event is an agent handoff.

        Args:
            new_agent_type (type, optional): Expected new agent class.

        Returns:
            AgentHandoffAssert: Assertion for the handoff.

        Raises:
            AssertionError: If the event is not an agent handoff or type mismatch.

        Example:
            >>> ev_assert.is_agent_handoff(new_agent_type=MyAgent)
        """
        __tracebackhide__ = True

        if not isinstance(self._event, AgentHandoffEvent):
            self._raise("Expected AgentHandoffEvent")

        assert isinstance(self._event, AgentHandoffEvent)  # type check

        if is_given(new_agent_type) and not isinstance(self._event.new_agent, new_agent_type):
            self._raise(
                f"Expected new_agent '{new_agent_type.__name__}', got '{type(self._event.new_agent).__name__}'"
            )
        return AgentHandoffAssert(self._event, self._parent, self._index)


class EventRangeAssert:
    def __init__(self, events: list[RunEvent], parent: RunAssert, rng: slice):
        self._events = events
        self._parent = parent
        self._rng = rng

    def contains_function_call(
        self,
        *,
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> FunctionCallAssert:
        """
        Assert that a function call matching criteria exists in the event range.

        Args:
            name (str, optional): Expected function name.
            arguments (dict, optional): Expected call arguments.

        Returns:
            FunctionCallAssert: Assertion for the matched function call.

        Raises:
            AssertionError: If no matching function call is found in range.

        Example:
            >>> result.expect[0:3].contains_function_call(name="foo")
        """
        __tracebackhide__ = True

        for idx, ev in enumerate(self._events):
            candidate = EventAssert(ev, self._parent, (self._rng.start or 0) + idx)
            with contextlib.suppress(AssertionError):
                return candidate.is_function_call(name=name, arguments=arguments)

        self._parent._raise_with_debug_info(
            f"No FunctionCallEvent satisfying criteria found in range {self._rng!r}"
        )
        raise RuntimeError("unreachable")

    def contains_message(
        self,
        *,
        role: NotGivenOr[llm.ChatRole] = NOT_GIVEN,
    ) -> ChatMessageAssert:
        """
        Assert that a message matching criteria exists in the event range.

        Args:
            role (ChatRole, optional): Expected sender role.

        Returns:
            ChatMessageAssert: Assertion for the matched message.

        Raises:
            AssertionError: If no matching message is found in range.

        Example:
            >>> result.expect[:2].contains_message(role="assistant")
        """
        __tracebackhide__ = True

        for idx, ev in enumerate(self._events):
            candidate = EventAssert(ev, self._parent, (self._rng.start or 0) + idx)
            with contextlib.suppress(AssertionError):
                return candidate.is_message(role=role)

        self._parent._raise_with_debug_info(
            f"No ChatMessageEvent matching criteria found in range {self._rng!r}"
        )
        raise RuntimeError("unreachable")

    def contains_function_call_output(
        self,
        *,
        output: NotGivenOr[str] = NOT_GIVEN,
        is_error: NotGivenOr[bool] = NOT_GIVEN,
    ) -> FunctionCallOutputAssert:
        """
        Assert that a function call output matching criteria exists in the event range.

        Args:
            output (str, optional): Expected output text.
            is_error (bool, optional): Expected error flag.

        Returns:
            FunctionCallOutputAssert: Assertion for the matched output.

        Raises:
            AssertionError: If no matching output is found in range.

        Example:
            >>> result.expect[1:4].contains_function_call_output(is_error=True)
        """
        __tracebackhide__ = True

        for idx, ev in enumerate(self._events):
            candidate = EventAssert(ev, self._parent, (self._rng.start or 0) + idx)
            with contextlib.suppress(AssertionError):
                return candidate.is_function_call_output(output=output, is_error=is_error)

        self._parent._raise_with_debug_info(
            f"No FunctionCallOutputEvent matching criteria found in range {self._rng!r}"
        )
        raise RuntimeError("unreachable")

    def contains_agent_handoff(
        self, *, new_agent_type: NotGivenOr[type[Agent]] = NOT_GIVEN
    ) -> AgentHandoffAssert:
        """
        Assert that an agent handoff matching criteria exists in the event range.

        Args:
            new_agent_type (type, optional): Expected new agent class.

        Returns:
            AgentHandoffAssert: Assertion for the matched handoff.

        Raises:
            AssertionError: If no matching handoff is found in range.

        Example:
            >>> result.expect[0:3].contains_agent_handoff(new_agent_type=MyAgent)
        """
        __tracebackhide__ = True

        for idx, ev in enumerate(self._events):
            candidate = EventAssert(ev, self._parent, (self._rng.start or 0) + idx)
            with contextlib.suppress(AssertionError):
                return candidate.is_agent_handoff(new_agent_type=new_agent_type)

        self._parent._raise_with_debug_info(
            f"No AgentHandoffEvent matching criteria found in range {self._rng!r}"
        )
        raise RuntimeError("unreachable")


class ChatMessageAssert:
    def __init__(self, event: ChatMessageEvent, parent: RunAssert, index: int):
        self._event = event
        self._parent = parent
        self._index = index

    def _raise(self, message: str) -> None:
        __tracebackhide__ = True
        self._parent._raise_with_debug_info(message, index=self._index)

    def event(self) -> ChatMessageEvent:
        return self._event

    async def judge(self, llm_v: llm.LLM, *, intent: str) -> ChatMessageAssert:
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

        msg_content = self._event.item.text_content

        if not msg_content:
            self._raise("The chat message is empty.")
            raise RuntimeError("unreachable")

        if not intent:
            self._raise("Intent is required to judge the message.")
            raise RuntimeError("unreachable")

        @function_tool
        async def check_intent(success: bool, reason: str) -> tuple[bool, str]:
            """
            Determines whether the message correctly fulfills the given intent.

            Args:
                success: Whether the message satisfies the intent.
                reason: A concise explanation justifying the result.
            """
            return success, reason

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(
            role="system",
            content=(
                "You are a test evaluator for conversational agents.\n"
                "You will be shown a message and a target intent. Determine whether the message accomplishes the intent.\n"
                "Only respond by calling the `check_intent(success: bool, reason: str)` function with your final judgment.\n"
                "Be strict: if the message does not clearly fulfill the intent, return `success = False` and explain why."
            ),
        )
        chat_ctx.add_message(
            role="user",
            content=(
                "Check if the following message fulfills the given intent.\n\n"
                f"Intent:\n{intent}\n\n"
                f"Message:\n{msg_content}"
            ),
        )

        arguments: str | None = None

        # TODO(theomonnom): LLMStream should provide utilities to make function calling easier.
        async for chunk in llm_v.chat(
            chat_ctx=chat_ctx,
            tools=[check_intent],
            tool_choice={"type": "function", "function": {"name": "check_intent"}},
            extra_kwargs={"temperature": 0.0},
        ):
            if not chunk.delta:
                continue

            if chunk.delta.tool_calls:
                tool = chunk.delta.tool_calls[0]
                arguments = tool.arguments

        if not arguments:
            self._raise("LLM did not return any arguments for evaluation.")

        assert isinstance(arguments, str)  # type check

        fnc_args, fnc_kwargs = llm_utils.prepare_function_arguments(
            fnc=check_intent, json_arguments=arguments
        )

        success, reason = await check_intent(*fnc_args, **fnc_kwargs)

        if not success:
            self._raise(f"Judgement failed: {reason}")
        elif lk_evals_verbose:
            from textwrap import shorten

            print_msg = shorten(msg_content.replace("\n", "\\n"), width=30, placeholder="...")
            print(f"- Judgment succeeded for `{print_msg}`: `{reason}`")

        return self


class FunctionCallAssert:
    def __init__(self, event: FunctionCallEvent, parent: RunAssert, index: int):
        self._event = event
        self._parent = parent
        self._index = index

    def event(self) -> FunctionCallEvent:
        return self._event


class FunctionCallOutputAssert:
    def __init__(self, event: FunctionCallOutputEvent, parent: RunAssert, index: int):
        self._event = event
        self._parent = parent
        self._index = index

    def event(self) -> FunctionCallOutputEvent:
        return self._event


class AgentHandoffAssert:
    def __init__(self, event: AgentHandoffEvent, parent: RunAssert, index: int):
        self._event = event
        self._parent = parent
        self._index = index

    def event(self) -> AgentHandoffEvent:
        return self._event


# to make testing easier, we allow sync Callable too
if TYPE_CHECKING:
    MockTools = dict[type[Agent], dict[str, Callable]]
_MockToolsContextVar = contextvars.ContextVar["MockTools"]("agents_mock_tools")


@contextmanager
def mock_tools(agent: type[Agent], mocks: dict[str, Callable]) -> Generator[None, None, None]:
    """
    Temporarily assign a set of mock tool callables to a specific Agent type within the current context.

    Usage:
        with mock_tools(MyAgentClass, {"tool_name": mock_fn}):
            # inside this block, MyAgentClass will see the given mocks
    """  # noqa: E501
    current = _MockToolsContextVar.get({})
    updated = {**current, agent: mocks}  # create a new dict
    token = _MockToolsContextVar.set(updated)
    try:
        yield
    finally:
        _MockToolsContextVar.reset(token)


def _format_events(events: list[RunEvent], *, selected_index: int | None = None) -> list[str]:
    lines: list[str] = []
    for i, event in enumerate(events):
        prefix = ""
        if selected_index is not None:
            prefix = ">>>" if i == selected_index else "   "

        if isinstance(event, (ChatMessageEvent, FunctionCallEvent, FunctionCallOutputEvent)):
            item_repr = event.item.model_dump(
                exclude_none=True,
                exclude_defaults=True,
                exclude={"type", "id", "call_id", "created_at"},
            )
            line = f"{prefix} [{i}] {event.__class__.__name__}(item={item_repr})"
        elif isinstance(event, AgentHandoffEvent):
            line = (
                f"{prefix} [{i}] AgentHandoffEvent("
                f"old_agent={event.old_agent}, new_agent={event.new_agent})"
            )
        else:
            line = f"{prefix} [{i}] {event}"

        lines.append(line)

    return lines
