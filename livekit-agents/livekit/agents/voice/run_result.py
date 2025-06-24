from __future__ import annotations

import json
import contextvars
import functools
import asyncio
from dataclasses import dataclass
import contextlib
from collections.abc import Generator
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Literal,
    TypeVar,
    Sequence,
    ContextManager,
    Type,
)


from contextlib import contextmanager

from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .. import llm, utils
from ..llm import function_tool, utils as llm_utils
from .speech_handle import SpeechHandle


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent


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


RunEvent = ChatMessageEvent | FunctionCallEvent | FunctionCallOutputEvent | AgentHandoffEvent


class RunResult(Generic[Run_T]):
    def __init__(self, *, output_type: type[Run_T]) -> None:
        self._handles: set[SpeechHandle | asyncio.Task] = set()

        self._done_fut = asyncio.Future[None]()
        self._output_type = output_type
        self._recorded_items: list[RunEvent] = []
        self._final_output: Run_T | None = None

        self.__last_speech_handle: SpeechHandle | None = None

    @property
    def events(self) -> list[RunEvent]:
        return self._recorded_items

    @functools.cached_property
    def expect(self) -> RunAssert:
        return RunAssert(self)

    @property
    def final_output(self) -> Run_T:
        if not self._done_fut.done():
            raise RuntimeError("cannot retrieve final_output, RunResult is not done")

        if not self._final_output:
            raise RuntimeError("no final output")

        return self._final_output

    def done(self) -> bool:
        return self._done_fut.done()

    def __await__(self) -> Generator[None, None, RunResult[Run_T]]:
        async def _await_impl() -> RunResult[Run_T]:
            await asyncio.shield(self._done_fut)
            return self

        return _await_impl().__await__()

    def _agent_handoff(self, *, old_agent: Agent | None, new_agent: Agent) -> None:
        self._recorded_items.append(AgentHandoffEvent(old_agent=old_agent, new_agent=new_agent))

    def _item_added(self, item: llm.ChatItem):
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

    def _mark_done_if_needed(self, handle: SpeechHandle | asyncio.Task):
        if isinstance(handle, SpeechHandle):
            self.__last_speech_handle = handle

        if all([handle.done() for handle in self._handles]):
            self._mark_done()

    def _mark_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            if self.__last_speech_handle is not None:
                self._final_output = self.__last_speech_handle._maybe_run_final_output
                
                if not isinstance(self._final_output, self._output_type):
                    self._done_fut.set_exception(RuntimeError(
                        f"Expected output of type {self._output_type.__name__}, "
                        f"got {type(self._final_output).__name__}"
                    ))
                    return

            self._done_fut.set_result(None)


class RunAssert:
    def __init__(self, run_result: RunResult):
        self._events_list = run_result.events
        self._current_index = 0

    def __getitem__(self, index: int) -> "EventAssert":
        if not (0 <= index < len(self._events_list)):
            self._raise_with_debug_info(
                f"nth({index}) out of range (total events: {len(self._events_list)})",
                index=index,
            )
        return EventAssert(self._events_list[index], self, index)

    def _current_event(self) -> "EventAssert":
        if self._current_index >= len(self._events_list):
            self._raise_with_debug_info("Expected another event, but none left.")
        event = self[self._current_index]
        return event

    def _raise_with_debug_info(self, message: str, index: int | None = None):
        marker_index = self._current_index if index is None else index
        lines: list[str] = []

        for i, event in enumerate(self._events_list):
            prefix = ">>>" if i == marker_index else "   "

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

        raise AssertionError(f"{message}\nContext around failure:\n" + "\n".join(lines))

    def skip_next(self, count: int = 1) -> "RunAssert":
        for i in range(count):
            if self._current_index >= len(self._events_list):
                self._raise_with_debug_info(
                    f"Tried to skip {count} event(s), but only {i} were available."
                )
            self._current_index += 1
        return self

    def maybe_message(
        self, *, role: NotGivenOr[llm.ChatRole] = NOT_GIVEN
    ) -> "ChatMessageAssert | None":
        try:
            ev = self._current_event().is_message(role=role)
            self._current_index += 1
            return ev
        except AssertionError:
            return None

    def maybe_function_call(
        self,
        *,
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "FunctionCallAssert | None":
        try:
            ev = self._current_event().is_function_call(name=name, arguments=arguments)
            self._current_index += 1
            return ev
        except AssertionError:
            return None

    def maybe_function_call_output(
        self, *, output: NotGivenOr[str] = NOT_GIVEN, is_error: NotGivenOr[bool] = NOT_GIVEN
    ) -> "FunctionCallOutputAssert | None":
        try:
            ev = self._current_event().is_function_call_output(output=output, is_error=is_error)
            self._current_index += 1
            return ev
        except AssertionError:
            return None

    def maybe_agent_handoff(
        self, *, new_agent_type: NotGivenOr[type[Agent]] = NOT_GIVEN
    ) -> "AgentHandoffAssert | None":
        try:
            ev = self._current_event().is_agent_handoff(new_agent_type=new_agent_type)
            self._current_index += 1
            return ev
        except AssertionError:
            return None

    def no_more_events(self) -> None:
        if self._current_index < len(self._events_list):
            event = self._events_list[self._current_index]
            self._raise_with_debug_info(
                f"Expected no more events, but found: {type(event).__name__}"
            )

    def message(self, *, role: NotGivenOr[llm.ChatRole] = NOT_GIVEN) -> "ChatMessageAssert":
        ev = self._current_event().is_message(role=role)
        self._current_index += 1
        return ev

    def function_call(
        self,
        *,
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "FunctionCallAssert":
        ev = self._current_event().is_function_call(name=name, arguments=arguments)
        self._current_index += 1
        return ev

    def function_call_output(
        self, *, output: NotGivenOr[str] = NOT_GIVEN, is_error: NotGivenOr[bool] = NOT_GIVEN
    ) -> "FunctionCallOutputAssert":
        ev = self._current_event().is_function_call_output(output=output, is_error=is_error)
        self._current_index += 1
        return ev

    def agent_handoff(
        self, *, new_agent_type: NotGivenOr[type[Agent]] = NOT_GIVEN
    ) -> "AgentHandoffAssert":
        ev = self._current_event().is_agent_handoff(new_agent_type=new_agent_type)
        self._current_index += 1
        return ev


class EventAssert:
    def __init__(self, event: RunEvent, parent: RunAssert, index: int = -1):
        self._event = event
        self._parent = parent
        self._index = index

    def _raise(self, message: str):
        self._parent._raise_with_debug_info(message, index=self._index)

    def is_function_call(
        self,
        *,
        name: NotGivenOr[str] = NOT_GIVEN,
        arguments: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "FunctionCallAssert":
        if not isinstance(self._event, FunctionCallEvent):
            self._raise("Expected FunctionCallEvent")
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
    ) -> "FunctionCallOutputAssert":
        if not isinstance(self._event, FunctionCallOutputEvent):
            self._raise("Expected FunctionCallOutputEvent")
        if is_given(output) and self._event.item.output != output:
            self._raise(f"Expected output '{output}', got '{self._event.item.output}'")
        if is_given(is_error) and self._event.item.is_error != is_error:
            self._raise(f"Expected is_error={is_error}, got {self._event.item.is_error}")
        return FunctionCallOutputAssert(self._event, self._parent, self._index)

    def is_message(self, *, role: NotGivenOr[llm.ChatRole] = NOT_GIVEN) -> "ChatMessageAssert":
        if not isinstance(self._event, ChatMessageEvent):
            self._raise("Expected ChatMessageEvent")
        if is_given(role) and self._event.item.role != role:
            self._raise(f"Expected role '{role}', got '{self._event.item.role}'")
        return ChatMessageAssert(self._event, self._parent, self._index)

    def is_agent_handoff(
        self, *, new_agent_type: NotGivenOr[type[Agent]] = NOT_GIVEN
    ) -> "AgentHandoffAssert":
        if not isinstance(self._event, AgentHandoffEvent):
            self._raise("Expected AgentHandoffEvent")
        if is_given(new_agent_type) and not isinstance(self._event.new_agent, new_agent_type):
            self._raise(
                f"Expected new_agent '{new_agent_type.__name__}', got '{type(self._event.new_agent).__name__}'"
            )
        return AgentHandoffAssert(self._event, self._parent, self._index)


class ChatMessageAssert:
    def __init__(self, event: ChatMessageEvent, parent: RunAssert, index: int):
        self._event = event
        self._parent = parent
        self._index = index

    def _raise(self, message: str):
        self._parent._raise_with_debug_info(message, index=self._index)

    def event(self) -> ChatMessageEvent:
        return self._event

    async def judge(self, llm_v: llm.LLM, *, intent: str) -> "ChatMessageAssert":
        msg_content = self._event.item.text_content

        if not msg_content:
            self._raise("The chat message is empty.")

        if not intent:
            self._raise("Intent is required to judge the message.")

        @function_tool
        async def check_intent(success: bool, reason: str):
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

        fnc_args, fnc_kwargs = llm_utils.prepare_function_arguments(
            fnc=check_intent, json_arguments=arguments
        )

        success, reason = await check_intent(*fnc_args, **fnc_kwargs)
        if not success:
            self._raise(f"Judgement failed: {reason}")

        return self


class FunctionCallAssert:
    def __init__(self, event: FunctionCallEvent, parent: RunAssert, index: int):
        self._event = event
        self._parent = parent
        self._index = index

    def item(self) -> FunctionCallEvent:
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
MockTools: dict[type[Agent], dict[str, Callable]]
_MockToolsContextVar = contextvars.ContextVar["MockTools"]("agents_mock_tools", default={})


@contextmanager
def mock_tools(agent: type["Agent"], mocks: dict[str, Callable]):
    """
    Temporarily assign a set of mock tool callables to a specific Agent type within the current context.

    Usage:
        with mock_tools(MyAgentClass, {"tool_name": mock_fn}):
            # inside this block, MyAgentClass will see the given mocks
    """
    current = _MockToolsContextVar.get()
    updated = {**current, agent: mocks}  # create a new dict
    token = _MockToolsContextVar.set(updated)
    try:
        yield
    finally:
        _MockToolsContextVar.reset(token)
