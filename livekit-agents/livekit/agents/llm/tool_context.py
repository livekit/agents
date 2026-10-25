# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import functools
import inspect
import itertools
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Flag, auto
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    TypeGuard,
    TypeVar,
    get_type_hints,
    overload,
)

from pydantic import Field
from typing_extensions import NotRequired, ParamSpec, Required, Self, TypedDict

from ..log import logger
from . import _provider_format

if TYPE_CHECKING:
    from ..voice.events import RunContext


class Tool(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...


class ProviderTool(Tool):
    def __init__(self, *, id: str) -> None:
        self._id = id

    @property
    def id(self) -> str:
        return self._id


class Toolset:
    @dataclass
    class ToolCalledEvent:
        ctx: RunContext
        arguments: dict[str, Any]

    @dataclass
    class ToolCompletedEvent:
        ctx: RunContext
        output: Any | Exception | None

    def __init__(self, *, id: str, tools: Sequence[Tool | Toolset] | None = None) -> None:
        self._id = id
        self._tools: Sequence[Tool | Toolset] = list(tools) if tools is not None else []
        self._tools.extend(find_function_tools(self))

    @property
    def id(self) -> str:
        return self._id

    @property
    def tools(self) -> Sequence[Tool | Toolset]:
        return self._tools

    async def setup(self) -> Self:
        """Initialize the toolset and any nested toolsets.

        Called automatically by ``AgentActivity`` when an agent starts.
        """
        toolsets = [tool for tool in self.tools if isinstance(tool, Toolset)]
        if toolsets:
            await asyncio.gather(*(toolset.setup() for toolset in toolsets))
        return self

    async def aclose(self) -> None:
        """Close the toolset and release any held resources.

        Agent-scoped toolsets (passed to ``Agent(tools=...)``) are closed when the
        ``AgentActivity`` ends (on agent transition or session close). Session-scoped
        toolsets (passed to ``LiveSession(tools=...)``) are closed only when the
        ``LiveSession`` shuts down.
        """
        toolsets = [tool for tool in self._tools if isinstance(tool, Toolset)]
        if toolsets:
            await asyncio.gather(*(toolset.aclose() for toolset in toolsets))


# Used by ToolChoice
class Function(TypedDict, total=False):
    name: Required[str]


class NamedToolChoice(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[Function]


ToolChoice = NamedToolChoice | Literal["auto", "required", "none"]


@dataclass
class ToolResult:
    """A function tool's output, whether it requests a follow-up reply, and whether the person
    gets it as written.

    Args:
        output: The value to return to the model, such as a string, dict, list,
            or None. For a handoff, return ``(agent, ToolResult(...))``.
        reply_required: Whether this result requests a reply. Defaults to True.
            False keeps the output in chat history without requesting a reply.
            Other tool results can still request a reply, and interruption or
            ``cancel_tool_reply()`` can still prevent one.
        verbatim: The output is text the person gets exactly as written: spoken with ``say()``
            on a call, sent unchanged in text, relayed unchanged by a delegate. With
            ``reply_required=False`` it ends the turn; otherwise the model continues after it.

    Example:
        return ToolResult("Successfully sent DTMF events: 1", reply_required=False)
    """

    output: Any
    reply_required: bool = field(default=True, kw_only=True)
    verbatim: bool = field(default=False, kw_only=True)


class ToolError(Exception):
    def __init__(self, message: str) -> None:
        """
        Exception raised within AI functions.

        This exception should be raised by users when an error occurs
        in the context of AI operations. The provided message will be
        visible to the LLM, allowing it to understand the context of
        the error during FunctionOutput generation.
        """
        super().__init__(message)
        self._message = message

    @property
    def message(self) -> str:
        return self._message


class StopResponse(Exception):
    def __init__(self) -> None:
        """
        Exception raised within AI functions.

        This exception can be raised by the user to indicate that
        the agent should not generate a response for the current
        function call.
        """
        super().__init__()


class ModelRetry(Exception):
    def __init__(self, message: str) -> None:
        """Send the model back with a reason.

        Raised by an output validator or a tool: the message reaches the model as the reason
        to try again, up to the run's step budget.
        """
        super().__init__(message)
        self._message = message

    @property
    def message(self) -> str:
        return self._message


Transition_T = TypeVar("Transition_T")


@dataclass(frozen=True)
class Completed(Generic[Transition_T]):
    """The task is done: ``value`` is its result, and its last message when it is text.
    ``verbatim`` makes that text reach the person exactly as written."""

    value: Transition_T
    verbatim: bool = field(default=False, kw_only=True)


@dataclass(frozen=True)
class Failed:
    """The task failed: ``error`` is its last message."""

    error: str


@dataclass(frozen=True)
class Rejected:
    """The task will not run this input, from ``on_enter``."""

    reason: str


@dataclass(frozen=True)
class InputRequired:
    """The turn or task waits on the person for an answer to ``question``, with no timeout.

    Only for an input a tool needs to go on (an email, a yes or no on a read-back, a card
    number); a conversational question is a plain reply, and the next message is a new request.
    The agent facing the person asks it in its own words, or exactly as written with
    ``verbatim=True``. The answer continues the waiting task as the person's next message, and
    the task's model reads it; the call keeps the question as its output. An input that fixed
    logic must collect is an ``AgentTask``.
    """

    question: str
    verbatim: bool = False


@dataclass(frozen=True)
class Working:
    """The task waits on a system: nothing is delivered, ``message`` is its status."""

    tool: FunctionTool | RawFunctionTool | None = None
    timeout: timedelta | None = None
    message: str | None = None


Transition = Completed[Any] | Failed | Rejected | InputRequired | Working
"""What a tool or a hook returns to change the state of the turn or the task."""


def is_transition(value: Any) -> TypeGuard[Transition]:
    return isinstance(value, (Completed, Failed, Rejected, InputRequired, Working))


class ToolFlag(Flag):
    NONE = 0
    IGNORE_ON_ENTER = auto()
    CANCELLABLE = auto()


DuplicateMode = Literal["allow", "reject", "replace", "confirm"]

DuplicateScope = Literal["name", "name_and_args"]
"""What counts as a duplicate of an in-flight call, i.e. what ``on_duplicate`` acts on.

``"name"``           any in-flight call of the same tool (default).
``"name_and_args"``  same tool *and* same arguments, so concurrent calls of one
                     tool with different arguments are not duplicates::

                         @function_tool(on_duplicate="reject", duplicate_scope="name_and_args")
                         async def check_order(ctx: RunContext, order_id: str) -> str: ...

                     Arguments are compared *after* validation, so a parameter the
                     LLM omitted on one call and passed explicitly on the next still
                     reads as the same call, and ``1`` matches ``1.0`` for a ``float``
                     parameter. Raw function tools (including MCP tools) have no
                     per-parameter schema, so their arguments are compared as sent.

                     Comparison fails open — an unrepresentable argument or a
                     validation error leaves the call treated as sent rather than
                     blocking it — so don't rely on this as an exactly-once
                     guarantee; put that in the tool body.
"""


@dataclass
class FunctionToolInfo:
    name: str
    description: str | None
    flags: ToolFlag
    on_duplicate: DuplicateMode = "allow"
    duplicate_scope: DuplicateScope = "name"
    requires_approval: bool = False


ReadBack = Callable[..., "str | InputRequired | Awaitable[str | InputRequired]"]
"""``requires_approval=`` as a function: the read-back the person approves. It takes the tool's
own parameters, ``self`` and the context included, and is called with the call's arguments.
An ``InputRequired`` instead asks for what is missing before there is anything to approve."""


class RawFunctionDescription(TypedDict):
    """
    Represents the raw function schema format used in LLM function calling APIs.

    This structure directly maps to OpenAI's function definition format as documented at:
    https://platform.openai.com/docs/guides/function-calling?api-mode=responses

    It is also compatible with other LLM providers that support raw JSON Schema-based
    function definitions.
    """

    name: str
    description: NotRequired[str | None]
    parameters: dict[str, object]


@dataclass
class RawFunctionToolInfo:
    name: str
    raw_schema: dict[str, Any]
    flags: ToolFlag
    on_duplicate: DuplicateMode = "allow"
    duplicate_scope: DuplicateScope = "name"


CONFIRM_DUPLICATE_PARAM = "lk_agents_confirm_duplicate"
"""Schema parameter added when ``@function_tool(on_duplicate='confirm')``."""

_CONFIRM_DUPLICATE_DESCRIPTION = (
    "Set this to True to confirm you want to run a duplicate. "
    "Only do this when user confirms the duplication is needed."
)


@dataclass
class _ApprovalScope:
    """Set by the executor around one call of a tool that requires approval: whether it may
    run, and whether it asked for approval instead."""

    granted: bool
    asked: bool = False


ANSWERS_KEY = "lk.answers"
"""``FunctionCall.extra`` of a call the framework made with the person's answer: the call of a
tool that relays answers (``relays_answer``) whose question it answers."""

HEARD_KEY = "lk.heard"
"""``FunctionCall.extra`` of such a call when the person's message cut the question off: what
they saw or heard of it, often nothing."""

ASKED_VERBATIM_KEY = "lk.verbatim"
"""``FunctionCall.extra`` of a call whose question reaches the person as written: kept so the
question can be asked again, joined with a later one of its task."""

APPROVAL_KEY = "lk.approval"
"""``FunctionCall.extra`` key on a call of a tool that requires approval: ``asked`` when it
returned the read-back, ``granted`` when it ran on the person's answer."""


TASK_KEY = "lk.task"
"""``extra`` key on a message or a call that a run recorded in a lane: the task it belongs to. A
call's output belongs to its call's task."""


_approval_scope: ContextVar[_ApprovalScope | None] = ContextVar(
    "agents_approval_scope", default=None
)


ASK_IN_OWN_WORDS = (
    "Ask the person this, in your own words, as your whole reply, and add nothing else: {question}"
)
"""How the agent facing the person is told to ask a question that is not verbatim."""

SAY_AS_WRITTEN = (
    "Say this to the person word for word, as your whole reply, and add nothing: {text}"
)
"""How a model that cannot ``say()`` is told to speak a verbatim text."""

DELIVERED = "delivered to the person as written: {text}"
"""What the model reads as the output of a verbatim result it did not write."""


def delivered(result: ToolResult) -> ToolResult:
    """A verbatim result as the model reads it once the person got it: that it was said, then
    what the result carries for the model alone (``_more``, from a delegate)."""
    text = DELIVERED.format(text=result.output)
    more = getattr(result, "_more", None)
    return ToolResult(f"{text}\n{more}" if more else text, reply_required=result.reply_required)


APPROVAL_ANSWERED = (
    "not run yet: it read back {question!r} and the person answered in the message that "
    "follows. If they agreed, call {name} again with exactly the same arguments to run it; if "
    "not, do not, and go on from what they said."
)
"""What the model reads as the output of a call that asked for approval, once answered: the
read-back alone, before the answer, reads like a call that ran."""

WAITING_PREFIX = "waiting for the person's answer to: "
"""The output of a call that asked the person: its question. The answer is the person's next
message, never this output."""


def relays_answer(tool: Any) -> bool:
    """Whether ``tool`` hands the person's answer to its question on to whoever asked, whose own
    model reads it, as a delegate does: the answer goes to the tool as said, with no model step.
    Private to the delegate; any other question's answer goes to the task's model."""
    return bool(getattr(tool, "_relays_answer", False))


def joint_question(questions: Sequence[InputRequired]) -> InputRequired:
    """The questions of one step as the one question its task pauses on: in order, each once,
    and verbatim only when every part is."""
    parts = list(dict.fromkeys(q.question for q in questions))
    return InputRequired("\n\n".join(parts), verbatim=all(q.verbatim for q in questions))


def _tasks_of(items: Sequence[Any]) -> dict[str, str | None]:
    """Each item's task by item id, ``TASK_KEY``; an output takes its call's."""
    from .chat_context import FunctionCall, FunctionCallOutput

    calls: dict[str, str | None] = {}
    tasks: dict[str, str | None] = {}
    for item in items:
        if isinstance(item, FunctionCallOutput):
            tasks[item.id] = calls.get(item.call_id)
            continue
        task = getattr(item, "extra", {}).get(TASK_KEY)
        if isinstance(item, FunctionCall):
            calls[item.call_id] = task
        tasks[item.id] = task
    return tasks


def _answered_read_backs(items: Sequence[Any]) -> set[str]:
    """The call ids of the read-backs the person answered: ``asked`` calls with a user message
    of the same task after them."""
    from .chat_context import ChatMessage, FunctionCall

    tasks = _tasks_of(items)
    answered: set[str] = set()
    waiting: dict[str | None, list[str]] = {}
    for item in items:
        if isinstance(item, FunctionCall) and item.extra.get(APPROVAL_KEY) == "asked":
            waiting.setdefault(tasks[item.id], []).append(item.call_id)
        elif isinstance(item, ChatMessage) and item.role == "user":
            answered.update(waiting.pop(tasks[item.id], []))
    return answered


def as_model_reads(items: Sequence[Any]) -> list[Any]:
    """The history as the model reads it: an answered read-back's output says the call has not
    run, and how to run it. What is stored stays the question."""
    from .chat_context import FunctionCall, FunctionCallOutput

    answered = _answered_read_backs(items)
    names = {i.call_id: i.name for i in items if isinstance(i, FunctionCall)}
    out: list[Any] = []
    for item in items:
        if isinstance(item, FunctionCallOutput) and item.call_id in answered:
            question = str(item.output).removeprefix(WAITING_PREFIX)
            text = APPROVAL_ANSWERED.format(question=question, name=names[item.call_id])
            item = item.model_copy(update={"output": text})
        out.append(item)
    return out


def approval_granted(items: Sequence[Any], call: Any) -> bool:
    """Whether ``call`` repeats the latest read-back of its tool in its task, with the same
    arguments, after the person answered it in that task, and no earlier repeat used that answer.

    A reading of the history by the calls' ``APPROVAL_KEY`` and ``TASK_KEY`` marks and the
    person's messages: a call that asked something else, such as a missing detail, is neither a
    read-back nor a run."""
    from .chat_context import FunctionCall

    def key(c: FunctionCall) -> str:
        try:
            return json.dumps(json.loads(c.arguments or "{}"), sort_keys=True)
        except ValueError:
            return c.arguments or ""

    answered = _answered_read_backs(items)
    task = call.extra.get(TASK_KEY)
    asked: str | None = None  # the arguments of the latest answered, unused read-back
    for item in items:
        if not isinstance(item, FunctionCall) or item.name != call.name:
            continue
        if item.extra.get(TASK_KEY) != task:
            continue
        if item.call_id == call.call_id:
            break
        mark = item.extra.get(APPROVAL_KEY)
        if mark == "granted" and key(item) == asked:
            asked = None  # this repeat ran on the answer
        elif mark == "asked":
            asked = key(item) if item.call_id in answered else None
    return asked is not None and key(call) == asked


@contextmanager
def approval_scope(call: Any, granted: bool) -> Iterator[None]:
    """Around one call of a tool that requires approval: whether it is the approved repeat of
    a read-back the person answered. Marks the call, for the next one to read."""
    scope = _ApprovalScope(granted=granted)
    token = _approval_scope.set(scope)
    try:
        yield
    finally:
        _approval_scope.reset(token)
        if scope.granted:
            call.extra[APPROVAL_KEY] = "granted"
        elif scope.asked:
            call.extra[APPROVAL_KEY] = "asked"


_InfoT = TypeVar("_InfoT", FunctionToolInfo, RawFunctionToolInfo)
_P = ParamSpec("_P")
_R = TypeVar("_R", bound=Awaitable[Any])


class _BaseFunctionTool(Tool, Generic[_InfoT, _P, _R]):
    """Base class for function tool wrappers with descriptor support."""

    def __init__(self, func: Callable[_P, _R], info: _InfoT, instance: Any = None) -> None:
        functools.update_wrapper(self, func)
        self._func = func
        self._info: _InfoT = info
        self._instance = instance

    @property
    def id(self) -> str:
        return self._info.name

    @property
    def info(self) -> _InfoT:
        return self._info

    def __get__(self, obj: Any, objtype: type | None = None) -> Self:
        if obj is None:
            return self

        # bind the tool to an instance
        bound_tool = self.__class__(self._func, self._info, instance=obj)
        sig = inspect.signature(self._func)
        # skip the instance parameter (e.g. usually the 'self')
        params = list(sig.parameters.values())[1:]
        bound_tool.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
        return bound_tool

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        if self._instance is not None:
            return self._func(self._instance, *args, **kwargs)
        return self._func(*args, **kwargs)


class FunctionTool(_BaseFunctionTool[FunctionToolInfo, _P, _R]):
    """Wrapper for a function decorated with @function_tool"""

    def __init__(
        self, func: Callable[_P, _R], info: FunctionToolInfo, instance: Any = None
    ) -> None:
        super().__init__(func, info, instance)
        setattr(self, "__livekit_tool_info", self._info)


class RawFunctionTool(_BaseFunctionTool[RawFunctionToolInfo, _P, _R]):
    """Wrapper for a function decorated with @function_tool(raw_schema=...)"""

    def __init__(
        self, func: Callable[_P, _R], info: RawFunctionToolInfo, instance: Any = None
    ) -> None:
        super().__init__(func, info, instance)
        setattr(self, "__livekit_raw_tool_info", self._info)


@overload
def function_tool(
    f: Callable[_P, _R],
    *,
    raw_schema: RawFunctionDescription | dict[str, Any],
    flags: ToolFlag = ToolFlag.NONE,
    on_duplicate: DuplicateMode = "allow",
    duplicate_scope: DuplicateScope = "name",
) -> RawFunctionTool[_P, _R]: ...


@overload
def function_tool(
    f: None = None,
    *,
    raw_schema: RawFunctionDescription | dict[str, Any],
    flags: ToolFlag = ToolFlag.NONE,
    on_duplicate: DuplicateMode = "allow",
    duplicate_scope: DuplicateScope = "name",
) -> Callable[[Callable[_P, _R]], RawFunctionTool[_P, _R]]: ...


@overload
def function_tool(
    f: Callable[_P, _R],
    *,
    name: str | None = None,
    description: str | None = None,
    flags: ToolFlag = ToolFlag.NONE,
    on_duplicate: DuplicateMode = "allow",
    duplicate_scope: DuplicateScope = "name",
    requires_approval: bool | ReadBack = False,
) -> FunctionTool[_P, _R]: ...


@overload
def function_tool(
    f: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    flags: ToolFlag = ToolFlag.NONE,
    on_duplicate: DuplicateMode = "allow",
    duplicate_scope: DuplicateScope = "name",
    requires_approval: bool | ReadBack = False,
) -> Callable[[Callable[_P, _R]], FunctionTool[_P, _R]]: ...


def function_tool(
    f: Callable[_P, _R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    raw_schema: RawFunctionDescription | dict[str, Any] | None = None,
    flags: ToolFlag = ToolFlag.NONE,
    on_duplicate: DuplicateMode = "allow",
    duplicate_scope: DuplicateScope = "name",
    requires_approval: bool | ReadBack = False,
) -> (
    FunctionTool[_P, _R]
    | RawFunctionTool[_P, _R]
    | Callable[[Callable[_P, _R]], FunctionTool[_P, _R] | RawFunctionTool[_P, _R]]
):
    """``requires_approval`` is plan 7.4's approval, held by the framework: a call returns
    ``InputRequired`` with a read-back instead of running, verbatim. The person's answer is
    their next message, and the body runs only for a repeat of the same call, same arguments,
    made after that answer. ``True`` builds the read-back from the description and
    the arguments; a ``ReadBack`` function writes it. Typed tools only."""

    def deco_raw(
        func: Callable[_P, _R],
    ) -> RawFunctionTool[_P, _R]:
        assert raw_schema is not None

        if requires_approval:
            raise ValueError("requires_approval needs a typed tool, not a raw_schema")
        if not raw_schema.get("name"):
            raise ValueError("raw function name cannot be empty")

        if "parameters" not in raw_schema:
            # support empty parameters
            raise ValueError("raw function description must contain a parameters key")

        schema = {**raw_schema}
        if on_duplicate == "confirm":
            schema["parameters"] = _inject_confirm_duplicate(schema["parameters"])

        info = RawFunctionToolInfo(
            name=raw_schema["name"],
            raw_schema=schema,
            flags=flags,
            on_duplicate=on_duplicate,
            duplicate_scope=duplicate_scope,
        )
        return RawFunctionTool(func, info)

    def deco_func(func: Callable[_P, _R]) -> FunctionTool[_P, _R]:
        from docstring_parser import parse_from_object

        wrapped: Callable[..., Any] = func
        if on_duplicate == "confirm":
            wrapped = _wrap_with_confirm_duplicate(func)

        docstring = parse_from_object(func)
        info = FunctionToolInfo(
            name=name or func.__name__,
            description=description or docstring.description,
            flags=flags,
            on_duplicate=on_duplicate,
            duplicate_scope=duplicate_scope,
            requires_approval=bool(requires_approval),
        )
        if requires_approval:
            read_back = requires_approval if callable(requires_approval) else None
            wrapped = _wrap_with_approval(wrapped, info.description or info.name, read_back)
        return FunctionTool(wrapped, info)

    if f is not None:
        return deco_raw(f) if raw_schema is not None else deco_func(f)
    return deco_raw if raw_schema is not None else deco_func


def _wrap_with_confirm_duplicate(func: Callable[..., Any]) -> Callable[..., Any]:
    """Extend ``func``'s signature with a CONFIRM_DUPLICATE_PARAM kwarg, stripped
    by the wrapper before delegating so direct calls with the original args still work."""
    try:
        resolved = get_type_hints(func, include_extras=True)
    except Exception:
        resolved = dict(getattr(func, "__annotations__", {}))

    annotation = Annotated[
        bool | None, Field(default=False, description=_CONFIRM_DUPLICATE_DESCRIPTION)
    ]
    new_annotations = {**resolved, CONFIRM_DUPLICATE_PARAM: annotation}

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop(CONFIRM_DUPLICATE_PARAM, None)
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    sig = inspect.signature(func)
    extra = inspect.Parameter(
        CONFIRM_DUPLICATE_PARAM,
        inspect.Parameter.KEYWORD_ONLY,
        default=False,
        annotation=annotation,
    )
    wrapper.__signature__ = sig.replace(parameters=[*sig.parameters.values(), extra])  # type: ignore[attr-defined]
    # set both for PEP 649: __annotations__ for 3.10-3.13, __annotate__ for 3.14.
    # __annotate__ must come last — assigning __annotations__ nulls it on 3.14.
    wrapper.__annotations__ = new_annotations
    wrapper.__annotate__ = lambda _format=1: dict(new_annotations)  # type: ignore[attr-defined]
    return wrapper


def _wrap_with_approval(
    func: Callable[..., Any], description: str, read_back: ReadBack | None
) -> Callable[..., Any]:
    """A call runs only inside an approval scope that grants it; any other returns the
    read-back as a verbatim ``InputRequired``. The schema is the tool's own: approval is not
    the model's to give."""
    from .utils import is_context_type

    try:
        resolved = get_type_hints(func, include_extras=True)
    except Exception:
        resolved = dict(getattr(func, "__annotations__", {}))

    sig = inspect.signature(func)
    # the description's first sentence is what the person is asked to confirm
    plain = description.strip().split(". ", 1)[0].splitlines()[0].rstrip(".") or "this"
    # the context, and a method's ``self``, are not the person's to confirm
    hidden = {k for k, t in resolved.items() if is_context_type(t, allow_subclasses=True)}
    hidden |= {k for k in sig.parameters if k in ("self", "cls")}

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        scope = _approval_scope.get()
        if scope is None or not scope.granted:
            if read_back is not None:
                question = read_back(*args, **kwargs)
                if inspect.isawaitable(question):
                    question = await question
                if isinstance(question, InputRequired):
                    return question  # something is missing before there is anything to approve
            else:
                bound = sig.bind_partial(*args, **kwargs)
                arguments = {k: v for k, v in bound.arguments.items() if k not in hidden}
                shown = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
                question = (
                    f"Please confirm: {plain} ({shown}). Yes or no?"
                    if shown
                    else f"Please confirm: {plain}. Yes or no?"
                )
            if scope is not None:
                scope.asked = True
            return InputRequired(str(question), verbatim=True)
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return wrapper


def _inject_confirm_duplicate(parameters: dict[str, Any]) -> dict[str, Any]:
    """Add CONFIRM_DUPLICATE_PARAM to a raw JSON-schema (strict-mode conformant)."""
    params = {**parameters}
    properties = {**params.get("properties", {})}
    properties[CONFIRM_DUPLICATE_PARAM] = {
        "type": ["boolean", "null"],
        "description": _CONFIRM_DUPLICATE_DESCRIPTION,
    }
    params["properties"] = properties
    required = list(params.get("required", []))
    if CONFIRM_DUPLICATE_PARAM not in required:
        required.append(CONFIRM_DUPLICATE_PARAM)
    params["required"] = required
    return params


def is_function_tool(f: Any) -> TypeGuard[FunctionTool]:
    # TODO(long): for backward compatibility, deprecate in future versions?
    return isinstance(f, FunctionTool)


def get_function_info(f: FunctionTool) -> FunctionToolInfo:
    return f.info


def is_raw_function_tool(f: Any) -> TypeGuard[RawFunctionTool]:
    return isinstance(f, RawFunctionTool)


def get_raw_function_info(f: RawFunctionTool) -> RawFunctionToolInfo:
    return f.info


def _resolve_wrapped_tool(tool: Any) -> FunctionTool | RawFunctionTool | None:
    """Convert a wrapped tool to a FunctionTool or RawFunctionTool with a warning."""
    if not callable(tool):
        return None

    if isinstance(tool, (FunctionTool, RawFunctionTool)):
        return tool

    resolved_tool: FunctionTool | RawFunctionTool | None = None
    if (
        hasattr(tool, "__wrapped__")  # automatically added by functools.wraps
        and isinstance(tool.__wrapped__, (FunctionTool, RawFunctionTool))
    ):
        wrapped = tool.__wrapped__
        resolved_tool = wrapped.__class__(tool, wrapped.info)  # type: ignore

    elif (info := getattr(tool, "__livekit_tool_info", None)) and isinstance(
        info, FunctionToolInfo
    ):
        resolved_tool = FunctionTool(tool, info)

    elif (info := getattr(tool, "__livekit_raw_tool_info", None)) and isinstance(
        info, RawFunctionToolInfo
    ):
        resolved_tool = RawFunctionTool(tool, info)

    if resolved_tool:
        tool_name = resolved_tool.info.name
        logger.warning(
            f"function tool {tool_name} is wrapped, this may cause unexpected behavior and not be supported in future versions, "
            "please wrap the original function before converting to a function tool.",
            extra={
                "function_tool": tool_name,
            },
        )

    return resolved_tool


def find_function_tools(cls_or_obj: Any) -> list[FunctionTool | RawFunctionTool]:
    """The ``@function_tool`` methods of a class, bound to ``cls_or_obj`` when it is an instance.

    Candidates are found on the class, so an instance property that raises outside a binding
    (``agent.session``, ``agent.chat_ctx``) is never evaluated.
    """
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
    methods: list[FunctionTool | RawFunctionTool] = []
    for name, member in inspect.getmembers(cls):
        if not (isinstance(member, (FunctionTool, RawFunctionTool)) or _is_wrapped_tool(member)):
            continue
        bound = getattr(cls_or_obj, name)
        if isinstance(bound, (FunctionTool, RawFunctionTool)):
            methods.append(bound)
        elif normalized_tool := _resolve_wrapped_tool(bound):
            methods.append(normalized_tool)

    return methods


def _is_wrapped_tool(member: Any) -> bool:
    if not callable(member):
        return False
    if hasattr(member, "__wrapped__") and isinstance(
        member.__wrapped__, (FunctionTool, RawFunctionTool)
    ):
        return True
    return isinstance(getattr(member, "__livekit_tool_info", None), FunctionToolInfo) or (
        isinstance(getattr(member, "__livekit_raw_tool_info", None), RawFunctionToolInfo)
    )


def get_fnc_tool_names(tools: Sequence[Tool | Toolset]) -> list[str]:
    """Get names of all function and raw function tools in the list, unwrapping tool sets."""
    names = []
    for tool in tools:
        if isinstance(tool, (FunctionTool, RawFunctionTool)):
            names.append(tool.info.name)
        elif isinstance(tool, Toolset):
            names.extend(get_fnc_tool_names(tool.tools))

    return names


class ToolContext:
    """Stateless container for a set of AI functions"""

    def __init__(self, tools: Sequence[Tool | Toolset]) -> None:
        self.update_tools(tools)

    @classmethod
    def empty(cls) -> ToolContext:
        return cls([])

    @property
    def function_tools(self) -> dict[str, FunctionTool | RawFunctionTool]:
        """A copy of all function tools in the tool context, including those in tool sets."""
        return self._fnc_tools_map.copy()

    @property
    def provider_tools(self) -> list[ProviderTool]:
        """A copy of all provider tools in the tool context, including those in tool sets."""
        return self._provider_tools

    @property
    def toolsets(self) -> list[Toolset]:
        """A copy of all tool sets in the tool context."""
        return self._tool_sets

    def flatten(self) -> list[Tool]:
        """Flatten the tool context to a list of tools."""
        tools: list[Tool] = []
        tools.extend(list(self._fnc_tools_map.values()))
        tools.extend(self._provider_tools)
        return tools

    def get_function_tool(self, name: str) -> FunctionTool | RawFunctionTool | None:
        return self._fnc_tools_map.get(name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolContext):
            return False

        if self._fnc_tools_map.keys() != other._fnc_tools_map.keys():
            return False

        for name in self._fnc_tools_map:
            if self._fnc_tools_map[name] is not other._fnc_tools_map[name]:
                return False

        if len(self._provider_tools) != len(other._provider_tools):
            return False

        self_provider_ids = {id(tool) for tool in self._provider_tools}
        other_provider_ids = {id(tool) for tool in other._provider_tools}
        if self_provider_ids != other_provider_ids:
            return False

        self_tool_set_ids = {id(tool_set) for tool_set in self._tool_sets}
        other_tool_set_ids = {id(tool_set) for tool_set in other._tool_sets}
        if self_tool_set_ids != other_tool_set_ids:
            return False

        return True

    def update_tools(self, tools: Sequence[Tool | Toolset]) -> None:
        self._update_tools(tools)

    def _update_tools(
        self, tools: Sequence[Tool | Toolset], *, exclude: Sequence[Tool] = ()
    ) -> None:
        self._tools = list(tools)
        self._fnc_tools_map: dict[str, FunctionTool | RawFunctionTool] = {}
        self._provider_tools: list[ProviderTool] = []
        self._tool_sets: list[Toolset] = []

        def add_tool(tool: Tool | Toolset) -> None:
            if any(tool is e for e in exclude):
                return

            if isinstance(tool, ProviderTool):
                self._provider_tools.append(tool)

            elif isinstance(tool, (FunctionTool, RawFunctionTool)):
                existing = self._fnc_tools_map.get(tool.info.name)
                if existing is not None:
                    if existing is not tool:
                        raise ValueError(f"duplicate function name: {tool.info.name}")
                    return  # same instance, skip
                self._fnc_tools_map[tool.info.name] = tool

            elif isinstance(tool, Toolset):
                for t in tool.tools:
                    add_tool(t)
                self._tool_sets.append(tool)

            elif normalized_tool := _resolve_wrapped_tool(tool):
                add_tool(normalized_tool)

            elif callable(tool):
                raise ValueError(
                    "Expected an instance of FunctionTool or RawFunctionTool, got a callable object. "
                    "If it's a wrapped tool, please consider wrapping the original function before converting to a function tool."
                )

            else:
                raise ValueError(f"unknown tool type: {type(tool)}")

        for tool in itertools.chain(tools, find_function_tools(self)):
            add_tool(tool)

    def _sync_flattened(self, tools: Sequence[Tool]) -> None:
        """Apply in-place edits of a ``flatten()`` list, preserving Toolset grouping.

        Added tools become top-level entries; removed tools are dropped from the
        flat lookup. A removed Toolset member stays in its toolset (membership and
        lifecycle remain the toolset's) — it just stops being callable.
        """
        current = self.flatten()
        current_ids = {id(t) for t in current}
        tool_ids = {id(t) for t in tools}
        if current_ids == tool_ids:
            return

        added = [t for t in tools if id(t) not in current_ids]
        removed_ids = current_ids - tool_ids
        removed = [c for c in current if id(c) in removed_ids]

        structured = [t for t in self._tools if not any(t is r for r in removed)]
        self._update_tools([*structured, *added], exclude=removed)

    def _exclude(self, tools: Sequence[Tool]) -> None:
        """Hide ``tools`` from the callable set while keeping their toolsets intact."""
        if not tools:
            return
        kept = [t for t in self.flatten() if not any(t is e for e in tools)]
        self._sync_flattened(kept)

    def copy(self) -> ToolContext:
        return ToolContext(self._tools.copy())

    @overload
    def parse_function_tools(
        self, format: Literal["openai"], *, strict: bool = True
    ) -> list[dict[str, Any]]: ...

    @overload
    def parse_function_tools(
        self,
        format: Literal["openai.responses"],
        *,
        strict: bool = True,
        provider_tool_type: type[ProviderTool] | None = None,
    ) -> list[dict[str, Any]]: ...

    @overload
    def parse_function_tools(
        self,
        format: Literal["google"],
        *,
        tool_behavior: _provider_format.google.TOOL_BEHAVIOR | None = None,
        use_parameters_json_schema: bool = True,
    ) -> list[dict[str, Any]]: ...

    @overload
    def parse_function_tools(self, format: Literal["aws"]) -> list[dict[str, Any]]: ...

    @overload
    def parse_function_tools(
        self, format: Literal["anthropic"], *, strict: bool = True
    ) -> list[dict[str, Any]]: ...

    def parse_function_tools(
        self,
        format: Literal["openai", "google", "aws", "anthropic"] | str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Parse the function tools to a provider-specific schema."""
        if format == "openai":
            return _provider_format.openai.to_fnc_ctx(self, **kwargs)
        elif format == "openai.responses":
            return _provider_format.openai.to_responses_fnc_ctx(self, **kwargs)
        elif format == "google":
            return _provider_format.google.to_fnc_ctx(self, **kwargs)
        elif format == "anthropic":
            return _provider_format.anthropic.to_fnc_ctx(self, **kwargs)
        elif format == "aws":
            return _provider_format.aws.to_fnc_ctx(self, **kwargs)

        raise ValueError(f"Unsupported provider format: {format}")
