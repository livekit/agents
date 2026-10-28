from __future__ import annotations

import asyncio
import functools
import inspect
import itertools
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
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

from .. import _core
from . import _provider_format

if TYPE_CHECKING:
    from ..voice.events import RunContext
    from ..voice.tool_executor import ToolHandlingOptions


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
        # the runs and calls using this shared toolset: the first sets it up, the last closes it
        self._users = 0
        self._lifecycle = asyncio.Lock()

    @property
    def id(self) -> str:
        return self._id

    @property
    def tools(self) -> Sequence[Tool | Toolset]:
        return self._tools

    async def setup(self) -> Self:
        """Connect the toolset and its nested toolsets, when a run or a call starts using it.

        A failed setup closes the nested toolsets that did set up before raising.
        """
        toolsets = [tool for tool in self.tools if isinstance(tool, Toolset)]
        results = await asyncio.gather(*(t.setup() for t in toolsets), return_exceptions=True)
        failed = [r for r in results if isinstance(r, BaseException)]
        if failed:
            await asyncio.gather(
                *(
                    t.aclose()
                    for t, r in zip(toolsets, results)
                    if not isinstance(r, BaseException)
                ),
                return_exceptions=True,
            )
            raise failed[0]
        return self

    async def aclose(self) -> None:
        """Release what ``setup()`` acquired, when the last run or call using it ends.

        A toolset on ``Agent(tools=...)`` is closed when the agent's run or activation ends;
        one on ``VoiceSession(tools=...)`` when the session closes.
        """
        toolsets = [tool for tool in self._tools if isinstance(tool, Toolset)]
        if toolsets:
            await asyncio.gather(*(toolset.aclose() for toolset in toolsets))

    async def _acquire(self) -> None:
        """One more run or call uses this toolset; the first sets it up."""
        async with self._lifecycle:
            if self._users == 0:
                await self.setup()
            self._users += 1

    async def _release(self) -> None:
        """A run or call stopped using this toolset; the last closes it, even when cancelled."""

        async def release() -> None:
            async with self._lifecycle:
                self._users -= 1
                if self._users == 0:
                    await self.aclose()

        await asyncio.shield(release())


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
    """The turn or task waits on the person for an answer to ``question``.

    The agent facing the person asks it in its own words, or exactly as written with
    ``verbatim=True``. From a conversation tool the next message is written as this call's
    output. ``tool`` names the tool that takes the answer and ``timeout`` schedules
    ``on_timeout``.
    """

    question: str
    tool: FunctionTool | RawFunctionTool | None = None
    timeout: timedelta | None = None
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
    needs_approval: bool = False
    read_back: ReadBack | None = None
    """``needs_approval=`` as a function: the read-back the person approves."""


ReadBack = Callable[..., "str | InputRequired | Awaitable[str | InputRequired]"]


"""``needs_approval=`` as a function: the read-back the person approves. It takes the tool's
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


ASK_IN_OWN_WORDS = (
    "Ask the person this, in your own words, as your whole reply, and add nothing else: {question}"
)


"""How the agent facing the person is told to ask a question that is not verbatim."""


DELIVERED = "delivered to the person as written: {text}"


"""What the model reads as the output of a verbatim result it did not write."""


WAITING_PREFIX = "waiting for the person's answer to: "


"""How a call's output reads while it waits on the person; the answer replaces it."""


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


class RawFunctionTool(_BaseFunctionTool[RawFunctionToolInfo, _P, _R]):
    """Wrapper for a function decorated with @function_tool(raw_schema=...)"""


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
    needs_approval: bool | ReadBack = False,
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
    needs_approval: bool | ReadBack = False,
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
    needs_approval: bool | ReadBack = False,
) -> (
    FunctionTool[_P, _R]
    | RawFunctionTool[_P, _R]
    | Callable[[Callable[_P, _R]], FunctionTool[_P, _R] | RawFunctionTool[_P, _R]]
):
    """``needs_approval``: the framework asks the person to confirm each call with a read-back
    and runs the body only on a repeat of the same validated call after the answer (tools.md
    "Approval"). ``True`` reads back the description and the arguments; a ``ReadBack`` function
    writes it. Typed tools only."""

    def deco_raw(
        func: Callable[_P, _R],
    ) -> RawFunctionTool[_P, _R]:
        assert raw_schema is not None

        if needs_approval:
            raise ValueError("needs_approval needs a typed tool, not a raw_schema")
        if not raw_schema.get("name"):
            raise ValueError("raw function name cannot be empty")

        if "parameters" not in raw_schema:
            # support empty parameters
            raise ValueError("raw function description must contain a parameters key")

        schema = {**raw_schema}
        if on_duplicate == "confirm":
            schema["parameters"] = _core.inject_confirm_duplicate(schema["parameters"])

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
            needs_approval=bool(needs_approval),
            read_back=needs_approval if callable(needs_approval) else None,
        )
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
        bool | None, Field(default=False, description=_core.confirm_duplicate_description())
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


def is_function_tool(f: Any) -> TypeGuard[FunctionTool]:
    return isinstance(f, FunctionTool)


def is_raw_function_tool(f: Any) -> TypeGuard[RawFunctionTool]:
    return isinstance(f, RawFunctionTool)


def find_function_tools(cls_or_obj: Any) -> list[FunctionTool | RawFunctionTool]:
    """The ``@function_tool`` methods of a class, bound to ``cls_or_obj`` when it is an instance.

    Candidates are found on the class, so an instance property that raises outside a binding
    (``agent.session``, ``agent.chat_ctx``) is never evaluated.
    """
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
    return [
        getattr(cls_or_obj, name)
        for name, member in inspect.getmembers(cls)
        if isinstance(member, (FunctionTool, RawFunctionTool))
    ]


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
        """The same tools: the same instances under the same names."""
        if not isinstance(other, ToolContext):
            return False
        mine, theirs = self._fnc_tools_map, other._fnc_tools_map
        return (
            mine.keys() == theirs.keys()
            and all(tool is theirs[name] for name, tool in mine.items())
            and len(self._provider_tools) == len(other._provider_tools)
            and {id(t) for t in self._provider_tools} == {id(t) for t in other._provider_tools}
            and {id(t) for t in self._tool_sets} == {id(t) for t in other._tool_sets}
        )

    def update_tools(self, tools: Sequence[Tool | Toolset]) -> None:
        self._tools = list(tools)
        self._fnc_tools_map: dict[str, FunctionTool | RawFunctionTool] = {}
        self._provider_tools: list[ProviderTool] = []
        self._tool_sets: list[Toolset] = []

        def add_tool(tool: Tool | Toolset) -> None:
            if isinstance(tool, ProviderTool):
                self._provider_tools.append(tool)
            elif isinstance(tool, (FunctionTool, RawFunctionTool)):
                # the same instance twice is one tool
                if self._fnc_tools_map.setdefault(tool.info.name, tool) is not tool:
                    raise ValueError(f"duplicate function name: {tool.info.name}")
            elif isinstance(tool, Toolset):
                for t in tool.tools:
                    add_tool(t)
                self._tool_sets.append(tool)
            else:
                raise ValueError(
                    f"unknown tool type: {type(tool)}; a wrapped function tool is not one: wrap "
                    "the original function before converting it to a function tool"
                )

        for tool in itertools.chain(tools, find_function_tools(self)):
            add_tool(tool)

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
        tool_behavior: _provider_format.TOOL_BEHAVIOR | None = None,
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
        return _provider_format.to_fnc_ctx(self, format, **kwargs)


class AsyncToolset(Toolset):
    """Session-scoped toolset whose tools survive agent handoff.

    Background updates from tools in this toolset are delivered to whichever agent
    is current at delivery time, so a ``ctx.update()`` started under agent A still
    completes after a handoff to agent B. Tools placed on ``Agent(tools=...)``
    instead use the activity-scoped executor and are cancelled/awaited on handoff.

    Example::

        @function_tool(on_duplicate="confirm", flags=ToolFlag.CANCELLABLE)
        async def book_flight(ctx, origin: str, destination: str) -> dict:
            await ctx.update(f"Looking up flights {origin} → {destination}...")
            flights = await search(origin, destination)
            await ctx.update(f"Found {len(flights)}, picking the best...")
            return await book_best(flights)

        session = VoiceSession(tools=[AsyncToolset(id="booking", tools=[book_flight])])
    """

    def __init__(
        self,
        *,
        id: str,
        tools: Sequence[Tool] | None = None,
        tool_handling: ToolHandlingOptions | None = None,
    ) -> None:
        super().__init__(id=id, tools=tools)
        self._async_tool_options_override = (
            tool_handling.get("async_options") if tool_handling is not None else None
        )
