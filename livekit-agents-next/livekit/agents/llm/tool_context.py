from __future__ import annotations

import asyncio
import functools
import inspect
import itertools
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Self,
    TypeGuard,
    get_type_hints,
    overload,
)

from pydantic import Field, JsonValue
from typing_extensions import ParamSpec, TypeVar

from livekit.agents._generated.function_tools import (
    FunctionToolInfo as FunctionToolInfo,
    InputRequired as InputRequired,
    RawFunctionToolInfo as RawFunctionToolInfo,
)
from livekit.agents._generated.tool_outcomes import (
    Failed as Failed,
    Rejected as Rejected,
    ToolResult as ToolResult,
    Working as Working,
)

from .. import _core
from .._generated.tool_context import (
    CONFIRM_DUPLICATE_DESCRIPTION,
    RawFunctionDescription as RawFunctionDescription,
)
from .._generated.tool_errors import (
    ModelRetry as ModelRetry,
    StopResponse as StopResponse,
    ToolError as ToolError,
)
from .._generated.tool_flags import ToolFlag as ToolFlag
from .._generated.tool_options import Function as Function, NamedToolChoice as NamedToolChoice
from . import _provider_format

if TYPE_CHECKING:
    from ..voice.events import RunContext
    from ..voice.tool_executor import ToolHandlingOptions


class Tool(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...


class ProviderTool(Tool):
    provider: ClassVar[str | None] = None

    def __init__(self, *, id: str) -> None:
        self._id = id

    @property
    def id(self) -> str:
        return self._id

    def to_dict(self) -> dict[str, JsonValue] | None:
        """Return a native wire declaration, when this tool supports a native transport."""
        return None


class Toolset:
    def __init__(self, *, id: str, tools: Sequence[Tool | Toolset] | None = None) -> None:
        self._id = id
        self._tools: Sequence[Tool | Toolset] = list(tools) if tools is not None else []
        self._tools.extend(find_function_tools(self))
        self._lifetime = _core._core_ToolsetLifetime()
        self._children: _core._core_ToolsetGroup | None = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def tools(self) -> Sequence[Tool | Toolset]:
        return self._tools

    def _toolset_dependencies(self) -> Sequence[Toolset]:
        """Resources acquired by setup, independently of the tools offered to the model."""
        return [tool for tool in self.tools if isinstance(tool, Toolset)]

    async def setup(self) -> Self:
        """Connect the toolset and its nested toolsets, when a run or a call starts using it.

        A failed setup closes the nested toolsets that did set up before raising.
        """
        self._children = _core._core_ToolsetGroup(self._toolset_dependencies())
        await self._children.start()
        return self

    async def aclose(self) -> None:
        """Release nested toolsets acquired by the default setup.

        The native lifetime calls this after the last run or activation releases its lease.
        """
        if self._children is not None:
            await self._children.aclose()

    def for_run(self, ctx: RunContext[Any]) -> Toolset:
        """The toolset one run or call uses: this one, unless its tools depend on the run, as an
        MCP server's per-user credentials do. A run resolves it once, before setup."""
        return self

    def filtered(self, predicate: Callable[[Tool], bool]) -> Toolset:
        """This toolset's tools for which ``predicate`` is true, nested toolsets filtered alike, as
        a toolset that sets up and closes with this one.

        Example::

            rewards.filtered(lambda tool: tool.info.name in {"balance", "redeem"})
        """
        return _FilteredToolset(self, predicate)

    def _core_lifetime(self) -> _core._core_ToolsetLifetime:
        return self._lifetime

    def _core_children(self) -> _core._core_ToolsetGroup | None:
        return self._children


# Used by ToolChoice


ToolChoice = NamedToolChoice | Literal["auto", "required", "none"]


Transition_T = TypeVar("Transition_T", covariant=True)


@dataclass(frozen=True)
class Completed(Generic[Transition_T]):
    """The task is done: ``value`` is its result, and its last message when it is text.
    ``verbatim`` makes that text reach the person exactly as written."""

    value: Transition_T
    verbatim: bool = field(default=False, kw_only=True)


Transition = Completed[object] | Failed | Rejected | InputRequired | Working


"""What a tool or a hook returns to change the state of the turn or the task."""


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


ReadBack = Callable[..., "str | InputRequired | Awaitable[str | InputRequired]"]


"""``needs_approval=`` as a function: the read-back the person approves. It takes the tool's
own parameters, ``self`` and the context included, and is called with the call's arguments.
An ``InputRequired`` instead asks for what is missing before there is anything to approve."""


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


_P = ParamSpec("_P", default=...)


_R = TypeVar("_R", default=object, covariant=True)


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

    def __get__(self, obj: object, objtype: type | None = None) -> Self:
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
    raw_schema: RawFunctionDescription | dict[str, JsonValue],
    flags: ToolFlag = ToolFlag.NONE,
    on_duplicate: DuplicateMode = "allow",
    duplicate_scope: DuplicateScope = "name",
    retries: int | None = None,
) -> RawFunctionTool[_P, _R]: ...


@overload
def function_tool(
    f: None = None,
    *,
    raw_schema: RawFunctionDescription | dict[str, JsonValue],
    flags: ToolFlag = ToolFlag.NONE,
    on_duplicate: DuplicateMode = "allow",
    duplicate_scope: DuplicateScope = "name",
    retries: int | None = None,
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
    retries: int | None = None,
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
    retries: int | None = None,
) -> Callable[[Callable[_P, _R]], FunctionTool[_P, _R]]: ...


def function_tool(
    f: Callable[..., object] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    raw_schema: RawFunctionDescription | dict[str, JsonValue] | None = None,
    flags: ToolFlag = ToolFlag.NONE,
    on_duplicate: DuplicateMode = "allow",
    duplicate_scope: DuplicateScope = "name",
    needs_approval: bool | ReadBack = False,
    retries: int | None = None,
) -> (
    FunctionTool[..., object]
    | RawFunctionTool[..., object]
    | Callable[[Callable[..., object]], FunctionTool[..., object] | RawFunctionTool[..., object]]
):
    """``needs_approval``: the framework asks the person to confirm each call with a read-back
    and runs the body only on a repeat of the same validated call after the answer (tools.md
    "Approval"). ``True`` reads back the description and the arguments; a ``ReadBack`` function
    writes it. Typed tools only.

    ``retries``: how many consecutive retries the model gets for this tool (invalid arguments,
    ``ModelRetry``) before the run fails; the agent's ``retries['tools']`` when ``None``."""

    if retries is not None and (isinstance(retries, bool) or retries < 0):
        raise ValueError(f"retries must be a non-negative integer, got {retries!r}")

    def deco_raw(
        func: Callable[_P, _R],
    ) -> RawFunctionTool[_P, _R]:
        assert raw_schema is not None

        if needs_approval:
            raise ValueError("needs_approval needs a typed tool, not a raw_schema")
        name = raw_schema.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("raw function name must be a non-empty string")

        if "parameters" not in raw_schema:
            # support empty parameters
            raise ValueError("raw function description must contain a parameters key")

        schema = {**raw_schema}
        if on_duplicate == "confirm":
            schema["parameters"] = _core.inject_confirm_duplicate(schema["parameters"])

        info = RawFunctionToolInfo(
            name=name,
            raw_schema=schema,
            flags=flags,
            on_duplicate=on_duplicate,
            duplicate_scope=duplicate_scope,
            retries=retries,
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
            retries=retries,
        )
        return FunctionTool(wrapped, info)

    if f is not None:
        return deco_raw(f) if raw_schema is not None else deco_func(f)
    return deco_raw if raw_schema is not None else deco_func


def _wrap_with_confirm_duplicate(func: Callable[..., object]) -> Callable[..., object]:
    """Extend ``func``'s signature with a CONFIRM_DUPLICATE_PARAM kwarg, stripped
    by the wrapper before delegating so direct calls with the original args still work."""
    try:
        resolved = get_type_hints(func, include_extras=True)
    except Exception:
        resolved = dict(getattr(func, "__annotations__", {}))

    annotation = Annotated[
        bool | None, Field(default=False, description=CONFIRM_DUPLICATE_DESCRIPTION)
    ]
    new_annotations = {**resolved, CONFIRM_DUPLICATE_PARAM: annotation}

    @functools.wraps(func)
    async def wrapper(*args: object, **kwargs: object) -> object:
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


def is_function_tool(f: object) -> TypeGuard[FunctionTool]:
    return isinstance(f, FunctionTool)


def is_raw_function_tool(f: object) -> TypeGuard[RawFunctionTool]:
    return isinstance(f, RawFunctionTool)


def find_function_tools(cls_or_obj: object) -> list[FunctionTool | RawFunctionTool]:
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
        fnc_tools_map: dict[str, FunctionTool | RawFunctionTool] = {}
        provider_tools: list[ProviderTool] = []
        tool_sets: list[Toolset] = []

        def add_tool(tool: Tool | Toolset) -> None:
            if isinstance(tool, ProviderTool):
                provider_tools.append(tool)
            elif isinstance(tool, (FunctionTool, RawFunctionTool)):
                # the same instance twice is one tool
                if fnc_tools_map.setdefault(tool.info.name, tool) is not tool:
                    raise ValueError(f"duplicate function name: {tool.info.name}")
            elif isinstance(tool, Toolset):
                for t in tool.tools:
                    add_tool(t)
                tool_sets.append(tool)
            else:
                raise ValueError(
                    f"unknown tool type: {type(tool)}; a wrapped function tool is not one: wrap "
                    "the original function before converting it to a function tool"
                )

        for tool in itertools.chain(tools, find_function_tools(self)):
            add_tool(tool)

        collisions = set(fnc_tools_map).intersection(tool.id for tool in provider_tools)
        if collisions:
            raise ValueError(
                "provider tool ID conflicts with function name: " + ", ".join(sorted(collisions))
            )
        self._tools = list(tools)
        self._fnc_tools_map = fnc_tools_map
        self._provider_tools = provider_tools
        self._tool_sets = tool_sets

    def copy(self) -> ToolContext:
        return ToolContext(self._tools.copy())

    @overload
    def parse_function_tools(
        self, format: Literal["openai"], *, strict: bool = True
    ) -> list[dict[str, object]]: ...

    @overload
    def parse_function_tools(
        self,
        format: Literal["openai.responses"],
        *,
        strict: bool = True,
        provider_tool_type: type[ProviderTool] | None = None,
    ) -> list[dict[str, object]]: ...

    @overload
    def parse_function_tools(
        self,
        format: Literal["google"],
        *,
        tool_behavior: _provider_format.TOOL_BEHAVIOR | None = None,
        use_parameters_json_schema: bool = True,
    ) -> list[dict[str, object]]: ...

    @overload
    def parse_function_tools(self, format: Literal["aws"]) -> list[dict[str, object]]: ...

    @overload
    def parse_function_tools(
        self, format: Literal["anthropic"], *, strict: bool = True
    ) -> list[dict[str, object]]: ...

    def parse_function_tools(
        self,
        format: Literal["openai", "google", "aws", "anthropic"] | str,
        *,
        strict: bool = True,
        provider_tool_type: type[ProviderTool] | None = None,
        tool_behavior: _provider_format.TOOL_BEHAVIOR | None = None,
        use_parameters_json_schema: bool = True,
    ) -> list[dict[str, object]]:
        """Parse the function tools to a provider-specific schema."""
        return _provider_format.to_fnc_ctx(
            self,
            format,
            strict=strict,
            provider_tool_type=provider_tool_type,
            tool_behavior=tool_behavior,
            use_parameters_json_schema=use_parameters_json_schema,
        )


class AsyncToolset(Toolset):
    """Session-scoped toolset whose tools survive agent handoff.

    Background updates from tools in this toolset are delivered to whichever agent
    is current at delivery time, so a ``ctx.update()`` started under agent A still
    completes after a handoff to agent B. Tools placed on ``Agent(tools=...)``
    instead use the activity-scoped executor and are cancelled/awaited on handoff.
    ``tool_handling`` overrides the agent's templates for this toolset's tools.

    Example::

        @function_tool(on_duplicate="confirm")
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
        self._tool_handling = tool_handling


class _FilteredToolset(AsyncToolset):
    """A toolset's tools that pass a predicate. It keeps the toolset's structure and
    ``tool_handling``, and leases the toolset itself."""

    def __init__(self, inner: Toolset, predicate: Callable[[Tool], bool]) -> None:
        handling = inner._tool_handling if isinstance(inner, AsyncToolset) else None
        super().__init__(id=inner.id, tool_handling=handling)
        self._inner = inner
        self._predicate = predicate
        self._views: dict[int, tuple[Toolset, _FilteredToolset]] = {}

    @property
    def tools(self) -> Sequence[Tool | Toolset]:
        kept: list[Tool | Toolset] = []
        for tool in self._inner.tools:
            if isinstance(tool, Toolset):
                cached = self._views.get(id(tool))
                if cached is None or cached[0] is not tool:
                    cached = self._views[id(tool)] = (tool, _FilteredToolset(tool, self._predicate))
                kept.append(cached[1])
            elif self._predicate(tool):
                kept.append(tool)
        return kept

    def _toolset_dependencies(self) -> Sequence[Toolset]:
        return [self._inner]

    def for_run(self, ctx: RunContext[Any]) -> Toolset:
        inner = self._inner.for_run(ctx)
        if inner is self._inner:
            return self
        cached = self._views.get(id(inner))
        if cached is None or cached[0] is not inner:
            cached = self._views[id(inner)] = (inner, _FilteredToolset(inner, self._predicate))
        return cached[1]


def tools_for_run(tools: Sequence[Tool | Toolset], ctx: RunContext[Any]) -> list[Tool | Toolset]:
    """``tools`` as the run or call of ``ctx`` uses them: each toolset's ``for_run``."""
    return [tool.for_run(ctx) if isinstance(tool, Toolset) else tool for tool in tools]
