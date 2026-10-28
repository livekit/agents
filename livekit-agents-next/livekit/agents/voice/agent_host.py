"""The application callbacks of an agent invocation, run for the core (``_core.AgentRunner``).

``lk-agents-core``'s ``AgentRun`` decides what runs, when, and what an outcome means
(continuation, approval, retries, phrasing); ``AgentHost`` only does host work: it binds and
validates tool arguments, runs tool bodies and classifies what they returned or raised, renders
the dynamic instructions, renders read-backs and validates output candidates. Each callback runs
in a copy of the context captured when ``run()`` was entered, so the invocation it serves is the
one ``_BindingContextVar`` holds there.

The ``dict`` shapes are the interim host records of ``lk_agents_abi::agent`` (until the records
family generates them).
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, to_jsonable_python

from ..llm import FunctionCall
from ..llm.tool_context import (
    CONFIRM_DUPLICATE_PARAM,
    FunctionTool,
    InputRequired,
    ModelRetry,
    RawFunctionTool,
    ToolError,
)
from ..llm.utils import (
    bind_function_arguments,
    function_arguments_to_pydantic_model,
    is_context_type,
    tool_outcome,
    validated_arguments,
)
from ..log import logger
from .agent import _BindingContextVar
from .events import RunContext

if TYPE_CHECKING:
    from .agent import Agent
    from .agent_run import _TextRun

__all__ = ["AgentHost", "ToolArgumentError"]


class ToolArgumentError(Exception):
    """A call's arguments did not bind or validate: the message is the model's feedback."""


@dataclass
class _Prepared:
    """A call's validated arguments, held by the runtime between preparation and the body."""

    tool: FunctionTool | RawFunctionTool
    ctx: RunContext[Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class AgentHost:
    """The host side of one ``Agent``'s invocations; shared by all of them."""

    def __init__(self, agent: Agent[Any, Any]) -> None:
        self._agent = agent
        self._output = OutputContract.of(agent)
        # the callback task of each operation in flight, for `cancel`
        self._running: dict[tuple[str, int], asyncio.Task[Any]] = {}

    async def render_instructions(self, invocation: dict[str, Any]) -> str:
        with self._operation(invocation):
            ctx = RunContext[Any](function_call=None, run=_current_run())
            return await self._agent._render_dynamic_instructions(ctx)

    async def prepare_tool(
        self, invocation: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Binds and validates the call's arguments in the call's scope; the prepared call and
        the identity approval binds to (the validated arguments as JSON, decision 15)."""
        with self._operation(invocation):
            run = _current_run()
            call = _function_call(invocation)
            tool = run.tools.get_function_tool(call.name)
            if tool is None:
                raise ToolArgumentError(f"unknown tool `{call.name}`")
            ctx = RunContext[Any](function_call=call, run=run)
            try:
                fields = validated_arguments(tool, arguments)
            except (ValidationError, ValueError, TypeError) as e:
                raise ToolArgumentError(f"Error parsing arguments for `{call.name}`: {e}") from e
            args, kwargs = bind_function_arguments(tool, fields, ctx)
            identified = {k: v for k, v in fields.items() if k != CONFIRM_DUPLICATE_PARAM}
            try:
                identity = {"kind": "json", "value": to_jsonable_python(identified)}
            except PydanticSerializationError:
                # no JSON for these arguments: a fresh fingerprint matches no earlier read-back,
                # so the call can never be approved by mistake
                identity = {"kind": "fingerprint", "value": uuid.uuid4().hex}
            return {"prepared": _Prepared(tool, ctx, args, kwargs), "identity": identity}

    async def read_back(self, invocation: dict[str, Any], prepared: _Prepared) -> dict[str, Any]:
        """The tool's read-back function on the call's arguments."""
        with self._operation(invocation):
            assert isinstance(prepared.tool, FunctionTool) and prepared.tool.info.read_back
            question = prepared.tool.info.read_back(*prepared.args, **prepared.kwargs)
            if inspect.isawaitable(question):
                question = await question
            if isinstance(question, InputRequired):
                return {
                    "kind": "input_required",
                    "question": question.question,
                    "verbatim": question.verbatim,
                }
            return {"kind": "question", "question": str(question)}

    async def run_tool(self, invocation: dict[str, Any], prepared: _Prepared) -> dict[str, Any]:
        """Runs the body, sync or async, and classifies what it returned or raised."""
        with self._operation(invocation):
            prepared.ctx._function_call = _function_call(invocation)
            try:
                output = prepared.tool(*prepared.args, **prepared.kwargs)
                if inspect.isawaitable(output):
                    output = await output
            except asyncio.CancelledError:
                return {"kind": "cancelled"}
            except Exception as e:
                outcome = tool_outcome(None, e)
                if outcome["kind"] == "internal":
                    logger.exception(
                        "exception occurred while executing a tool",
                        extra={"function": prepared.ctx.function_call.name},
                    )
                return outcome
            outcome = tool_outcome(output)
            if outcome["kind"] == "invalid":
                logger.error(
                    f"tool `{prepared.ctx.function_call.name}` returned an invalid output",
                    extra={"lk.pii.output": output},
                )
            return outcome

    async def validate_output(
        self, invocation: dict[str, Any], candidate: dict[str, Any]
    ) -> dict[str, Any]:
        """The output contract and the output validators on a candidate."""
        with self._operation(invocation):
            ctx = RunContext[Any](function_call=None, run=_current_run())
            try:
                if candidate["kind"] == "text":
                    text = await self._agent._validate_output(ctx, candidate["text"])
                    return {"kind": "accepted", "text": str(text), "value": None}
                value = await self._output.accept(ctx, candidate["value"])
                value = await self._agent._validate_output(ctx, value)
            except ValidationError as e:
                return {"kind": "retry", "reason": str(e)}
            except (ModelRetry, ToolError) as e:
                return {"kind": "retry", "reason": e.message}
            except Exception as e:
                logger.exception("the output validation failed")
                return {"kind": "failed", "error": f"{type(e).__name__}: {e}"}
            return {"kind": "accepted", "text": None, "value": value}

    def cancel(self, invocation: dict[str, Any]) -> None:
        """Cancels the callback in flight for ``invocation``; it still settles."""
        task = self._running.get(_operation_key(invocation))
        if task is not None:
            task.cancel()

    def _operation(self, invocation: dict[str, Any]) -> _Operation:
        return _Operation(self._running, _operation_key(invocation))


class _Operation:
    """Registers the current task as the callback of one operation while it runs."""

    def __init__(self, running: dict[tuple[str, int], asyncio.Task[Any]], key: tuple[str, int]):
        self._running = running
        self._key = key

    def __enter__(self) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._running[self._key] = task

    def __exit__(self, *exc: object) -> None:
        self._running.pop(self._key, None)


class OutputContract:
    """``output_type`` as the core sees it (``OutputSpec``) and as the host accepts a candidate:
    ``str``, a type validated as its schema, or an output function whose parameters are the
    final output's arguments and whose return value is the output (agents.md "Structured
    output")."""

    def __init__(
        self,
        *,
        schema: dict[str, Any] | None,
        accept: Callable[[RunContext[Any], Any], Awaitable[Any]],
    ) -> None:
        self.schema = schema
        self.accept = accept

    @classmethod
    def of(cls, agent: Agent[Any, Any]) -> OutputContract:
        output = agent._output_spec
        if output is str:
            return cls(schema=None, accept=_unexpected_value)
        if isinstance(output, type) or not callable(output):
            adapter: TypeAdapter[Any] = TypeAdapter(output)
            schema = adapter.json_schema()
            schema.pop("title", None)

            async def validate(ctx: RunContext[Any], value: Any) -> Any:
                return adapter.validate_python(value)

            return cls(schema=schema, accept=validate)
        return cls._of_function(output)

    @classmethod
    def _of_function(cls, fn: Callable[..., Any]) -> OutputContract:
        parameters = function_arguments_to_pydantic_model(fn)
        schema = parameters.model_json_schema()
        schema.pop("title", None)
        hints = get_type_hints(fn, include_extras=True)
        returns: TypeAdapter[Any] | None = (
            TypeAdapter(hints["return"]) if hints.get("return") not in (None, Any) else None
        )
        context = [
            n
            for n, h in hints.items()
            if n != "return" and is_context_type(h, allow_subclasses=True)
        ]

        async def call(ctx: RunContext[Any], value: Any) -> Any:
            fields = dict(parameters.model_validate(value))
            result = fn(**fields, **dict.fromkeys(context, ctx))
            if inspect.isawaitable(result):
                result = await result
            return returns.validate_python(result) if returns is not None else result

        return cls(schema=schema, accept=call)

    def spec(self, *, validated: bool) -> dict[str, Any]:
        """The core's ``OutputSpec``."""
        if self.schema is None:
            return {"kind": "text", "validated": validated}
        return {"kind": "structured", "schema": self.schema}


async def _unexpected_value(ctx: RunContext[Any], value: Any) -> Any:
    raise RuntimeError("a str output takes no structured candidate")


def _current_run() -> _TextRun:
    from .agent_run import _TextRun

    run = _BindingContextVar.get(None)
    if not isinstance(run, _TextRun):
        raise RuntimeError("an AgentHost callback ran outside its invocation's context")
    return run


def _function_call(invocation: dict[str, Any]) -> FunctionCall:
    from .._core_view import from_core_item

    call = invocation["function_call"]
    if isinstance(call, FunctionCall):
        return call
    if isinstance(call, dict) and "type" not in call:
        call = {**call, "type": "function_call"}
    item = from_core_item(call)
    assert isinstance(item, FunctionCall)
    return item


def _operation_key(invocation: dict[str, Any]) -> tuple[str, int]:
    return invocation["invocation_id"], invocation["operation"]
