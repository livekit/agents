from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import pydantic
from pydantic import BaseModel, TypeAdapter, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticSerializationError, PydanticUndefined, to_json
from typing_extensions import TypeVar

from livekit import rtc

from .. import _core
from .._generated.tool_report import (
    ToolOutcomeCompleted,
    ToolOutcomeFailed,
    ToolOutcomeInputRequired,
    ToolOutcomeInternal,
    ToolOutcomeInvalid,
    ToolOutcomeModelRetry,
    ToolOutcomeNothing,
    ToolOutcomeRejected,
    ToolOutcomeStopResponse,
    ToolOutcomeToolError,
    ToolOutcomeValue,
    ToolOutcomeWorking,
    ToolReport,
)
from ..log import logger
from .chat_context import ChatContext, ImageContent
from .tool_context import FunctionTool, RawFunctionTool, ToolError, is_raw_function_tool

if TYPE_CHECKING:
    from .._generated.llm import FunctionToolCall
    from ..voice.events import RunContext
    from .chat_context import FunctionCall, FunctionCallOutput
    from .tool_context import ToolContext

THINK_TAG_START = "<think>"
THINK_TAG_END = "</think>"


@dataclass
class DiffOps:
    to_remove: list[str]
    to_create: list[
        tuple[str | None, str]
    ]  # (previous_item_id, id), if previous_item_id is None, add to the root
    to_update: list[
        tuple[str | None, str]
    ]  # (previous_item_id, id), the items with the same id but different content


def compute_chat_ctx_diff(old_ctx: ChatContext, new_ctx: ChatContext) -> DiffOps:
    """The minimal create/remove/update operations turning ``old_ctx`` into ``new_ctx``."""
    from .._core_view import to_core_items

    to_remove, to_create, to_update = _core.compute_chat_ctx_diff(
        to_core_items(old_ctx.items), to_core_items(new_ctx.items)
    )
    return DiffOps(to_remove=to_remove, to_create=to_create, to_update=to_update)


def is_context_type(ty: type, *, allow_subclasses: bool = False) -> bool:
    from ..voice.events import RunContext

    origin = get_origin(ty)

    if not allow_subclasses:
        return ty is RunContext or origin is RunContext

    if origin is not None:
        try:
            return issubclass(origin, RunContext)
        except TypeError:
            return False

    try:
        return issubclass(ty, RunContext)
    except TypeError:
        return False


@dataclass
class SerializedImage:
    inference_detail: str
    mime_type: str | None
    data_bytes: bytes | None = None
    external_url: str | None = None


def serialize_image(image: ImageContent, *, use_cache: bool = True) -> SerializedImage:
    """The image's bytes (a frame encoded, resized to its inference size) or its URL."""
    if use_cache and (cached := image._cache.get("serialized_image")) is not None:
        return cast(SerializedImage, cached)
    if isinstance(image.image, str):
        data, mime_type, url, _ = _core.serialize_image_url(image.image, image.mime_type)
    elif isinstance(image.image, rtc.VideoFrame):
        frame = image.image
        if frame.type != rtc.VideoBufferType.RGBA:
            frame = frame.convert(rtc.VideoBufferType.RGBA)
        data, mime_type = _core.serialize_image_frame(
            bytes(frame.data),
            frame.width,
            frame.height,
            image.inference_width,
            image.inference_height,
        )
        url = None
    else:
        raise ValueError("Unsupported image type")
    serialized = SerializedImage(
        inference_detail=image.inference_detail,
        mime_type=mime_type,
        data_bytes=data,
        external_url=url,
    )
    if use_cache:
        image._cache["serialized_image"] = serialized
    return serialized


def tool_schema(tool: FunctionTool | RawFunctionTool) -> tuple[str, str, dict[str, Any]]:
    """A tool's name, description and parameters' JSON schema: what every form of its schema
    (the providers', the tool search's) is built from."""
    if isinstance(tool, RawFunctionTool):
        raw = tool.info.raw_schema
        return raw["name"], raw.get("description", ""), raw.get("parameters", {})
    parameters = function_arguments_to_pydantic_model(tool).model_json_schema()
    return tool.info.name, tool.info.description or "", parameters


def build_legacy_openai_schema(
    function_tool: FunctionTool, *, internally_tagged: bool = False
) -> dict[str, Any]:
    """non-strict mode tool description
    see https://serde.rs/enum-representations.html for the internally tagged representation"""
    return _core.build_legacy_openai_schema(  # type: ignore[no-any-return]
        *tool_schema(function_tool), internally_tagged=internally_tagged
    )


def build_strict_openai_schema(
    function_tool: FunctionTool,
) -> dict[str, Any]:
    """strict mode tool description"""
    return _core.build_strict_openai_schema(*tool_schema(function_tool))  # type: ignore[no-any-return]


ResponseFormatT = TypeVar("ResponseFormatT", default=None)


def is_typed_dict(cls: type | Any) -> bool:
    return isinstance(cls, type) and issubclass(cls, dict) and hasattr(cls, "__annotations__")


# mostly from https://github.com/openai/openai-python/blob/main/src/openai/lib/_parsing/_completions.py
# and https://github.com/instructor-ai/instructor/blob/be7821e34fb10f7dabf658d684135297a2e40ef3/instructor/process_response.py#L812C1-L816C10


def to_response_format_param(
    response_format: type | dict[str, Any],
) -> tuple[str, type[BaseModel] | TypeAdapter[Any]]:
    if isinstance(response_format, dict):
        raise TypeError("Unsupported response_format type")

    # add support for TypedDict
    if is_typed_dict(response_format):
        response_format = create_model(
            response_format.__name__,
            **{k: (v, ...) for k, v in response_format.__annotations__.items()},  # type: ignore
        )
    json_schema_type: type[BaseModel] | TypeAdapter[Any] | None = None
    if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
        name = response_format.__name__
        json_schema_type = response_format
    elif inspect.isclass(response_format) and hasattr(
        response_format, "__pydantic_config__"
    ):  # @pydantic.dataclass
        name = response_format.__name__
        json_schema_type = TypeAdapter(response_format)
    else:
        raise TypeError(f"Unsupported response_format type - {response_format}")

    return name, json_schema_type


def to_openai_response_format(response_format: type | dict[str, Any]) -> dict[str, Any]:
    name, json_schema_type = to_response_format_param(response_format)
    raw = (
        json_schema_type.json_schema()
        if isinstance(json_schema_type, TypeAdapter)
        else json_schema_type.model_json_schema()
    )
    return _core.to_openai_response_format(name, raw)  # type: ignore[no-any-return]


def function_arguments_to_pydantic_model(func: Callable[..., Any]) -> type[BaseModel]:
    """Create a Pydantic model from a function's signature. (excluding context types)"""

    from docstring_parser import parse_from_object

    fnc_names = func.__name__.split("_")
    fnc_name = "".join(x.capitalize() for x in fnc_names)
    model_name = fnc_name + "Args"

    docstring = parse_from_object(func)
    param_docs = {p.arg_name: p.description for p in docstring.params}

    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    # field_name -> (type, FieldInfo or default)
    fields: dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        type_hint = type_hints[param_name]

        if is_context_type(type_hint, allow_subclasses=True):
            continue

        default_value = param.default if param.default is not param.empty else ...
        field_info = FieldInfo.from_annotation(type_hint)

        if default_value is not ... and field_info.default is PydanticUndefined:
            field_info.default = default_value

        if field_info.description is None:
            field_info.description = param_docs.get(param_name)

        fields[param_name] = (field_info.annotation, field_info)

    return create_model(model_name, **fields)


def parse_function_arguments(json_arguments: str) -> dict[str, Any]:
    """A tool call's raw JSON arguments as a dict, repaired when malformed (leaked chat-template
    tokens stripped, double encoding unwrapped). Raises ``ValueError`` when unrecoverable or not
    a dict."""
    return _core.parse_function_arguments(json_arguments)  # type: ignore[no-any-return]


def answer_arguments(fnc: FunctionTool | RawFunctionTool, answer: str) -> dict[str, Any] | None:
    """The arguments that hand ``answer`` to ``fnc`` as an ``InputRequired(tool=)`` names it:
    the tool's first string parameter. ``None`` for a tool without one: the model reads the
    answer and calls the tool itself."""
    if is_raw_function_tool(fnc):
        properties = fnc.info.raw_schema.get("parameters", {}).get("properties", {})
        names = [n for n, p in properties.items() if p.get("type") == "string"]
    else:
        hints = get_type_hints(fnc, include_extras=True)
        names = [n for n in inspect.signature(fnc).parameters if _is_str(hints.get(n))]
    return {names[0]: answer} if names else None


def _is_str(hint: Any) -> bool:
    return hint is str or (get_origin(hint) is Annotated and get_args(hint)[0] is str)


def prepare_function_arguments(
    *,
    fnc: FunctionTool | RawFunctionTool,
    json_arguments: str | dict[str, Any],
    call_ctx: RunContext[Any] | None = None,
    fnc_call: FunctionCall | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:  # returns args, kwargs
    """Create the positional and keyword arguments to call a function tool from
    the raw function output from the LLM.

    Argument-validation failures (bad JSON, pydantic ValidationError, missing
    required params) are surfaced as :class:`ToolError` so the LLM gets a
    concrete error message and can self-correct on its next turn.

    When ``fnc_call`` is provided and ``json_arguments`` is a string, the
    canonicalized JSON (post json_repair) is written back to
    ``fnc_call.arguments`` BEFORE validation runs.
    """
    # phase 1: parse — raw JSON failures raise ToolError immediately (no
    # canonical to provide since the input itself was unparseable)
    if isinstance(json_arguments, dict):
        args_dict = json_arguments
    else:
        try:
            args_dict = parse_function_arguments(json_arguments)
        except ValueError as e:
            logger.error(
                f"error parsing arguments for `{fnc.info.name}`",
                extra={"function": fnc.info.name, "lk.pii.arguments": json_arguments},
            )
            raise ToolError(f"Error parsing arguments for `{fnc.info.name}`: {e}") from e

        # write canonical BEFORE validation so a downstream validation failure
        # still leaves valid JSON in chat history
        if fnc_call is not None:
            canonical = json.dumps(args_dict, default=str)
            if canonical != json_arguments:
                fnc_call.arguments = canonical

    # phase 2: validate + bind
    try:
        return _prepare_function_arguments(fnc=fnc, args_dict=args_dict, call_ctx=call_ctx)
    except ToolError:
        raise
    except (pydantic.ValidationError, ValueError, TypeError) as e:
        logger.error(
            f"error parsing arguments for `{fnc.info.name}`",
            extra={"function": fnc.info.name, "lk.pii.arguments": json_arguments},
        )
        raise ToolError(f"Error parsing arguments for `{fnc.info.name}`: {e}") from e
    except Exception:
        logger.exception(
            f"error parsing arguments for `{fnc.info.name}`",
            extra={"function": fnc.info.name, "lk.pii.arguments": json_arguments},
        )
        raise


def validated_arguments(
    fnc: FunctionTool | RawFunctionTool, args_dict: dict[str, Any]
) -> dict[str, Any]:
    """A tool call's arguments after schema validation, keyed by parameter name.

    Every declared parameter is present — an omitted one carries its default — and
    values are coerced to their annotated types, so ``{"order_id": "5"}`` and
    ``{"order_id": "5", "locale": None}`` produce the same mapping, as do ``1`` and
    ``1.0`` for a ``float`` parameter. That makes this the form to compare two calls
    on (see ``DuplicateScope``), not the raw arguments the LLM emitted.

    Raises the same validation errors as calling the tool would. Raw function tools
    have no per-parameter schema, so their arguments are returned exactly as sent.
    """
    if isinstance(fnc, RawFunctionTool):
        return args_dict

    model_type = function_arguments_to_pydantic_model(fnc)

    # Strict LLM schemas represent defaulted fields as nullable (see
    # _ensure_strict_json_schema): null means "use the default". Resolve
    # the sentinel, including in nested models, before validation.
    schema = model_type.model_json_schema()
    args_dict = _core.inject_schema_defaults(args_dict, schema)

    model = model_type.model_validate(args_dict)  # can raise ValidationError
    return _shallow_model_dump(model)


def _prepare_function_arguments(
    *,
    fnc: FunctionTool | RawFunctionTool,
    args_dict: dict[str, Any],
    call_ctx: RunContext[Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return bind_function_arguments(fnc, validated_arguments(fnc, args_dict), call_ctx)


def bind_function_arguments(
    fnc: FunctionTool | RawFunctionTool,
    fields: dict[str, Any],
    call_ctx: RunContext[Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The positional and keyword arguments of a call of ``fnc``: its validated ``fields`` and
    the ``RunContext`` where the signature takes one."""
    signature = inspect.signature(fnc)
    type_hints = get_type_hints(fnc, include_extras=True)

    if isinstance(fnc, FunctionTool):
        raw_fields = fields
    elif isinstance(fnc, RawFunctionTool):
        # e.g async def open_gate(self, raw_arguments: dict[str, object]):
        # raw_arguments is required when using raw function tools
        raw_fields = {"raw_arguments": fields}
    else:
        raise ValueError(f"Unsupported function tool type: {type(fnc)}")

    # inject RunContext (or subclasses) if needed
    context_dict = {}
    for param_name, _ in signature.parameters.items():
        type_hint = type_hints[param_name]
        if not is_context_type(type_hint, allow_subclasses=True) or call_ctx is None:
            continue

        expected_type = get_origin(type_hint) or type_hint
        if isinstance(call_ctx, expected_type):
            context_dict[param_name] = call_ctx
        else:
            logger.error(
                f"context type mismatch for parameter '{param_name}': "
                f"expected {expected_type.__name__}, got {type(call_ctx).__name__}"
            )

    bound = signature.bind(**{**raw_fields, **context_dict})
    bound.apply_defaults()
    return bound.args, bound.kwargs


def _shallow_model_dump(model: BaseModel, *, by_alias: bool = False) -> dict[str, Any]:
    result = {}
    for name, field_info in model.__class__.model_fields.items():
        key = field_info.alias if by_alias and field_info.alias else name
        result[key] = getattr(model, name)
    return result


@dataclass
class ThinkingTokenFilter:
    """State for stripping thinking tokens from streamed content."""

    start_tag: str = THINK_TAG_START
    end_tag: str = THINK_TAG_END
    _buffer: str = ""
    _end_marker: str | None = None


def strip_thinking_tokens(
    content: str | list[Any] | None, state: ThinkingTokenFilter, *, final: bool = False
) -> str | None:
    """The visible part of a streamed delta, the thinking spans removed; ``state`` carries a
    span or a partial tag across deltas."""
    if isinstance(content, list):
        parts = [
            p
            if isinstance(p, str)
            else (p.get("text") if isinstance(p, dict) else getattr(p, "text", None))
            for p in content
        ]
        content = _core.flatten_delta_text([p for p in parts if isinstance(p, str)])
    elif content is not None and not isinstance(content, str):
        logger.warning(
            "unexpected streaming delta content type %s; dropping the chunk", type(content).__name__
        )
        content = None
    visible, state._buffer, state._end_marker = _core.strip_thinking(
        content, state.start_tag, state.end_tag, state._buffer, state._end_marker, final
    )
    return visible  # type: ignore[no-any-return]


def _is_valid_function_output(value: Any) -> bool:
    VALID_TYPES = (str, int, float, bool, complex, type(None))

    if isinstance(value, VALID_TYPES):
        return True
    elif (
        isinstance(value, list)
        or isinstance(value, set)
        or isinstance(value, frozenset)
        or isinstance(value, tuple)
    ):
        return all(_is_valid_function_output(item) for item in value)
    elif isinstance(value, dict):
        return all(
            isinstance(key, VALID_TYPES) and _is_valid_function_output(val)
            for key, val in value.items()
        )
    return False


@dataclass
class FunctionCallResult:
    fnc_call: FunctionCall
    fnc_call_out: FunctionCallOutput
    raw_output: Any
    raw_exception: BaseException | None
    fnc_call_updates: list[tuple[FunctionCall, FunctionCallOutput]] = field(default_factory=list)
    """Synthesized pairs from any ``ctx.update()`` calls during this standalone
    execution. Empty unless the tool actually called ``ctx.update()``."""


def _value_text(value: Any) -> str | None:
    """What the model reads of a value: the value itself, a model or dataclass as its JSON;
    ``None`` when the model cannot be told about it."""
    if not _is_valid_function_output(value):
        # dataclasses and models, alone or in lists and dicts, reach the model as their JSON
        with contextlib.suppress(PydanticSerializationError):
            value = to_json(value).decode()
    return str(value) if _is_valid_function_output(value) else None


def tool_report(output: Any, exception: BaseException | None = None) -> ToolReport:
    """What a tool returned or raised, classified for the core, which decides what the model
    reads and whether it replies."""
    from .tool_context import (
        Completed,
        Failed,
        InputRequired,
        ModelRetry,
        Rejected,
        StopResponse,
        ToolResult,
        Working,
    )

    if isinstance(output, ToolReport):
        return output
    if isinstance(output, BaseException):
        exception, output = output, None
    if isinstance(exception, ToolError):
        return ToolReport(outcome=ToolOutcomeToolError(message=exception.message))
    if isinstance(exception, ModelRetry):
        return ToolReport(outcome=ToolOutcomeModelRetry(message=exception.message))
    if isinstance(exception, StopResponse):
        return ToolReport(outcome=ToolOutcomeStopResponse())
    if exception is not None:
        return ToolReport(outcome=ToolOutcomeInternal())
    if isinstance(output, Completed):
        text = "completed" if output.value is None else _value_text(output.value)
        if text is None:
            return ToolReport(outcome=ToolOutcomeInvalid())
        return ToolReport(outcome=ToolOutcomeCompleted(text=text, verbatim=output.verbatim))
    if isinstance(output, Failed):
        return ToolReport(outcome=ToolOutcomeFailed(error=output.error))
    if isinstance(output, Rejected):
        return ToolReport(outcome=ToolOutcomeRejected(reason=output.reason))
    if isinstance(output, InputRequired):
        tool = output.tool.info.name if output.tool is not None else None
        asked = ToolOutcomeInputRequired(
            question=output.question, verbatim=output.verbatim, tool=tool
        )
        return ToolReport(outcome=asked, transition=output)
    if isinstance(output, Working):
        return ToolReport(outcome=ToolOutcomeWorking(message=output.message))
    if output is None:
        return ToolReport(outcome=ToolOutcomeNothing())
    if isinstance(output, ToolResult):
        text = "" if output.output is None else _value_text(output.output)
        if text is None:
            return ToolReport(outcome=ToolOutcomeInvalid())
        value = ToolOutcomeValue(
            text=text,
            reply_required=output.reply_required,
            verbatim=output.verbatim,
            more=getattr(output, "_more", None),
        )
        return ToolReport(outcome=value)
    text = _value_text(output)
    if text is None:
        return ToolReport(outcome=ToolOutcomeInvalid())
    return ToolReport(outcome=ToolOutcomeValue(text=text))


def make_function_call_output(
    *,
    fnc_call: FunctionCall,
    output: Any,
    exception: BaseException | None,
) -> FunctionCallResult:
    """Create a FunctionCallResult, handling ToolError, StopResponse, and validation. The
    output each outcome gives the model is the core's (``_core.tool_call_output``)."""
    if isinstance(output, BaseException):
        exception, output = output, None
    outcome = tool_report(output, exception).outcome
    if isinstance(outcome, ToolOutcomeInvalid):
        logger.error(
            f"AI function `{fnc_call.name}` returned an invalid output",
            extra={"call_id": fnc_call.call_id, "lk.pii.output": output},
        )
    text, is_error, reply_required = _core.tool_call_output(outcome)
    from .chat_context import FunctionCallOutput

    return FunctionCallResult(
        fnc_call=fnc_call,
        fnc_call_out=FunctionCallOutput(
            name=fnc_call.name,
            call_id=fnc_call.call_id,
            output=text,
            is_error=is_error,
            reply_required=reply_required,
        ),
        raw_output=output,
        raw_exception=exception,
    )


async def execute_function_call(
    tool_call: FunctionToolCall,
    tool_ctx: ToolContext,
    *,
    call_ctx: RunContext[Any] | None = None,
) -> FunctionCallResult:
    """Execute a function tool call and return the result."""
    from .chat_context import FunctionCall, FunctionCallOutput

    fnc_call = FunctionCall(
        call_id=tool_call.call_id,
        name=tool_call.name,
        arguments=tool_call.arguments or "{}",
        extra=tool_call.extra or {},
    )

    function_tool = tool_ctx.function_tools.get(tool_call.name)
    if function_tool is None:
        logger.warning(f"unknown AI function `{tool_call.name}`")
        msg = _core.unknown_tool_message(tool_call.name, list(tool_ctx.function_tools))
        return FunctionCallResult(
            fnc_call=fnc_call,
            fnc_call_out=FunctionCallOutput(
                name=tool_call.name,
                call_id=tool_call.call_id,
                output=msg,
                is_error=True,
            ),
            raw_output=None,
            raw_exception=ValueError(msg),
        )

    try:
        raw_args = tool_call.arguments or "{}"
        fnc_args, fnc_kwargs = prepare_function_arguments(
            fnc=function_tool,
            json_arguments=raw_args,
            call_ctx=call_ctx,
            fnc_call=fnc_call,
        )
        result = function_tool(*fnc_args, **fnc_kwargs)
        if asyncio.iscoroutine(result):
            result = await result

        out = make_function_call_output(fnc_call=fnc_call, output=result, exception=None)

    except Exception as e:
        if not isinstance(e, ToolError):
            logger.exception(
                f"exception executing AI function `{tool_call.name}`",
                extra={"call_id": tool_call.call_id, "lk.pii.arguments": tool_call.arguments},
            )
        out = make_function_call_output(fnc_call=fnc_call, output=None, exception=e)

    # surface any ctx.update() calls so callers can inspect them
    if call_ctx is not None and call_ctx._updates:
        out.fnc_call_updates = list(call_ctx._updates)
    return out
