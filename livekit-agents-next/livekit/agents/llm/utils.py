from __future__ import annotations

import contextlib
import inspect
import json
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    cast,
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
from .._generated import public_llm_utils as _public
from .._generated.provider_data import (
    TOOL_BEHAVIOR,
    ProviderFormat,
    ToolSchema,
    ToolSchemaLegacy,
    ToolSchemaProvider,
    ToolSchemaRaw,
    ToolSchemaStrict,
)
from .._generated.provider_format import DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE
from .._generated.public_llm_utils import *  # noqa: F403
from ..log import logger
from .chat_context import ChatContext, ChatRole, ImageContent
from .tool_context import FunctionTool, RawFunctionTool, ToolError

if TYPE_CHECKING:
    from ..voice.events import RunContext
    from .chat_context import FunctionCall

_ContextT = TypeVar("_ContextT")


def remove_instructions(chat_ctx: ChatContext) -> None:
    """Remove framework instructions, preserving application-authored system messages."""
    for index in reversed(_core.remove_instructions([item.id for item in chat_ctx.items])):
        del chat_ctx.items[index]


def compute_chat_ctx_diff(old_ctx: ChatContext, new_ctx: ChatContext) -> DiffOps:
    """The minimal create/remove/update operations turning ``old_ctx`` into ``new_ctx``."""
    return _core.compute_chat_ctx_diff(old_ctx.items, new_ctx.items)


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


async def serialize_image(image: ImageContent, *, use_cache: bool = True) -> SerializedImage:
    """The image's bytes (a frame encoded, resized to its inference size) or its URL."""
    source = image.image
    width, height = image.inference_width, image.inference_height
    detail, declared_mime = image.inference_detail, image.mime_type
    cache_key = (source, width, height, detail, declared_mime)
    if use_cache and (cached := image._cache.get("serialized_image")) is not None:
        key, serialized = cast(tuple[tuple[object, ...], SerializedImage], cached)
        if key == cache_key:
            return serialized
    if isinstance(source, str):
        serialized = await _core.serialize_image_url(source, detail, declared_mime)
    elif isinstance(source, rtc.VideoFrame):
        serialized = await _core.serialize_image_frame(source, detail, width, height)
    else:
        raise ValueError("Unsupported image type")
    if use_cache:
        image._cache["serialized_image"] = (cache_key, serialized)
    return serialized


def tool_schema(tool: FunctionTool | RawFunctionTool) -> tuple[str, str, dict[str, object]]:
    """A tool's name, description and parameters' JSON schema: what every form of its schema
    (the providers', the tool search's) is built from."""
    from ..voice import _LocalAgentTool

    if isinstance(tool, _LocalAgentTool):
        return tool.info.name, tool.info.description or "", tool.schema
    if isinstance(tool, RawFunctionTool):
        raw = tool.info.raw_schema
        return raw["name"], raw.get("description", ""), raw.get("parameters", {})
    parameters = function_arguments_to_pydantic_model(tool).model_json_schema()
    return tool.info.name, tool.info.description or "", parameters


def build_legacy_openai_schema(
    function_tool: FunctionTool, *, internally_tagged: bool = False
) -> dict[str, object]:
    """Non-strict tool schema. ``internally_tagged`` follows
    https://serde.rs/enum-representations.html."""
    return _core.build_legacy_openai_schema(
        *tool_schema(function_tool), internally_tagged=internally_tagged
    )


def build_strict_openai_schema(
    function_tool: FunctionTool,
) -> dict[str, object]:
    """Strict tool schema."""
    return _core.build_strict_openai_schema(*tool_schema(function_tool))


ResponseFormatT = TypeVar("ResponseFormatT", default=None)


def is_typed_dict(cls: object) -> bool:
    return isinstance(cls, type) and issubclass(cls, dict) and hasattr(cls, "__annotations__")


# mostly from https://github.com/openai/openai-python/blob/main/src/openai/lib/_parsing/_completions.py
# and https://github.com/instructor-ai/instructor/blob/be7821e34fb10f7dabf658d684135297a2e40ef3/instructor/process_response.py#L812C1-L816C10


def to_response_format_param(
    response_format: type | dict[str, object],
) -> tuple[str, type[BaseModel] | TypeAdapter[object]]:
    if isinstance(response_format, dict):
        raise TypeError("Unsupported response_format type")

    if is_typed_dict(response_format):
        response_format = create_model(
            response_format.__name__,
            **{k: (v, ...) for k, v in response_format.__annotations__.items()},  # type: ignore
        )
    json_schema_type: type[BaseModel] | TypeAdapter[object] | None = None
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


def to_openai_response_format(response_format: type | dict[str, object]) -> dict[str, object]:
    name, json_schema_type = to_response_format_param(response_format)
    raw = (
        json_schema_type.json_schema()
        if isinstance(json_schema_type, TypeAdapter)
        else json_schema_type.model_json_schema()
    )
    return _core.to_openai_response_format(name, raw)


def function_arguments_to_pydantic_model(func: Callable[..., object]) -> type[BaseModel]:
    """Create a Pydantic model from a function's signature, excluding context parameters."""
    from docstring_parser import parse_from_object

    fnc_names = func.__name__.split("_")
    fnc_name = "".join(x.capitalize() for x in fnc_names)
    model_name = fnc_name + "Args"

    docstring = parse_from_object(func)
    param_docs = {p.arg_name: p.description for p in docstring.params}

    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

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


def parse_function_arguments(json_arguments: str) -> dict[str, object]:
    """A tool call's raw JSON arguments as a dict, repaired when malformed (leaked chat-template
    tokens stripped, double encoding unwrapped). Raises ``ValueError`` when unrecoverable or not
    a dict."""
    return _core.parse_function_arguments(json_arguments)


def prepare_function_arguments(
    *,
    fnc: FunctionTool | RawFunctionTool,
    json_arguments: str | dict[str, object],
    call_ctx: RunContext[_ContextT] | None = None,
    fnc_call: FunctionCall | None = None,
) -> tuple[tuple[object, ...], dict[str, object]]:
    """The positional and keyword arguments to call a function tool with the LLM's arguments.

    Validation failures (bad JSON, invalid or missing parameters) raise :class:`ToolError` so
    the LLM gets a concrete message and can self-correct. When ``fnc_call`` is given and
    ``json_arguments`` is a string, the repaired JSON is written back to ``fnc_call.arguments``.
    """
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

        # before validation, so a validation failure still leaves valid JSON in the history
        if fnc_call is not None:
            canonical = json.dumps(args_dict, default=str)
            if canonical != json_arguments:
                fnc_call.arguments = canonical

    try:
        return bind_function_arguments(fnc, validated_arguments(fnc, args_dict), call_ctx)
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
    fnc: FunctionTool | RawFunctionTool, args_dict: dict[str, object]
) -> dict[str, object]:
    """A tool call's arguments after schema validation, keyed by parameter name.

    Every declared parameter is present (an omitted one carries its default) and values are
    coerced to their annotated types, so ``{"order_id": "5"}`` and
    ``{"order_id": "5", "locale": None}`` produce the same mapping, as do ``1`` and ``1.0``
    for a ``float`` parameter. Compare two calls on this form (see ``DuplicateScope``), not
    on the raw arguments the LLM emitted.

    Raises the same validation errors as calling the tool would. Raw function tools have no
    per-parameter schema, so their arguments are returned exactly as sent.
    """
    if isinstance(fnc, RawFunctionTool):
        return args_dict

    model_type = function_arguments_to_pydantic_model(fnc)

    # strict schemas make defaulted fields nullable, null meaning "use the default"
    schema = model_type.model_json_schema()
    resolved = _core.inject_schema_defaults(args_dict, schema)

    model = model_type.model_validate(resolved)  # can raise ValidationError
    return _shallow_model_dump(model)


def bind_function_arguments(
    fnc: FunctionTool | RawFunctionTool,
    fields: dict[str, object],
    call_ctx: RunContext[_ContextT] | None,
) -> tuple[tuple[object, ...], dict[str, object]]:
    """The positional and keyword arguments of a call of ``fnc``: its validated ``fields`` and
    the ``RunContext`` where the signature takes one."""
    signature = inspect.signature(fnc)
    type_hints = get_type_hints(fnc, include_extras=True)

    if isinstance(fnc, FunctionTool):
        raw_fields = fields
    elif isinstance(fnc, RawFunctionTool):
        # raw tools take a required ``raw_arguments: dict[str, object]`` parameter
        raw_fields = {"raw_arguments": fields}
    else:
        raise ValueError(f"Unsupported function tool type: {type(fnc)}")

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


def _shallow_model_dump(model: BaseModel, *, by_alias: bool = False) -> dict[str, object]:
    result = {}
    for name, field_info in model.__class__.model_fields.items():
        key = field_info.alias if by_alias and field_info.alias else name
        result[key] = getattr(model, name)
    return result


def _is_valid_function_output(value: object) -> bool:
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


def _value_text(value: object) -> str | None:
    if not _is_valid_function_output(value):
        # dataclasses and models, alone or in lists and dicts, reach the model as their JSON
        with contextlib.suppress(PydanticSerializationError):
            value = to_json(value).decode()
    return str(value) if _is_valid_function_output(value) else None


if TYPE_CHECKING:
    from anthropic.types import MessageParam as AnthropicMessage
    from google.genai.types import ContentDict as GoogleMessage
    from mistralai.client.models import InputEntriesTypedDict as MistralMessage
    from openai.types.chat import ChatCompletionMessageParam as OpenAIMessage
    from openai.types.responses import ResponseInputItemParam as OpenAIResponseItem
    from types_aiobotocore_bedrock_runtime.type_defs import MessageTypeDef as BedrockMessage

    ProviderMessages: TypeAlias = (
        list[OpenAIMessage]
        | list[OpenAIResponseItem]
        | list[GoogleMessage]
        | list[BedrockMessage]
        | list[AnthropicMessage]
        | list[MistralMessage]
    )
    from .tool_context import ProviderTool, ToolContext


def convert_mid_conversation_instructions(
    chat_ctx: ChatContext,
    *,
    role: ChatRole = "user",
    template: str = DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE,
) -> ChatContext:
    """Rewrite every system or developer message but a leading preamble as a ``role`` message
    wrapped in ``template``, so it keeps its position; one without text is dropped."""

    plan = _core.convert_mid_conversation_instructions(chat_ctx.items, role=role, template=template)
    return ChatContext(
        [chat_ctx.items[c.index] if c.rewritten is None else c.rewritten for c in plan]
    )


_DATA: dict[ProviderFormat, type[Any]] = {
    "anthropic": AnthropicFormatData,
    "aws": BedrockFormatData,
    "google": GoogleFormatData,
    "mistralai": MistralFormatData,
}


async def to_chat_ctx(
    chat_ctx: ChatContext,
    format: ProviderFormat,
    *,
    inject_dummy_user_message: bool = True,
    inject_trailing_user_message: bool = False,
    thought_signatures: dict[str, bytes] | None = None,
) -> tuple[
    ProviderMessages,
    GoogleFormatData | BedrockFormatData | AnthropicFormatData | MistralFormatData | None,
]:
    images = {
        content.id: await serialize_image(content)
        for item in chat_ctx.items
        if item.type == "message"
        for content in item.content
        if isinstance(content, ImageContent)
    }
    formatted = await _core.to_provider_format(
        format,
        chat_ctx.items,
        images=images,
        inject_dummy_user_message=inject_dummy_user_message,
        inject_trailing_user_message=inject_trailing_user_message,
        thought_signatures=thought_signatures,
    )
    cls = _DATA.get(format)
    data = cls(**cast("dict[str, Any]", formatted.data)) if cls is not None else None
    return cast("ProviderMessages", formatted.messages), data


def to_fnc_ctx(
    tool_ctx: ToolContext,
    format: ProviderFormat,
    *,
    strict: bool = True,
    provider_tool_type: type[ProviderTool] | None = None,
    tool_behavior: TOOL_BEHAVIOR | None = None,
    use_parameters_json_schema: bool = True,
) -> list[dict[str, Any]]:
    responses = format == "openai.responses"
    specs: list[ToolSchema] = []
    for tool in tool_ctx.flatten() if responses else tool_ctx.function_tools.values():
        if isinstance(tool, RawFunctionTool):
            specs.append(ToolSchemaRaw(name=tool.info.name, raw_schema=tool.info.raw_schema))
        elif isinstance(tool, FunctionTool):
            if strict and format in ("openai", "anthropic"):
                specs.append(ToolSchemaStrict(schema=build_strict_openai_schema(tool)))
                continue
            fnc = build_legacy_openai_schema(tool, internally_tagged=format != "openai")
            specs.append(ToolSchemaLegacy(schema=fnc))
        elif responses and provider_tool_type and isinstance(tool, provider_tool_type):
            if hasattr(tool, "to_dict"):
                specs.append(ToolSchemaProvider(schema=tool.to_dict()))
    return cast(
        "list[dict[str, Any]]",
        _core.to_fnc_ctx(
            format,
            specs,
            tool_behavior=tool_behavior,
            use_parameters_json_schema=use_parameters_json_schema,
        ),
    )


__all__ = [
    "TOOL_BEHAVIOR",
    "to_chat_ctx",
    "to_fnc_ctx",
]
__all__ += _public.__all__
