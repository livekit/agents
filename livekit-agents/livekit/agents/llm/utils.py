from __future__ import annotations

import base64
import inspect
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, TypeAdapter, create_model
from pydantic.fields import Field, FieldInfo
from pydantic_core import PydanticUndefined
from typing_extensions import TypeVar

from livekit import rtc
from livekit.agents import llm, utils

from ..log import logger
from . import _strict
from .chat_context import ChatContext
from .tool_context import FunctionTool, get_function_info

if TYPE_CHECKING:
    from ..voice.events import RunContext


def _compute_lcs(old_ids: list[str], new_ids: list[str]) -> list[str]:
    """
    Standard dynamic-programming LCS to get the common subsequence
    of IDs (in order) that appear in both old_ids and new_ids.
    """
    n, m = len(old_ids), len(new_ids)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if old_ids[i - 1] == new_ids[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the actual LCS sequence
    lcs_ids = []
    i, j = n, m
    while i > 0 and j > 0:
        if old_ids[i - 1] == new_ids[j - 1]:
            lcs_ids.append(old_ids[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return list(reversed(lcs_ids))


@dataclass
class DiffOps:
    to_remove: list[str]
    to_create: list[
        tuple[str | None, str]
    ]  # (previous_item_id, id), if previous_item_id is None, add to the root


def compute_chat_ctx_diff(old_ctx: ChatContext, new_ctx: ChatContext) -> DiffOps:
    """Computes the minimal list of create/remove operations to transform old_ctx into new_ctx."""
    # TODO(theomonnom): Make ChatMessage hashable and also add update ops

    old_ids = [m.id for m in old_ctx.items]
    new_ids = [m.id for m in new_ctx.items]
    lcs_ids = set(_compute_lcs(old_ids, new_ids))

    to_remove = [msg.id for msg in old_ctx.items if msg.id not in lcs_ids]
    to_create: list[tuple[str | None, str]] = []

    last_id_in_sequence: str | None = None
    for new_msg in new_ctx.items:
        if new_msg.id in lcs_ids:
            last_id_in_sequence = new_msg.id
        else:
            if last_id_in_sequence is None:
                prev_id = None  # root
            else:
                prev_id = last_id_in_sequence

            to_create.append((prev_id, new_msg.id))
            last_id_in_sequence = new_msg.id

    return DiffOps(to_remove=to_remove, to_create=to_create)


def is_context_type(ty: type) -> bool:
    from ..voice.events import RunContext

    origin = get_origin(ty)
    is_call_context = ty is RunContext or origin is RunContext

    return is_call_context


@dataclass
class SerializedImage:
    inference_detail: str
    mime_type: str | None
    data_bytes: bytes | None = None
    external_url: str | None = None


def serialize_image(image: llm.ImageContent) -> SerializedImage:
    if isinstance(image.image, str):
        if image.image.startswith("data:"):
            header, b64_data = image.image.split(",", 1)
            encoded_data = base64.b64decode(b64_data)
            header_mime = header.split(";")[0].split(":")[1]
            if image.mime_type and image.mime_type != header_mime:
                logger.warning(
                    f"""Provided mime_type '{image.mime_type}' does not match data URL mime type
                    '{header_mime}'. Using provided mime_type."""
                )
                mime_type = image.mime_type
            else:
                mime_type = header_mime
            supported_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
            if mime_type not in supported_types:
                raise ValueError(
                    f"Unsupported mime_type {mime_type}. Must be jpeg, png, webp, or gif"
                )

            return SerializedImage(
                data_bytes=encoded_data,
                mime_type=mime_type,
                inference_detail=image.inference_detail,
            )
        else:
            return SerializedImage(
                mime_type=image.mime_type,
                inference_detail=image.inference_detail,
                external_url=image.image,
            )

    elif isinstance(image.image, rtc.VideoFrame):
        opts = utils.images.EncodeOptions()
        if image.inference_width and image.inference_height:
            opts.resize_options = utils.images.ResizeOptions(
                width=image.inference_width,
                height=image.inference_height,
                strategy="scale_aspect_fit",
            )
        encoded_data = utils.images.encode(image.image, opts)

        return SerializedImage(
            data_bytes=encoded_data,
            mime_type="image/jpeg",
            inference_detail=image.inference_detail,
        )
    raise ValueError("Unsupported image type")


def build_legacy_openai_schema(
    function_tool: FunctionTool, *, internally_tagged: bool = False
) -> dict[str, Any]:
    """non-strict mode tool description
    see https://serde.rs/enum-representations.html for the internally tagged representation"""
    model = function_arguments_to_pydantic_model(function_tool)
    info = get_function_info(function_tool)
    schema = model.model_json_schema()

    if internally_tagged:
        return {
            "name": info.name,
            "description": info.description or "",
            "parameters": schema,
            "type": "function",
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": info.name,
                "description": info.description or "",
                "parameters": schema,
            },
        }


def build_strict_openai_schema(
    function_tool: FunctionTool,
) -> dict[str, Any]:
    """strict mode tool description"""
    model = function_arguments_to_pydantic_model(function_tool)
    info = get_function_info(function_tool)
    schema = _strict.to_strict_json_schema(model)

    return {
        "type": "function",
        "function": {
            "name": info.name,
            "strict": True,
            "description": info.description or "",
            "parameters": schema,
        },
    }


ResponseFormatT = TypeVar("ResponseFormatT", default=None)


def is_typed_dict(cls) -> bool:
    return isinstance(cls, type) and issubclass(cls, dict) and hasattr(cls, "__annotations__")


# mostly from https://github.com/openai/openai-python/blob/main/src/openai/lib/_parsing/_completions.py
# and https://github.com/instructor-ai/instructor/blob/be7821e34fb10f7dabf658d684135297a2e40ef3/instructor/process_response.py#L812C1-L816C10


def to_response_format_param(
    response_format: type | dict,
) -> tuple[str, type[BaseModel] | TypeAdapter[Any]]:
    if isinstance(response_format, dict):
        # TODO(theomonnom): better type validation, copy TypedDict from OpenAI
        if response_format.get("type", "") not in ("text", "json_schema", "json_object"):
            raise TypeError("Unsupported response_format type")

        return response_format

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


def to_openai_response_format(response_format: type | dict) -> dict:
    name, json_schema_type = to_response_format_param(response_format)

    schema = _strict.to_strict_json_schema(json_schema_type)
    return {
        "type": "json_schema",
        "json_schema": {
            "schema": schema,
            "name": name,
            "strict": True,
        },
    }


def function_arguments_to_pydantic_model(func: Callable) -> type[BaseModel]:
    """Create a Pydantic model from a function’s signature. (excluding context types)"""

    from docstring_parser import parse_from_object

    fnc_name = func.__name__.split("_")
    fnc_name = "".join(x.capitalize() for x in fnc_name)
    model_name = fnc_name + "Args"

    docstring = parse_from_object(func)
    param_docs = {p.arg_name: p.description for p in docstring.params}

    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    # field_name -> (type, FieldInfo or default)
    fields: dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        type_hint = type_hints[param_name]

        if is_context_type(type_hint):
            continue

        default_value = param.default if param.default is not param.empty else ...
        field_info = Field()

        # Annotated[str, Field(description="...")]
        if get_origin(type_hint) is Annotated:
            annotated_args = get_args(type_hint)
            type_hint = annotated_args[0]
            field_info = next(
                (x for x in annotated_args[1:] if isinstance(x, FieldInfo)), field_info
            )

        if default_value is not ... and field_info.default is PydanticUndefined:
            field_info.default = default_value

        if field_info.description is None:
            field_info.description = param_docs.get(param_name, None)

        fields[param_name] = (type_hint, field_info)

    return create_model(model_name, **fields)


def pydantic_model_to_function_arguments(
    *,
    function_tool: Callable,
    model: BaseModel,
    call_ctx: RunContext | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Convert a model’s fields into function args/kwargs.
    Raises TypeError if required params are missing
    """
    signature = inspect.signature(function_tool)
    type_hints = get_type_hints(function_tool, include_extras=True)

    context_dict = {}
    for param_name, _ in signature.parameters.items():
        type_hint = type_hints[param_name]
        if is_context_type(type_hint) and call_ctx is not None:
            context_dict[param_name] = call_ctx

    bound = signature.bind(**{**model.model_dump(), **context_dict})
    bound.apply_defaults()
    return bound.args, bound.kwargs
