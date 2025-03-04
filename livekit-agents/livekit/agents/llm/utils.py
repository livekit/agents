from __future__ import annotations

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

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from . import _strict
from .chat_context import ChatContext
from .function_context import AIFunction, get_function_info

if TYPE_CHECKING:
    from ..voice.events import CallContext


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


# Convert FunctionContext to LLM API format


def is_context_type(ty: type) -> bool:
    from ..voice.events import CallContext

    origin = get_origin(ty)
    is_call_context = ty is CallContext or origin is CallContext

    return is_call_context


def build_legacy_openai_schema(
    ai_function: AIFunction, *, internally_tagged: bool = False
) -> dict[str, Any]:
    """non-strict mode tool description
    see https://serde.rs/enum-representations.html for the internally tagged representation"""
    model = function_arguments_to_pydantic_model(ai_function)
    info = get_function_info(ai_function)
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
    ai_function: AIFunction,
) -> dict[str, Any]:
    """strict mode tool description"""
    model = function_arguments_to_pydantic_model(ai_function)
    info = get_function_info(ai_function)
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


def function_arguments_to_pydantic_model(
    func: Callable,
) -> type[BaseModel]:
    """
    Create a Pydantic model from a function’s signature. (excluding context types)
    """
    fnc_name = func.__name__.split("_")
    fnc_name = "".join(x.capitalize() for x in fnc_name)
    model_name = fnc_name + "Args"

    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    # field_name -> (type, FieldInfo or default)
    fields: dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        type_hint = type_hints[param_name]

        if is_context_type(type_hint):
            continue

        default_value = param.default if param.default is not param.empty else ...

        # Annotated[str, Field(description="...")]
        if get_origin(type_hint) is Annotated:
            annotated_args = get_args(type_hint)
            actual_type = annotated_args[0]
            field_info = None

            for extra in annotated_args[1:]:
                if isinstance(extra, FieldInfo):
                    field_info = extra  # get the first FieldInfo
                    break

            if field_info:
                if default_value is not ... and field_info.default is None:
                    field_info.default = default_value
                fields[param_name] = (actual_type, field_info)
            else:
                fields[param_name] = (actual_type, default_value)

        else:
            fields[param_name] = (type_hint, default_value)

    return create_model(model_name, **fields)


def pydantic_model_to_function_arguments(
    *,
    ai_function: Callable,
    model: BaseModel,
    call_ctx: CallContext | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Convert a model’s fields into function args/kwargs.
    Raises TypeError if required params are missing
    """

    from ..voice.events import CallContext

    signature = inspect.signature(ai_function)
    type_hints = get_type_hints(ai_function, include_extras=True)

    context_dict = {}
    for param_name, _ in signature.parameters.items():
        type_hint = type_hints[param_name]
        if type_hint is CallContext and call_ctx is not None:
            context_dict[param_name] = call_ctx

    bound = signature.bind(**{**model.model_dump(), **context_dict})
    bound.apply_defaults()
    return bound.args, bound.kwargs
