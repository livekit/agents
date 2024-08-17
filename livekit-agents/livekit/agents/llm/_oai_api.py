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

import inspect
import json
import typing
from typing import Any

from . import function_context

__all__ = ["build_oai_function_description"]


def create_ai_function_info(
    fnc_ctx: function_context.FunctionContext,
    tool_call_id: str,
    fnc_name: str,
    raw_arguments: str,  # JSON string
) -> function_context.FunctionCallInfo:
    if fnc_name not in fnc_ctx.ai_functions:
        raise ValueError(f"AI function {fnc_name} not found")

    parsed_arguments: dict[str, Any] = {}
    try:
        if raw_arguments:  # ignore empty string
            parsed_arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        raise ValueError(
            f"AI function {fnc_name} received invalid JSON arguments - {raw_arguments}"
        )

    fnc_info = fnc_ctx.ai_functions[fnc_name]

    # Ensure all necessary arguments are present and of the correct type.
    sanitized_arguments: dict[str, Any] = {}
    for arg_info in fnc_info.arguments.values():
        if arg_info.name not in parsed_arguments:
            if arg_info.default is inspect.Parameter.empty:
                raise ValueError(
                    f"AI function {fnc_name} missing required argument {arg_info.name}"
                )
            continue

        arg_value = parsed_arguments[arg_info.name]
        if typing.get_origin(arg_info.type) is not None:
            if not isinstance(arg_value, list):
                raise ValueError(
                    f"AI function {fnc_name} argument {arg_info.name} should be a list"
                )

            inner_type = typing.get_args(arg_info.type)[0]
            sanitized_value = [
                _sanitize_primitive(
                    value=v, expected_type=inner_type, choices=arg_info.choices
                )
                for v in arg_value
            ]
        else:
            sanitized_value = _sanitize_primitive(
                value=arg_value, expected_type=arg_info.type, choices=arg_info.choices
            )

        sanitized_arguments[arg_info.name] = sanitized_value

    return function_context.FunctionCallInfo(
        tool_call_id=tool_call_id,
        raw_arguments=raw_arguments,
        function_info=fnc_info,
        arguments=sanitized_arguments,
    )


def build_oai_function_description(
    fnc_info: function_context.FunctionInfo,
) -> dict[str, Any]:
    def build_oai_property(arg_info: function_context.FunctionArgInfo):
        def type2str(t: type) -> str:
            if t is str:
                return "string"
            elif t in (int, float):
                return "number"
            elif t is bool:
                return "boolean"

            raise ValueError(f"unsupported type {t} for ai_property")

        p: dict[str, Any] = {}

        if arg_info.description:
            p["description"] = arg_info.description

        if typing.get_origin(arg_info.type) is list:
            inner_type = typing.get_args(arg_info.type)[0]
            p["type"] = "array"
            p["items"] = {}
            p["items"]["type"] = type2str(inner_type)

            if arg_info.choices:
                p["items"]["enum"] = arg_info.choices
        else:
            p["type"] = type2str(arg_info.type)
            if arg_info.choices:
                p["enum"] = arg_info.choices

        return p

    properties_info: dict[str, dict[str, Any]] = {}
    required_properties: list[str] = []

    for arg_info in fnc_info.arguments.values():
        if arg_info.default is inspect.Parameter.empty:
            required_properties.append(arg_info.name)

        properties_info[arg_info.name] = build_oai_property(arg_info)

    return {
        "type": "function",
        "function": {
            "name": fnc_info.name,
            "description": fnc_info.description,
            "parameters": {
                "type": "object",
                "properties": properties_info,
                "required": required_properties,
            },
        },
    }


def _sanitize_primitive(
    *, value: Any, expected_type: type, choices: tuple | None
) -> Any:
    if expected_type is str:
        if not isinstance(value, str):
            raise ValueError(f"expected str, got {type(value)}")
    elif expected_type in (int, float):
        if not isinstance(value, (int, float)):
            raise ValueError(f"expected number, got {type(value)}")

        if expected_type is int:
            if value % 1 != 0:
                raise ValueError("expected int, got float")

            value = int(value)
        elif expected_type is float:
            value = float(value)

    elif expected_type is bool:
        if not isinstance(value, bool):
            raise ValueError(f"expected bool, got {type(value)}")

    if choices and value not in choices:
        raise ValueError(f"invalid value {value}, not in {choices}")

    return value
