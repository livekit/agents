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
import typing
from typing import Any

from livekit.agents.llm import function_context, llm
from livekit.agents.llm.function_context import _is_optional_type

__all__ = ["build_oai_function_description"]


def build_oai_function_description(
    fnc_info: function_context.FunctionInfo,
    capabilities: llm.LLMCapabilities | None = None,
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

        is_optional, inner_th = _is_optional_type(arg_info.type)

        if typing.get_origin(inner_th) is list:
            inner_type = typing.get_args(inner_th)[0]
            p["type"] = "array"
            p["items"] = {}
            p["items"]["type"] = type2str(inner_type)

            if arg_info.choices:
                p["items"]["enum"] = arg_info.choices
        else:
            p["type"] = type2str(inner_th)
            if arg_info.choices:
                p["enum"] = arg_info.choices
                if (
                    inner_th is int
                    and capabilities
                    and not capabilities.supports_choices_on_int
                ):
                    raise ValueError(
                        f"Parameter '{arg_info.name}' uses 'choices' with 'int', which is not supported by this model."
                    )

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
