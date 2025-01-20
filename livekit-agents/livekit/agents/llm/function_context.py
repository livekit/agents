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
from typing import (
    Annotated,
    Any,
    Callable,
    List,
    Protocol,
    Type,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from typing_extensions import TypeGuard


@runtime_checkable
class AIFunction(Protocol):
    __livekit_agents_ai_callable: bool
    __name__: str
    __doc__: str | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def ai_function(f: Callable | None = None) -> Callable[[Callable], AIFunction]:
    def deco(f) -> AIFunction:
        setattr(f, "__livekit_agents_ai_callable", True)
        return f

    if callable(f):
        return deco(f)

    return deco


def is_ai_function(f: Callable) -> TypeGuard[AIFunction]:
    return getattr(f, "__livekit_agents_ai_callable", False)


def find_ai_functions(cls: Type) -> List[AIFunction]:
    methods: list[AIFunction] = []
    for _, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if is_ai_function(method):
            methods.append(method)
    return methods


class FunctionContext:
    """Stateless container for a set of AI functions"""

    def __init__(self, ai_functions: list[AIFunction]) -> None:
        self.update_ai_functions(ai_functions)

    @classmethod
    def empty(cls) -> FunctionContext:
        return cls([])

    @property
    def ai_functions(self) -> dict[str, AIFunction]:
        return self._ai_functions_map.copy()

    def update_ai_functions(self, ai_functions: list[AIFunction]) -> None:
        self._ai_functions = ai_functions

        for method in find_ai_functions(self.__class__):
            ai_functions.append(method)

        self._ai_functions_map = {}
        for fnc in ai_functions:
            if fnc.__name__ in self._ai_functions_map:
                raise ValueError(f"duplicate function name: {fnc.__name__}")

            self._ai_functions_map[fnc.__name__] = fnc

    def copy(self) -> FunctionContext:
        return FunctionContext(self._ai_functions.copy())


def build_legacy_openai_schema(
    ai_function: AIFunction, *, internally_tagged: bool = False
) -> dict[str, Any]:
    """non-strict mode tool description
    see https://serde.rs/enum-representations.html for the internally tagged representation"""
    model = build_pydantic_model_from_function(ai_function)
    schema = model.model_json_schema()

    fnc_name = ai_function.__name__
    fnc_description = ai_function.__doc__

    if internally_tagged:
        return {
            "name": fnc_name,
            "description": fnc_description or "",
            "parameters": schema,
            "type": "function",
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": fnc_name,
                "description": fnc_description or "",
                "parameters": schema,
            },
        }


def build_pydantic_model_from_function(
    func: Callable,
) -> type[BaseModel]:
    fnc_name = func.__name__.split("_")
    fnc_name = "".join(x.capitalize() for x in fnc_name)
    model_name = fnc_name + "Args"

    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    # field_name -> (type, FieldInfo or default)
    fields: dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        annotation = type_hints[param_name]
        default_value = param.default if param.default is not param.empty else ...

        # Annotated[str, Field(description="...")]
        if get_origin(annotation) is Annotated:
            annotated_args = get_args(annotation)
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
            fields[param_name] = (annotation, default_value)

    return create_model(model_name, **fields)
