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
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Protocol,
    runtime_checkable,
)

from typing_extensions import TypeGuard


class AIError(Exception):
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


@dataclass
class _AIFunctionInfo:
    name: str
    description: str | None


@runtime_checkable
class AIFunction(Protocol):
    __livekit_agents_ai_callable: _AIFunctionInfo

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def ai_function(
    f: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable], AIFunction]:
    def deco(func) -> AIFunction:
        from docstring_parser import parse_from_object

        docstring = parse_from_object(func)
        info = _AIFunctionInfo(
            name=name or func.__name__,
            description=description or docstring.description,
        )
        setattr(func, "__livekit_agents_ai_callable", info)
        return func

    if callable(f):
        return deco(f)

    return deco


def is_ai_function(f: Callable) -> TypeGuard[AIFunction]:
    return hasattr(f, "__livekit_agents_ai_callable")


def get_function_info(f: AIFunction) -> _AIFunctionInfo:
    return getattr(f, "__livekit_agents_ai_callable")


def find_ai_functions(cls_or_obj: Any) -> list[AIFunction]:
    methods: list[AIFunction] = []
    for _, member in inspect.getmembers(cls_or_obj):
        if is_ai_function(member):
            methods.append(member)
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

        for method in find_ai_functions(self):
            ai_functions.append(method)

        self._ai_functions_map = {}
        for fnc in ai_functions:
            info = get_function_info(fnc)
            if info.name in self._ai_functions_map:
                raise ValueError(f"duplicate function name: {info.name}")

            self._ai_functions_map[info.name] = fnc

    def copy(self) -> FunctionContext:
        return FunctionContext(self._ai_functions.copy())
