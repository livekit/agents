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

from collections.abc import Sequence

from ...llm.tool_context import FunctionTool, RawFunctionTool, Tool, Toolset


class Skill(Toolset):
    """A Skill is a Toolset that bundles instructions with tools.

    Skills are the building blocks for composable agent capabilities. Each skill
    provides a name, description, instructions (system prompt fragment), and a
    set of function tools.

    Tools can be provided explicitly via the constructor or discovered
    automatically from ``@function_tool``-decorated methods on subclasses.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        instructions: str,
        tools: list[Tool] | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._instructions = instructions
        super().__init__(id=f"skill_{name}", tools=tools)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def tools(self) -> Sequence[FunctionTool | RawFunctionTool]:
        """Return only FunctionTool and RawFunctionTool instances."""
        return [t for t in super().tools if isinstance(t, (FunctionTool, RawFunctionTool))]

    def __repr__(self) -> str:
        return f"Skill(name={self._name!r}, tools={len(self.tools)})"
