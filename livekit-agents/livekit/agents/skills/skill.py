from __future__ import annotations

from ..llm.tool_context import FunctionTool, RawFunctionTool, Tool, Toolset, find_function_tools


class Skill(Toolset):
    """A reusable bundle of instructions and tools that can be added to an agent.

    Can be subclassed with ``@function_tool`` decorated methods, or instantiated directly
    with a list of tools.

    Args:
        name: Unique name for the skill.
        description: Short description of what the skill does.
        instructions: Instructions to inject into the agent when this skill is active.
        tools: Optional list of tools to include with this skill.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        instructions: str,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(id=f"skill_{name}")
        self._name = name
        self._description = description
        self._instructions = instructions
        self._skill_tools: list[FunctionTool | RawFunctionTool] = []

        if tools:
            for tool in tools:
                if isinstance(tool, (FunctionTool, RawFunctionTool)):
                    self._skill_tools.append(tool)

        # discover @function_tool decorated methods on subclasses
        for tool in find_function_tools(self):
            if tool not in self._skill_tools:
                self._skill_tools.append(tool)

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
    def tools(self) -> list[Tool]:
        return list(self._skill_tools)

    def __repr__(self) -> str:
        return f"Skill(name={self._name!r}, tools={len(self._skill_tools)})"
