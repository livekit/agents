"""Tests for the Skill class in beta/skills/."""

from __future__ import annotations

import pytest

from livekit.agents.beta.skills import Skill
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool, function_tool


@function_tool
async def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@function_tool
async def farewell(name: str) -> str:
    """Say goodbye to someone."""
    return f"Goodbye, {name}!"


@function_tool
async def weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


class TestSkillCreation:
    def test_create_skill_with_tools(self) -> None:
        skill = Skill(
            name="greeter",
            description="A greeting skill",
            instructions="You greet people warmly.",
            tools=[greet],
        )
        assert skill.name == "greeter"
        assert skill.description == "A greeting skill"
        assert skill.instructions == "You greet people warmly."
        assert skill.id == "skill_greeter"
        assert len(skill.tools) == 1

    def test_create_skill_with_no_tools(self) -> None:
        skill = Skill(
            name="empty",
            description="An empty skill",
            instructions="Do nothing.",
        )
        assert skill.name == "empty"
        assert skill.description == "An empty skill"
        assert skill.instructions == "Do nothing."
        assert skill.id == "skill_empty"
        assert len(skill.tools) == 0

    def test_create_skill_with_multiple_tools(self) -> None:
        skill = Skill(
            name="multi",
            description="A multi-tool skill",
            instructions="Use multiple tools.",
            tools=[greet, farewell, weather],
        )
        assert len(skill.tools) == 3
        tool_names = [t.info.name for t in skill.tools]
        assert "greet" in tool_names
        assert "farewell" in tool_names
        assert "weather" in tool_names

    def test_tools_are_function_tools(self) -> None:
        skill = Skill(
            name="typed",
            description="Typed skill",
            instructions="Check types.",
            tools=[greet],
        )
        for tool in skill.tools:
            assert isinstance(tool, (FunctionTool, RawFunctionTool))


class TestSkillRepr:
    def test_repr_with_tools(self) -> None:
        skill = Skill(
            name="greeter",
            description="A greeting skill",
            instructions="You greet people warmly.",
            tools=[greet, farewell],
        )
        r = repr(skill)
        assert "greeter" in r
        assert "2" in r

    def test_repr_no_tools(self) -> None:
        skill = Skill(
            name="empty",
            description="An empty skill",
            instructions="Do nothing.",
        )
        r = repr(skill)
        assert "empty" in r
        assert "0" in r


class TestSkillSubclassing:
    def test_subclass_with_function_tool_methods(self) -> None:
        class GreeterSkill(Skill):
            def __init__(self) -> None:
                super().__init__(
                    name="greeter",
                    description="Greets people",
                    instructions="Be friendly.",
                )

            @function_tool
            async def say_hello(self, name: str) -> str:
                """Say hello."""
                return f"Hello, {name}!"

            @function_tool
            async def say_hi(self, name: str) -> str:
                """Say hi."""
                return f"Hi, {name}!"

        skill = GreeterSkill()
        assert len(skill.tools) == 2
        tool_names = [t.info.name for t in skill.tools]
        assert "say_hello" in tool_names
        assert "say_hi" in tool_names

    def test_subclass_with_explicit_and_method_tools(self) -> None:
        class ComboSkill(Skill):
            def __init__(self) -> None:
                super().__init__(
                    name="combo",
                    description="Combo skill",
                    instructions="Use everything.",
                    tools=[greet],
                )

            @function_tool
            async def custom_tool(self, x: int) -> int:
                """Double a number."""
                return x * 2

        skill = ComboSkill()
        assert len(skill.tools) >= 2
        tool_names = [t.info.name for t in skill.tools]
        assert "greet" in tool_names
        assert "custom_tool" in tool_names


class TestSkillProperties:
    def test_properties_are_readonly(self) -> None:
        skill = Skill(
            name="test",
            description="Test skill",
            instructions="Test instructions.",
        )
        with pytest.raises(AttributeError):
            skill.name = "new_name"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            skill.description = "new_desc"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            skill.instructions = "new_inst"  # type: ignore[misc]
