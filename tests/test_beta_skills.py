"""Tests for the Skill class in beta/skills/."""

from __future__ import annotations

from pathlib import Path

import pytest

from livekit.agents.beta.skills import (
    Skill,
    SkillRegistry,
    SkillSelector,
    load_skill_from_directory,
)
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool, ToolError, function_tool


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


# ---------------------------------------------------------------------------
# SkillRegistry tests
# ---------------------------------------------------------------------------


class TestSkillRegistry:
    def _make_skill(self, name: str = "test") -> Skill:
        return Skill(name=name, description=f"{name} desc", instructions=f"{name} instructions")

    def test_register_and_get(self) -> None:
        registry = SkillRegistry()
        skill = self._make_skill("alpha")
        registry.register(skill)
        assert registry.get("alpha") is skill

    def test_get_missing_returns_none(self) -> None:
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_duplicate_register_raises(self) -> None:
        registry = SkillRegistry()
        registry.register(self._make_skill("dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(self._make_skill("dup"))

    def test_unregister(self) -> None:
        registry = SkillRegistry()
        registry.register(self._make_skill("removable"))
        registry.unregister("removable")
        assert registry.get("removable") is None

    def test_unregister_missing_raises(self) -> None:
        registry = SkillRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("ghost")

    def test_available_skills_is_copy(self) -> None:
        registry = SkillRegistry()
        skill = self._make_skill("orig")
        registry.register(skill)
        copy = registry.available_skills
        copy["injected"] = self._make_skill("injected")
        assert registry.get("injected") is None

    def test_skills_list(self) -> None:
        registry = SkillRegistry()
        registry.register(self._make_skill("a"))
        registry.register(self._make_skill("b"))
        lst = registry.skills_list()
        assert isinstance(lst, list)
        assert len(lst) == 2
        names = {s.name for s in lst}
        assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

_SKILL_MD_TEMPLATE = """\
---
name: {name}
description: {description}
---

{body}
"""

_TOOLS_PY_CONTENT = """\
from livekit.agents.llm.tool_context import function_tool

@function_tool
async def sample_tool(x: int) -> int:
    \"\"\"Double a number.\"\"\"
    return x * 2
"""


class TestLoader:
    def test_load_from_directory(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "myskill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            _SKILL_MD_TEMPLATE.format(name="myskill", description="My skill", body="Do the thing.")
        )
        skill = load_skill_from_directory(skill_dir)
        assert skill.name == "myskill"
        assert skill.description == "My skill"
        assert skill.instructions == "Do the thing."
        assert len(skill.tools) == 0

    def test_load_with_tools_py(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "tooled"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            _SKILL_MD_TEMPLATE.format(
                name="tooled", description="Tooled skill", body="Use the tool."
            )
        )
        (skill_dir / "tools.py").write_text(_TOOLS_PY_CONTENT)
        skill = load_skill_from_directory(skill_dir)
        assert skill.name == "tooled"
        assert len(skill.tools) >= 1
        tool_names = [t.info.name for t in skill.tools]
        assert "sample_tool" in tool_names

    def test_missing_directory_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_skill_from_directory("/nonexistent/path/to/skill")

    def test_missing_skill_md_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "empty_dir"
        skill_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="skill.md"):
            load_skill_from_directory(skill_dir)

    def test_no_frontmatter_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "no_fm"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("Just some text without frontmatter.")
        with pytest.raises(ValueError, match="frontmatter"):
            load_skill_from_directory(skill_dir)

    def test_empty_body_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "nobody"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: nobody\ndescription: empty\n---\n")
        with pytest.raises(ValueError, match="empty"):
            load_skill_from_directory(skill_dir)

    def test_registry_from_directory(self, tmp_path: Path) -> None:
        for sname in ("alpha", "beta"):
            d = tmp_path / sname
            d.mkdir()
            (d / "skill.md").write_text(
                _SKILL_MD_TEMPLATE.format(
                    name=sname, description=f"{sname} desc", body=f"{sname} instructions."
                )
            )
        registry = SkillRegistry.from_directory(tmp_path)
        assert registry.get("alpha") is not None
        assert registry.get("beta") is not None
        assert len(registry.skills_list()) == 2

    def test_registry_from_directory_skips_invalid(self, tmp_path: Path) -> None:
        # Valid skill
        good = tmp_path / "good"
        good.mkdir()
        (good / "skill.md").write_text(
            _SKILL_MD_TEMPLATE.format(name="good", description="Good", body="Good instructions.")
        )
        # Invalid skill (no frontmatter)
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "skill.md").write_text("no frontmatter here")
        registry = SkillRegistry.from_directory(tmp_path)
        assert registry.get("good") is not None
        assert registry.get("bad") is None
        assert len(registry.skills_list()) == 1


# ---------------------------------------------------------------------------
# SkillSelector tests
# ---------------------------------------------------------------------------



def _make_weather_skill() -> Skill:
    return Skill(
        name="weather",
        description="Get weather information",
        instructions="Use the weather tool to check forecasts.",
        tools=[weather],
    )


def _make_calendar_skill() -> Skill:
    @function_tool
    async def schedule_event(title: str, date: str) -> str:
        """Schedule a calendar event."""
        return f"Scheduled {title} on {date}"

    return Skill(
        name="calendar",
        description="Manage calendar events",
        instructions="Use calendar tools to schedule and manage events.",
        tools=[schedule_event],
    )


class TestSkillSelector:
    def test_create_skill_selector(self) -> None:
        ws = _make_weather_skill()
        cs = _make_calendar_skill()
        selector = SkillSelector(skills=[ws, cs])
        assert selector.active_skills == []

    def test_create_from_registry(self) -> None:
        registry = SkillRegistry()
        registry.register(_make_weather_skill())
        registry.register(_make_calendar_skill())
        selector = SkillSelector(skills=registry)
        assert selector.active_skills == []

    @pytest.mark.asyncio
    async def test_search_activates_skill(self) -> None:
        ws = _make_weather_skill()
        cs = _make_calendar_skill()
        selector = SkillSelector(skills=[ws, cs])
        await selector.setup()

        await selector._handle_search({"query": "weather"})
        active = selector.active_skills
        assert any(s.name == "weather" for s in active)

    @pytest.mark.asyncio
    async def test_search_replaces_skills_by_default(self) -> None:
        ws = _make_weather_skill()
        cs = _make_calendar_skill()
        selector = SkillSelector(skills=[ws, cs])
        await selector.setup()

        await selector._handle_search({"query": "weather"})
        assert any(s.name == "weather" for s in selector.active_skills)

        await selector._handle_search({"query": "calendar schedule"})
        active_names = [s.name for s in selector.active_skills]
        assert "calendar" in active_names
        assert "weather" not in active_names

    @pytest.mark.asyncio
    async def test_max_active_skills_accumulate(self) -> None:
        ws = _make_weather_skill()
        cs = _make_calendar_skill()
        selector = SkillSelector(skills=[ws, cs], max_active_skills=5)
        await selector.setup()

        await selector._handle_search({"query": "weather"})
        await selector._handle_search({"query": "calendar schedule"})
        active_names = [s.name for s in selector.active_skills]
        assert "weather" in active_names
        assert "calendar" in active_names

    @pytest.mark.asyncio
    async def test_max_active_skills_eviction(self) -> None:
        ws = _make_weather_skill()
        cs = _make_calendar_skill()
        selector = SkillSelector(skills=[ws, cs], max_active_skills=1)
        await selector.setup()

        await selector._handle_search({"query": "weather"})
        assert len(selector.active_skills) == 1
        assert selector.active_skills[0].name == "weather"

        await selector._handle_search({"query": "calendar schedule"})
        assert len(selector.active_skills) == 1
        assert selector.active_skills[0].name == "calendar"

    @pytest.mark.asyncio
    async def test_on_change_callback_fires(self) -> None:
        ws = _make_weather_skill()
        callback_results: list[list[Skill]] = []

        async def on_change(skills: list[Skill]) -> None:
            callback_results.append(skills)

        selector = SkillSelector(skills=[ws], on_change=on_change)
        await selector.setup()

        await selector._handle_search({"query": "weather"})
        assert len(callback_results) == 1
        assert any(s.name == "weather" for s in callback_results[0])

    @pytest.mark.asyncio
    async def test_on_change_not_called_when_no_match(self) -> None:
        ws = _make_weather_skill()
        callback_results: list[list[Skill]] = []

        async def on_change(skills: list[Skill]) -> None:
            callback_results.append(skills)

        selector = SkillSelector(skills=[ws], on_change=on_change)
        await selector.setup()

        with pytest.raises(ToolError):
            await selector._handle_search({"query": "xyznonexistent123"})
        assert len(callback_results) == 0

    @pytest.mark.asyncio
    async def test_search_result_includes_instructions(self) -> None:
        ws = _make_weather_skill()
        selector = SkillSelector(skills=[ws])
        await selector.setup()

        result = await selector._handle_search({"query": "weather"})
        assert "Use the weather tool to check forecasts." in result
        assert "weather" in result

    @pytest.mark.asyncio
    async def test_tools_property_includes_search_tool(self) -> None:
        ws = _make_weather_skill()
        selector = SkillSelector(skills=[ws])
        await selector.setup()

        tools = selector.tools
        tool_names = []
        for t in tools:
            if hasattr(t, "info"):
                tool_names.append(t.info.name)
            elif hasattr(t, "id"):
                tool_names.append(t.id)
        assert "tool_search" in tool_names
