from __future__ import annotations

import os
import tempfile

import pytest

from livekit.agents import Agent
from livekit.agents.llm import function_tool
from livekit.agents.skills import Skill, SkillRegistry, load_skill_from_directory

# -- Helpers --


@function_tool
async def weather_tool(location: str) -> str:
    """Get weather for a location.

    Args:
        location: The city name.
    """
    return f"Sunny in {location}"


@function_tool
async def calendar_tool(date: str) -> str:
    """Check calendar for a date.

    Args:
        date: The date string.
    """
    return f"No events on {date}"


class WeatherSkill(Skill):
    def __init__(self) -> None:
        super().__init__(
            name="weather",
            description="Get weather information",
            instructions="You can check weather for any city.",
        )

    @function_tool
    async def get_forecast(self, city: str) -> str:
        """Get a weather forecast.

        Args:
            city: The city name.
        """
        return f"Forecast for {city}: sunny"


# -- Skill tests --


class TestSkill:
    def test_create_skill_directly(self):
        skill = Skill(
            name="test",
            description="A test skill",
            instructions="Test instructions",
            tools=[weather_tool],
        )
        assert skill.name == "test"
        assert skill.description == "A test skill"
        assert skill.instructions == "Test instructions"
        assert skill.id == "skill_test"
        assert len(skill.tools) == 1
        assert skill.tools[0].id == "weather_tool"

    def test_create_skill_no_tools(self):
        skill = Skill(
            name="empty",
            description="No tools",
            instructions="Just instructions",
        )
        assert skill.name == "empty"
        assert len(skill.tools) == 0

    def test_create_skill_subclass(self):
        skill = WeatherSkill()
        assert skill.name == "weather"
        assert len(skill.tools) == 1
        assert skill.tools[0].id == "get_forecast"

    def test_skill_repr(self):
        skill = Skill(
            name="test",
            description="desc",
            instructions="instr",
            tools=[weather_tool],
        )
        assert "test" in repr(skill)

    def test_skill_with_multiple_tools(self):
        skill = Skill(
            name="multi",
            description="Multiple tools",
            instructions="Use these tools.",
            tools=[weather_tool, calendar_tool],
        )
        assert len(skill.tools) == 2
        tool_ids = {t.id for t in skill.tools}
        assert "weather_tool" in tool_ids
        assert "calendar_tool" in tool_ids


# -- SkillRegistry tests --


class TestSkillRegistry:
    def test_register_and_get(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="d", instructions="i")
        registry.register(skill)
        assert registry.get("test") is skill
        assert "test" in registry.available_skills

    def test_register_duplicate_raises(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="d", instructions="i")
        registry.register(skill)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(skill)

    def test_unregister(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="d", instructions="i")
        registry.register(skill)
        removed = registry.unregister("test")
        assert removed is skill
        assert registry.get("test") is None

    def test_unregister_missing_raises(self):
        registry = SkillRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("nonexistent")

    def test_get_missing_returns_none(self):
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_available_skills_is_copy(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="d", instructions="i")
        registry.register(skill)
        available = registry.available_skills
        available.clear()
        assert len(registry.available_skills) == 1


# -- File-based loader tests --


class TestLoader:
    def test_load_skill_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = (
                "---\n"
                "name: test_skill\n"
                "description: A test skill\n"
                "---\n\n"
                "These are test instructions.\n"
            )
            with open(os.path.join(tmpdir, "skill.md"), "w") as f:
                f.write(md_content)

            skill = load_skill_from_directory(tmpdir)
            assert skill.name == "test_skill"
            assert skill.description == "A test skill"
            assert "test instructions" in skill.instructions.lower()

    def test_load_skill_with_tools_py(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = (
                "---\n"
                "name: with_tools\n"
                "description: Has tools\n"
                "---\n\n"
                "Use the tool.\n"
            )
            with open(os.path.join(tmpdir, "skill.md"), "w") as f:
                f.write(md_content)

            tools_py = (
                "from livekit.agents.llm import function_tool\n\n"
                "@function_tool\n"
                "async def my_loaded_tool(x: str) -> str:\n"
                '    """A dynamically loaded tool.\n\n'
                "    Args:\n"
                "        x: Input value.\n"
                '    """\n'
                "    return x\n"
            )
            with open(os.path.join(tmpdir, "tools.py"), "w") as f:
                f.write(tools_py)

            skill = load_skill_from_directory(tmpdir)
            assert len(skill.tools) == 1
            assert skill.tools[0].id == "my_loaded_tool"

    def test_load_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            load_skill_from_directory("/nonexistent/path")

    def test_load_missing_md_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="skill.md"):
                load_skill_from_directory(tmpdir)

    def test_load_missing_name_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = "---\ndescription: No name\n---\n\nSome instructions.\n"
            with open(os.path.join(tmpdir, "skill.md"), "w") as f:
                f.write(md_content)
            with pytest.raises(ValueError, match="name"):
                load_skill_from_directory(tmpdir)

    def test_load_no_frontmatter_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "skill.md"), "w") as f:
                f.write("Just plain text, no frontmatter.\n")
            with pytest.raises(ValueError, match="frontmatter"):
                load_skill_from_directory(tmpdir)

    def test_load_empty_body_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = "---\nname: empty_body\ndescription: d\n---\n"
            with open(os.path.join(tmpdir, "skill.md"), "w") as f:
                f.write(md_content)
            with pytest.raises(ValueError, match="instructions"):
                load_skill_from_directory(tmpdir)

    def test_registry_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # create two skill subdirectories
            for name in ["skill_a", "skill_b"]:
                skill_dir = os.path.join(tmpdir, name)
                os.makedirs(skill_dir)
                md_content = (
                    f"---\nname: {name}\ndescription: Skill {name}\n---\n\n"
                    f"Instructions for {name}.\n"
                )
                with open(os.path.join(skill_dir, "skill.md"), "w") as f:
                    f.write(md_content)

            registry = SkillRegistry.from_directory(tmpdir)
            assert "skill_a" in registry.available_skills
            assert "skill_b" in registry.available_skills


# -- Agent integration tests --


class TestAgentSkillIntegration:
    def test_agent_with_initial_skills(self):
        skill = Skill(
            name="weather",
            description="Weather info",
            instructions="Check weather.",
            tools=[weather_tool],
        )
        agent = Agent(
            instructions="You are a helpful assistant.",
            skills=[skill],
        )
        assert "Active Skills" in str(agent.instructions)
        assert "weather" in str(agent.instructions)
        assert "Check weather." in str(agent.instructions)
        # the skill's tool should be accessible through the agent's tools
        tool_ids = {t.id for t in agent.tools}
        assert "skill_weather" in tool_ids

    def test_agent_with_registry(self):
        registry = SkillRegistry()
        skill = Skill(name="test", description="A test", instructions="Test it.")
        registry.register(skill)

        agent = Agent(
            instructions="Base instructions.",
            skill_registry=registry,
        )
        # should have meta-tools
        tool_ids = {t.id for t in agent.tools}
        assert "activate_skill" in tool_ids
        assert "deactivate_skill" in tool_ids
        # should mention available skills
        assert "Available Skills" in str(agent.instructions)
        assert "test" in str(agent.instructions)

    async def test_add_and_remove_skill(self):
        skill = Skill(
            name="calendar",
            description="Calendar management",
            instructions="Manage calendar events.",
            tools=[calendar_tool],
        )
        agent = Agent(instructions="Base instructions.")

        await agent.add_skill(skill)
        assert "calendar" in str(agent.instructions)
        assert "Manage calendar events." in str(agent.instructions)
        tool_ids = {t.id for t in agent.tools}
        assert "skill_calendar" in tool_ids

        await agent.remove_skill("calendar")
        assert "calendar" not in str(agent.instructions)
        # after removal, the skill toolset should be gone
        tool_ids = {t.id for t in agent.tools}
        assert "skill_calendar" not in tool_ids

    async def test_add_skill_by_name_from_registry(self):
        registry = SkillRegistry()
        skill = Skill(name="lookup", description="Look up", instructions="Look things up.")
        registry.register(skill)

        agent = Agent(instructions="Base.", skill_registry=registry)
        await agent.add_skill("lookup")
        assert "lookup" in str(agent.instructions)
        assert any(s.name == "lookup" for s in agent._active_skills)

    async def test_add_skill_by_name_without_registry_raises(self):
        agent = Agent(instructions="Base.")
        with pytest.raises(ValueError, match="no skill_registry"):
            await agent.add_skill("nonexistent")

    async def test_add_skill_by_name_not_found_raises(self):
        registry = SkillRegistry()
        agent = Agent(instructions="Base.", skill_registry=registry)
        with pytest.raises(ValueError, match="not found"):
            await agent.add_skill("nonexistent")

    async def test_remove_nonexistent_skill_raises(self):
        agent = Agent(instructions="Base.")
        with pytest.raises(KeyError, match="not active"):
            await agent.remove_skill("nonexistent")

    async def test_add_duplicate_skill_is_noop(self):
        skill = Skill(name="dup", description="d", instructions="i")
        agent = Agent(instructions="Base.")
        await agent.add_skill(skill)
        await agent.add_skill(skill)  # should not raise
        assert len(agent._active_skills) == 1

    def test_agent_no_skills(self):
        agent = Agent(instructions="Simple agent.")
        assert "Active Skills" not in str(agent.instructions)
        assert "Available Skills" not in str(agent.instructions)

    def test_compose_instructions_preserves_base(self):
        agent = Agent(
            instructions="Base instructions.",
            skills=[
                Skill(name="s1", description="d", instructions="Skill 1 instructions."),
            ],
        )
        # base instructions should still be at the start
        assert str(agent.instructions).startswith("Base instructions.")


# -- Meta-tools tests --


class TestMetaTools:
    async def test_activate_skill_tool(self):
        registry = SkillRegistry()
        skill = Skill(
            name="activatable",
            description="Can be activated",
            instructions="Activated skill instructions.",
        )
        registry.register(skill)

        agent = Agent(instructions="Base.", skill_registry=registry)

        # find the activate_skill tool
        activate = None
        for t in agent.tools:
            if t.id == "activate_skill":
                activate = t
                break

        assert activate is not None
        result = await activate(skill_name="activatable")
        assert "Activated skill 'activatable'" in result
        assert "Activated skill instructions." in result
        assert any(s.name == "activatable" for s in agent._active_skills)

    async def test_deactivate_skill_tool(self):
        registry = SkillRegistry()
        skill = Skill(name="removable", description="d", instructions="i")
        registry.register(skill)

        agent = Agent(instructions="Base.", skill_registry=registry)
        await agent.add_skill(skill)

        # find the deactivate_skill tool
        deactivate = None
        for t in agent.tools:
            if t.id == "deactivate_skill":
                deactivate = t
                break

        assert deactivate is not None
        result = await deactivate(skill_name="removable")
        assert "Deactivated skill 'removable'" in result
        assert not any(s.name == "removable" for s in agent._active_skills)

    async def test_activate_nonexistent_skill(self):
        registry = SkillRegistry()
        agent = Agent(instructions="Base.", skill_registry=registry)

        activate = None
        for t in agent.tools:
            if t.id == "activate_skill":
                activate = t
                break

        assert activate is not None
        result = await activate(skill_name="nonexistent")
        assert "Error" in result
        assert "not found" in result

    async def test_deactivate_inactive_skill(self):
        registry = SkillRegistry()
        agent = Agent(instructions="Base.", skill_registry=registry)

        deactivate = None
        for t in agent.tools:
            if t.id == "deactivate_skill":
                deactivate = t
                break

        assert deactivate is not None
        result = await deactivate(skill_name="nonexistent")
        assert "Error" in result
        assert "not active" in result
