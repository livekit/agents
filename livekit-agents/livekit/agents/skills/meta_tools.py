from __future__ import annotations

from typing import TYPE_CHECKING

from ..llm.tool_context import FunctionTool, function_tool

if TYPE_CHECKING:
    from ..voice.agent import Agent


def create_skill_meta_tools(agent: Agent) -> list[FunctionTool]:
    """Create meta-tools for LLM self-activation of skills.

    Returns ``activate_skill`` and ``deactivate_skill`` function tools
    that are bound to the given agent.
    """

    @function_tool
    async def activate_skill(skill_name: str) -> str:
        """Activate a skill from the registry, adding its tools and instructions.

        Args:
            skill_name: Name of the skill to activate.
        """
        from .registry import SkillRegistry

        registry: SkillRegistry | None = getattr(agent, "_skill_registry", None)
        if registry is None:
            return "Error: No skill registry configured."

        skill = registry.get(skill_name)
        if skill is None:
            available = ", ".join(registry.available_skills.keys())
            return f"Error: Skill '{skill_name}' not found. Available: {available}"

        active_names = [s.name for s in getattr(agent, "_active_skills", [])]
        if skill_name in active_names:
            return f"Skill '{skill_name}' is already active."

        await agent.add_skill(skill)

        tool_names = [t.id for t in skill.tools]
        tool_list = ", ".join(tool_names) if tool_names else "(none)"
        return (
            f"Activated skill '{skill_name}'.\n"
            f"New tools available: {tool_list}\n"
            f"Instructions: {skill.instructions}"
        )

    @function_tool
    async def deactivate_skill(skill_name: str) -> str:
        """Deactivate an active skill, removing its tools and instructions.

        Args:
            skill_name: Name of the skill to deactivate.
        """
        active_names = [s.name for s in getattr(agent, "_active_skills", [])]
        if skill_name not in active_names:
            active = ", ".join(active_names) if active_names else "(none)"
            return f"Error: Skill '{skill_name}' is not active. Active skills: {active}"

        await agent.remove_skill(skill_name)
        return f"Deactivated skill '{skill_name}'."

    return [activate_skill, deactivate_skill]
