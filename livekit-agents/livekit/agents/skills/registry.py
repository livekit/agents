from __future__ import annotations

from pathlib import Path

from ..log import logger
from .loader import load_skill_from_directory
from .skill import Skill


class SkillRegistry:
    """Registry of available-but-not-yet-active skills.

    Skills can be registered programmatically or loaded from a directory of
    skill subdirectories.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill in the registry.

        Args:
            skill: The skill to register.

        Raises:
            ValueError: If a skill with the same name is already registered.
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' is already registered")
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> Skill:
        """Remove a skill from the registry.

        Args:
            name: Name of the skill to remove.

        Returns:
            The removed skill.

        Raises:
            KeyError: If no skill with that name is registered.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' is not registered")
        return self._skills.pop(name)

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name.

        Args:
            name: Name of the skill.

        Returns:
            The skill, or ``None`` if not found.
        """
        return self._skills.get(name)

    @property
    def available_skills(self) -> dict[str, Skill]:
        """A copy of all registered skills."""
        return self._skills.copy()

    @classmethod
    def from_directory(cls, path: str | Path) -> SkillRegistry:
        """Create a registry by scanning a directory for skill subdirectories.

        Each subdirectory must contain a ``skill.md`` file.

        Args:
            path: Path to the parent directory containing skill directories.

        Returns:
            A new :class:`SkillRegistry` populated with the discovered skills.
        """
        registry = cls()
        skills_dir = Path(path)

        if not skills_dir.is_dir():
            raise FileNotFoundError(f"Skills directory not found: {skills_dir}")

        for child in sorted(skills_dir.iterdir()):
            if not child.is_dir():
                continue
            md_path = child / "skill.md"
            if not md_path.exists():
                continue
            try:
                skill = load_skill_from_directory(child)
                registry.register(skill)
            except Exception:
                logger.exception(f"Failed to load skill from {child}")

        return registry
