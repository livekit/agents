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

from pathlib import Path

from ...log import logger
from .loader import load_skill_from_directory
from .skill import Skill


class SkillRegistry:
    """Registry for managing available skills.

    Provides methods to register, unregister, and look up :class:`Skill`
    instances by name.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill.

        Args:
            skill: The skill to register.

        Raises:
            ValueError: If a skill with the same name is already registered.
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' is already registered")
        self._skills[skill.name] = skill
        logger.debug("Registered skill: %s", skill.name)

    def unregister(self, name: str) -> None:
        """Remove a skill from the registry.

        Args:
            name: Name of the skill to remove.

        Raises:
            KeyError: If no skill with the given name is registered.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' is not registered")
        del self._skills[name]
        logger.debug("Unregistered skill: %s", name)

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name.

        Returns:
            The skill if found, otherwise ``None``.
        """
        return self._skills.get(name)

    @property
    def available_skills(self) -> dict[str, Skill]:
        """Return a copy of the registered skills dictionary."""
        return dict(self._skills)

    def skills_list(self) -> list[Skill]:
        """Return a list of all registered skills."""
        return list(self._skills.values())

    @classmethod
    def from_directory(cls, path: str | Path) -> SkillRegistry:
        """Create a registry by scanning subdirectories for skill definitions.

        Each subdirectory containing a ``skill.md`` file will be loaded as a
        skill and registered automatically.

        Args:
            path: Root directory containing skill subdirectories.

        Returns:
            A new :class:`SkillRegistry` with all discovered skills registered.
        """
        registry = cls()
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"Skills directory not found: {root}")

        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "skill.md").is_file():
                try:
                    skill = load_skill_from_directory(child)
                    registry.register(skill)
                except Exception:
                    logger.warning("Failed to load skill from %s", child, exc_info=True)

        return registry

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={list(self._skills.keys())})"
