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

from collections.abc import Awaitable, Callable

from ...llm.tool_context import ToolError, Toolset
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ..toolsets.tool_search import SearchStrategy, ToolSearchToolset
from .registry import SkillRegistry
from .skill import Skill


class SkillSelector(ToolSearchToolset):
    """Search-driven skill discovery with instruction hot-swapping.

    Extends :class:`ToolSearchToolset` to manage :class:`Skill` activation.
    When the LLM searches for tools, matched skills are activated and their
    instructions are returned inline so the LLM can follow skill-specific
    guidance.
    """

    def __init__(
        self,
        *,
        skills: list[Skill] | SkillRegistry,
        max_active_skills: int | None = None,
        on_change: Callable[[list[Skill]], Awaitable[None]] | None = None,
        search_strategy: NotGivenOr[SearchStrategy] = NOT_GIVEN,
        max_results: int = 5,
    ) -> None:
        if isinstance(skills, SkillRegistry):
            skill_list = skills.skills_list()
        else:
            skill_list = list(skills)

        self._skill_list = skill_list
        self._active_skills: list[Skill] = []
        self._max_active_skills = max_active_skills
        self._on_change = on_change

        # Map skill id -> Skill for resolving search results back to Skills
        self._source_to_skill: dict[str, Skill] = {s.id: s for s in skill_list}

        super().__init__(
            id="skill_selector",
            tools=skill_list,  # type: ignore[arg-type]
            max_results=max_results,
            search_strategy=search_strategy,
        )

    @property
    def active_skills(self) -> list[Skill]:
        """Return a copy of the currently active skills."""
        return list(self._active_skills)

    async def _handle_search(self, raw_arguments: dict[str, object]) -> str:
        query = str(raw_arguments.get("query", ""))
        tools = await self._search_tools(query)
        if not tools:
            raise ToolError(f"No tools found matching '{query}'.")

        matched_skills: list[Skill] = []
        for tool in tools:
            if isinstance(tool, Skill):
                matched_skills.append(tool)
            elif isinstance(tool, Toolset) and tool.id in self._source_to_skill:
                matched_skills.append(self._source_to_skill[tool.id])

        if self._max_active_skills is None:
            self._active_skills = matched_skills
        else:
            for skill in matched_skills:
                if skill in self._active_skills:
                    self._active_skills.remove(skill)
                self._active_skills.append(skill)

            while len(self._active_skills) > self._max_active_skills:
                evicted = self._active_skills.pop(0)
                logger.debug("Evicted skill: %s", evicted.name)

        self._loaded_tools = tools

        if self._on_change is not None:
            await self._on_change(list(self._active_skills))

        parts = [f"[{s.name}]: {s.instructions}" for s in matched_skills]
        skill_names = ", ".join(s.name for s in matched_skills)
        return f"Skills loaded: {skill_names}\n\n" + "\n\n".join(parts)
