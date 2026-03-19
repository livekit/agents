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

import importlib.util
import re
import sys
from pathlib import Path

from ...llm.tool_context import Tool, Toolset, find_function_tools
from ...log import logger
from .skill import Skill

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FIELD_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)


def _load_tools_from_file(path: Path) -> list[Tool]:
    """Dynamically import a tools.py file and return all function tools found in it."""
    module_name = f"_skill_tools_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[module_name]
        raise ImportError(f"Failed to load tools from {path}: {exc}") from exc

    return list(find_function_tools(module))


def load_skill_from_directory(path: str | Path) -> Skill:
    """Load a Skill from a directory containing a ``skill.md`` file.

    The ``skill.md`` file must have YAML-like frontmatter with at least a
    ``name`` and ``description`` field, followed by instructions in the body.

    An optional ``tools.py`` in the same directory will be imported and any
    ``@function_tool``-decorated functions will be attached to the skill.

    Args:
        path: Path to the skill directory.

    Returns:
        A fully constructed :class:`Skill` instance.

    Raises:
        FileNotFoundError: If the directory or ``skill.md`` does not exist.
        ValueError: If frontmatter is missing/invalid or the body is empty.
        ImportError: If ``tools.py`` exists but fails to load.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {directory}")

    skill_md = directory / "skill.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"skill.md not found in {directory}")

    content = skill_md.read_text(encoding="utf-8")

    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise ValueError(f"No valid frontmatter found in {skill_md}")

    frontmatter_text = match.group(1)
    fields = dict(_FIELD_RE.findall(frontmatter_text))

    if "name" not in fields:
        raise ValueError(f"Frontmatter missing required 'name' field in {skill_md}")

    name = fields["name"].strip()
    description = fields.get("description", "").strip()

    body = content[match.end() :].strip()
    if not body:
        raise ValueError(f"skill.md body (instructions) is empty in {skill_md}")

    tools: list[Tool | Toolset] = []
    tools_py = directory / "tools.py"
    if tools_py.is_file():
        logger.debug("Loading tools from %s", tools_py)
        tools = list(_load_tools_from_file(tools_py))

    return Skill(name=name, description=description, instructions=body, tools=tools)
