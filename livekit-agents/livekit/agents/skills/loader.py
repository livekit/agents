from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from ..llm.tool_context import Tool, find_function_tools
from ..log import logger
from .skill import Skill

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FIELD_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)


def _parse_skill_md(text: str) -> tuple[dict[str, str], str]:
    """Parse a skill.md file into (frontmatter fields, body instructions).

    Raises:
        ValueError: If the file has no valid frontmatter block.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("skill.md must start with a --- frontmatter block")

    fields = dict(_FIELD_RE.findall(match.group(1)))
    body = text[match.end() :].strip()
    return fields, body


def load_skill_from_directory(path: str | Path) -> Skill:
    """Load a skill from a directory containing ``skill.md`` and optionally ``tools.py``.

    The ``skill.md`` uses markdown with frontmatter::

        ---
        name: calendar
        description: Manage calendar events and scheduling
        ---

        You can help users manage their calendar. Use the available
        calendar tools to book, check, and cancel meetings.

    Args:
        path: Path to the skill directory.

    Returns:
        A fully constructed :class:`Skill` instance.

    Raises:
        FileNotFoundError: If the directory or ``skill.md`` does not exist.
        ValueError: If required fields are missing.
    """
    skill_dir = Path(path)
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

    md_path = skill_dir / "skill.md"
    if not md_path.exists():
        raise FileNotFoundError(f"skill.md not found in {skill_dir}")

    text = md_path.read_text()
    fields, instructions = _parse_skill_md(text)

    name = fields.get("name")
    if not name:
        raise ValueError(f"skill.md in {skill_dir} must define 'name' in frontmatter")

    description = fields.get("description", "")

    if not instructions:
        raise ValueError(f"skill.md in {skill_dir} must have instructions after the frontmatter")

    # load tools from tools.py if present
    tools: list[Tool] = []
    tools_path = skill_dir / "tools.py"
    if tools_path.exists():
        tools = list(_load_tools_from_file(tools_path))

    return Skill(name=name, description=description, instructions=instructions, tools=tools)


def _load_tools_from_file(path: Path) -> list[Tool]:
    """Dynamically import a Python file and extract function tools from it."""
    module_name = f"_livekit_skill_{path.parent.name}_{id(path)}_tools"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        logger.warning(f"Could not load tools from {path}")
        return []

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.exception("Error loading tools from %s", path)
        sys.modules.pop(module_name, None)
        raise ImportError(f"Error loading tools from {path}") from e

    return list(find_function_tools(module))
