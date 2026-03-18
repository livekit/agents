from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from ..llm.tool_context import Tool, find_function_tools
from ..log import logger
from .skill import Skill


def load_skill_from_directory(path: str | Path) -> Skill:
    """Load a skill from a directory containing ``skill.yaml`` and optionally ``tools.py``.

    The ``skill.yaml`` must define ``name``, ``description``, and either ``instructions``
    (inline) or ``instructions_file`` (path relative to the skill directory).

    Args:
        path: Path to the skill directory.

    Returns:
        A fully constructed :class:`Skill` instance.

    Raises:
        FileNotFoundError: If the directory or ``skill.yaml`` does not exist.
        ValueError: If required fields are missing from the YAML.
        ImportError: If ``pyyaml`` is not installed.
    """
    skill_dir = Path(path)
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

    yaml_path = skill_dir / "skill.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"skill.yaml not found in {skill_dir}")

    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml is required for file-based skill loading. Install it with: pip install pyyaml"
        ) from e

    with open(yaml_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    name = config.get("name")
    if not name:
        raise ValueError(f"skill.yaml in {skill_dir} must define 'name'")

    description = config.get("description", "")

    instructions = config.get("instructions")
    instructions_file = config.get("instructions_file")

    if instructions and instructions_file:
        raise ValueError(
            f"skill.yaml in {skill_dir} must define either 'instructions' or "
            "'instructions_file', not both"
        )

    if instructions_file:
        instr_path = skill_dir / instructions_file
        if not instr_path.exists():
            raise FileNotFoundError(f"Instructions file not found: {instr_path}")
        instructions = instr_path.read_text()
    elif not instructions:
        raise ValueError(
            f"skill.yaml in {skill_dir} must define 'instructions' or 'instructions_file'"
        )

    # load tools from tools.py if present
    tools: list[Tool] = []
    tools_path = skill_dir / "tools.py"
    if tools_path.exists():
        tools = list(_load_tools_from_file(tools_path))

    return Skill(name=name, description=description, instructions=instructions, tools=tools)


def _load_tools_from_file(path: Path) -> list[Tool]:
    """Dynamically import a Python file and extract function tools from it."""
    module_name = f"_livekit_skill_{path.parent.name}_tools"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        logger.warning(f"Could not load tools from {path}")
        return []

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        logger.exception(f"Error loading tools from {path}")
        del sys.modules[module_name]
        return []

    return list(find_function_tools(module))
