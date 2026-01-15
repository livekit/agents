#!/usr/bin/env python3
"""Run mypy type checking on all livekit packages.

Auto-discovers all plugin packages in livekit-plugins/ and runs mypy on them.
"""

import subprocess
import sys
from pathlib import Path

# Plugins to exclude from type checking
EXCLUDED_PLUGINS = [
    "browser",
    "nvidia",
    "rtzr",
]


def get_packages(repo_root: Path) -> list[str]:
    """Return all packages to type-check."""
    packages = ["livekit.agents"]

    plugins_dir = repo_root / "livekit-plugins"
    for plugin_dir in sorted(plugins_dir.glob("livekit-plugins-*")):
        if plugin_dir.is_dir():
            # livekit-plugins-openai -> openai
            # livekit-plugins-turn-detector -> turn_detector
            plugin_name = plugin_dir.name.replace("livekit-plugins-", "").replace("-", "_")
            if plugin_name in EXCLUDED_PLUGINS:
                continue
            packages.append(f"livekit.plugins.{plugin_name}")

    return packages


def main() -> int:
    repo_root = Path(__file__).parent.parent
    packages = get_packages(repo_root)

    # Build mypy command
    cmd = [
        "uv",
        "run",
        "mypy",
        "--install-types",
        "--non-interactive",
        "--untyped-calls-exclude=smithy_aws_core",
    ]
    for pkg in packages:
        cmd.extend(["-p", pkg])

    # Ensure pip is available (required for mypy --install-types)
    subprocess.run(
        ["uv", "pip", "install", "pip"],
        capture_output=True,
        cwd=repo_root,
    )

    # Run mypy
    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
