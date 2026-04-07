#!/usr/bin/env python3
"""Run mypy type checking on all livekit packages.

Auto-discovers all plugin packages in livekit-plugins/ and runs mypy on them.
Uses mypy's incremental mode (.mypy_cache) for fast re-checks after the first run.
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

_TYPES_MARKER = ".mypy_cache/.types_installed"


def _run_or_exit(cmd: list[str], cwd: Path, label: str) -> None:
    result = subprocess.run(cmd, capture_output=True, cwd=cwd)
    if result.returncode != 0:
        stdout = result.stdout.decode("utf-8").rstrip()
        stderr = result.stderr.decode("utf-8").rstrip()
        if stdout:
            print(stdout, file=sys.stderr)
        if stderr:
            print(stderr, file=sys.stderr)
        print(f"{label} failed (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


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


def ensure_types_installed(repo_root: Path, pkg_args: list[str]) -> None:
    """Install missing type stubs, re-running only when uv.lock has changed."""
    marker = repo_root / _TYPES_MARKER
    lock_file = repo_root / "uv.lock"

    marker_mtime = marker.stat().st_mtime if marker.exists() else 0.0
    lock_mtime = lock_file.stat().st_mtime if lock_file.exists() else 0.0

    if marker_mtime >= lock_mtime > 0:
        return

    # Ensure pip is available (required for mypy --install-types)
    _run_or_exit(["uv", "pip", "install", "pip"], repo_root, "pip install")

    # mypy --install-types installs type stubs but also runs the type checker (doubled runtime)
    # https://github.com/python/mypy/issues/10600
    _run_or_exit(
        [
            "uv",
            "run",
            "mypy",
            "--install-types",
            "--non-interactive",
            "--untyped-calls-exclude=smithy_aws_core",
            *pkg_args,
        ],
        repo_root,
        "mypy install types",
    )

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def main() -> None:
    repo_root = Path(__file__).parent.parent
    packages = get_packages(repo_root)

    pkg_args: list[str] = []
    for pkg in packages:
        pkg_args.extend(["-p", pkg])

    ensure_types_installed(repo_root, pkg_args)

    # mypy's incremental mode reads/writes .mypy_cache and skips unchanged
    # modules, making the second and subsequent runs much faster (~1s vs 30s+).
    _run_or_exit(
        ["uv", "run", "mypy", "--untyped-calls-exclude=smithy_aws_core", *pkg_args],
        repo_root,
        "mypy",
    )


if __name__ == "__main__":
    main()
