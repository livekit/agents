#!/usr/bin/env python3
"""Run mypy type checking on all livekit packages.

Auto-discovers all plugin packages in livekit-plugins/ and runs mypy on them.
Uses mypy's incremental mode (.mypy_cache) for fast re-checks after the first run.
Passes given arguments to mypy. Arguments after `--` passed to `uv run`.

Third-party type stubs are declared in the `typing` dependency group in
pyproject.toml and locked in uv.lock. We intentionally do NOT use
`mypy --install-types`: it requires pip, installs unpinned stubs,
and requires a forward pass in most cases.

When a dependency introduces a stub not declared yet, mypy records
the complete set of stub packages it wants in `.mypy_cache/missing_stubs`,
the same list `--install-types` consumes.
See https://github.com/python/mypy/issues/10600#issuecomment-2481074163.

We read that file for the full set, then fail with the
exact `uv add` command to declare and lock them. The script never installs stubs
itself, so every run is a single deterministic pass..
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

# mypy records the full set of stub packages it wants here, one per line.
_MISSING_STUBS = ".mypy_cache/missing_stubs"


INSTALL_STUBS_MESSAGE = """
check_types: mypy needs type stubs that aren't currently installed.

Make sure to add them to the `typing` group and lock them with:

    uv add --group typing {}
"""


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


def read_missing_stubs(repo_root: Path) -> list[str]:
    """The full stub-package list mypy recorded in .mypy_cache/missing_stubs."""
    marker = repo_root / _MISSING_STUBS
    if not marker.exists():
        return []
    return sorted(set(map(str.strip, marker.read_text().splitlines())))


def main() -> None:
    """
    Command:
        `python scripts/check_types.py --verbose -- --no-sync`
    Translates to:
        `uv run --no-sync mypy ... --verbose`
    """
    repo_root = Path(__file__).parent.parent
    packages = get_packages(repo_root)

    try:
        mypy_args_end = sys.argv.index("--")
    except ValueError:
        mypy_args, uv_run_args = sys.argv[1:], []
    else:
        mypy_args, uv_run_args = sys.argv[1:mypy_args_end], sys.argv[mypy_args_end + 1 :]

    pkg_args: list[str] = []
    for pkg in packages:
        pkg_args.extend(["-p", pkg])

    command = [
        "uv",
        "run",
        "--group",
        "typing",
        *uv_run_args,
        "mypy",
        "--untyped-calls-exclude=smithy_aws_core",
        *pkg_args,
        *mypy_args,
    ]
    print(*command, "\n")
    returncode = subprocess.run(command, cwd=repo_root).returncode

    if returncode and (missing := read_missing_stubs(repo_root)):
        print(INSTALL_STUBS_MESSAGE.format(" ".join(missing)), file=sys.stderr)

    sys.exit(returncode)


if __name__ == "__main__":
    main()
