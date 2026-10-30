"""Python-only adapter for ``python -m livekit.agents start agent.py``."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from . import _core
from ._generated.cli import CliParseResultDisplay


def main(argv: list[str] | None = None) -> int:
    """Run the entrypoint's normal ``__main__`` block with its command arguments."""
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="python -m livekit.agents")
    parser.add_argument("command", help="agent command, such as start or console")
    parser.add_argument(
        "entrypoint", type=Path, help="Python entrypoint calling cli.run_app(server)"
    )
    invocation = parser.parse_args(argv[:2])
    entrypoint: Path = invocation.entrypoint
    command = [invocation.command, *argv[2:]]

    # Let the shared CLI handle diagnostics before executing application code.
    parsed = _core.parse_cli([str(entrypoint), *command])
    if isinstance(parsed, CliParseResultDisplay):
        print(parsed.message, file=sys.stderr if parsed.stderr else sys.stdout, end="")
        return parsed.exit_code
    if not entrypoint.is_file():
        parser.error(f"entrypoint is not a file: {entrypoint}")

    previous_argv, previous_path = sys.argv, sys.path.copy()
    try:
        sys.argv = [str(entrypoint), *command]
        sys.path.insert(0, str(entrypoint.resolve().parent))
        runpy.run_path(str(entrypoint), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
    return 0


if __name__ == "__main__":
    sys.exit(main())
