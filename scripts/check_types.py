#!/usr/bin/env python3
"""Run strict mypy and pyright on the public API, every plugin and examples.

Auto-discovers all plugin packages in livekit-plugins/ and runs mypy on them.
Uses mypy's incremental mode (.mypy_cache) for fast re-checks after the first run.
Use --checker=mypy or --checker=pyright for one checker. `--no-sync` and arguments after `--` go
to `uv run`; other arguments go to mypy.

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

import argparse
import subprocess
import sys
from pathlib import Path

# mypy records the full set of stub packages it wants here, one per line.
_MISSING_STUBS = ".mypy_cache/missing_stubs"


INSTALL_STUBS_MESSAGE = """
check_types: mypy needs type stubs that aren't currently installed.

Make sure to add them to the `typing` group and lock them with:

    uv add --group typing {}
"""


def get_packages(repo_root: Path) -> list[str]:
    """Return all packages to type-check."""
    # tests.type_checks pins the public generics: each expected error carries a type: ignore
    packages = ["livekit.agents", "tests.type_checks"]

    plugins_dir = repo_root / "livekit-plugins"
    for plugin_dir in sorted(plugins_dir.glob("livekit-plugins-*")):
        if plugin_dir.is_dir():
            # livekit-plugins-openai -> openai
            # livekit-plugins-microsoft-ai -> microsoft_ai
            plugin_name = plugin_dir.name.replace("livekit-plugins-", "").replace("-", "_")
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", choices=("mypy", "pyright", "all"), default="all")
    # uv's, not mypy's: `--no-sync` before or after `--` keeps uv from rebuilding the extension
    parser.add_argument("--no-sync", action="store_true")
    options, mypy_args = parser.parse_known_args(mypy_args)
    if options.no_sync and "--no-sync" not in uv_run_args:
        uv_run_args.append("--no-sync")

    pkg_args: list[str] = []
    for pkg in packages:
        pkg_args.extend(["-p", pkg])

    # Every extra: the plugins it checks and the SDKs prepare_sdk_types.py inspects (sphn, riva)
    # are optional dependencies of livekit-agents-next, absent from a plain environment.
    uv = ["uv", "run", "--all-extras", "--group", "typing", *uv_run_args]
    preparation = subprocess.run(
        [*uv, "python", "scripts/prepare_sdk_types.py"], cwd=repo_root
    ).returncode
    if preparation:
        sys.exit(preparation)
    mypy = [
        *uv,
        "mypy",
        "--strict",
        # Analyze for linux regardless of the host OS, matching the CI gate.
        # Host-platform analysis breaks on Windows: the unix branch of
        # cli/readchar.py resolves against the empty win32 termios stub
        # (20 attr-defined errors) and the win32 branches demand stubs the
        # typing group intentionally doesn't carry (types-colorama,
        # types-pywin32, ...). User-supplied args come later, so an explicit
        # --platform still overrides this default.
        "--platform",
        "linux",
    ]
    returncode = 0
    if options.checker in ("mypy", "all"):
        targets = [pkg_args, ["livekit-plugins/livekit-blockguard/livekit/blockguard"]]
        # Examples are separate applications and can use the same module names.
        targets.extend(
            [str(path.relative_to(repo_root))]
            for path in sorted((repo_root / "examples").iterdir())
            if path.is_dir() or path.suffix == ".py"
        )
        for target in targets:
            command = [*mypy, *target, *mypy_args]
            print(*command, "\n", flush=True)
            result = subprocess.run(command, cwd=repo_root).returncode
            if not returncode:
                returncode = result
        if returncode and (missing := read_missing_stubs(repo_root)):
            print(INSTALL_STUBS_MESSAGE.format(" ".join(missing)), file=sys.stderr)

    if options.checker in ("pyright", "all"):
        source_paths = [
            "livekit-agents-next/livekit/agents",
            "tests/type_checks",
            "examples",
            "livekit-plugins/livekit-blockguard/livekit/blockguard",
        ]
        source_paths.extend(
            str(path.relative_to(repo_root))
            for path in sorted((repo_root / "livekit-plugins").glob("livekit-plugins-*/livekit"))
        )
        pyright = [
            *uv,
            "pyright",
            "--pythonpath",
            sys.executable,
            *source_paths,
        ]
        print(*pyright, "\n", flush=True)
        pyright_returncode = subprocess.run(pyright, cwd=repo_root).returncode
        if not returncode:
            returncode = pyright_returncode

    sys.exit(returncode)


if __name__ == "__main__":
    main()
