"""Acceptance of a built livekit-agents-next wheel, run by CI from a clean venv with it installed.

Checks the file name's ``cp311-abi3`` tag and the ``WHEEL`` metadata, that the wheel ships
exactly the package's tracked files plus the one extension module and its bundled libraries
(resources as real audio, not LFS pointers), and that ``dir(_core)`` of the installed extension is the surface its
shipped ``_core.pyi`` declares. Prints the wheel's size record (and adds it to the GitHub step
summary). ``--no-import`` skips the ``dir(_core)`` check, for wheels the runner cannot run.

    python -I livekit-agents-next/scripts/check_wheel.py dist/livekit_agents_next-*.whl
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import zipfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
EXTENSION_SUFFIXES = (".abi3.so", ".pyd")


def check_tag(wheel: Path, archive: zipfile.ZipFile) -> list[str]:
    name, version, python, abi, platform = wheel.stem.split("-")
    errors = []
    if (python, abi) != ("cp311", "abi3"):
        errors.append(f"file name tag is {python}-{abi}, expected cp311-abi3")
    metadata = archive.read(f"{name}-{version}.dist-info/WHEEL").decode()
    if f"Tag: cp311-abi3-{platform}" not in metadata.splitlines():
        errors.append(f"WHEEL metadata does not tag cp311-abi3-{platform}")
    if "Root-Is-Purelib: false" not in metadata.splitlines():
        errors.append("WHEEL metadata does not mark the wheel platform-specific")
    return errors


def check_contents(archive: zipfile.ZipFile) -> list[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "livekit"], cwd=PACKAGE, capture_output=True, text=True, check=True
    ).stdout.split()
    # metadata, and the shared libraries auditwheel (`<dist>.libs/`) or delocate (`.dylibs/`) bundle
    shipped = {
        n
        for n in archive.namelist()
        if not n.split("/")[0].endswith((".dist-info", ".libs")) and "/.dylibs/" not in n
    }
    extensions = {n for n in shipped if n.endswith(EXTENSION_SUFFIXES)}
    errors = []
    if [Path(n).name.split(".")[0] for n in extensions] != ["_core"]:
        errors.append(f"expected one _core extension module, found {sorted(extensions)}")
    for missing in sorted(set(tracked) - shipped):
        errors.append(f"missing {missing}")
    for extra in sorted(shipped - extensions - set(tracked)):
        errors.append(f"unexpected {extra}")
    for name in sorted(shipped & set(tracked)):
        if name.endswith(".ogg") and not archive.read(name).startswith(b"OggS"):
            errors.append(f"{name} is not Ogg audio (an LFS pointer?)")
    return errors


def check_core_surface() -> list[str]:
    from livekit.agents import _core

    stub = Path(_core.__file__).with_name("_core.pyi")
    declared: set[str] = set()
    for node in ast.parse(stub.read_text()).body:
        if isinstance(node, ast.ImportFrom):
            declared.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            declared.add(node.name)
        elif isinstance(node, (ast.AnnAssign, ast.Assign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            declared.update(t.id for t in targets if isinstance(t, ast.Name))
    exported = {n for n in dir(_core) if not n.startswith("__")}
    return [f"_core.{n} is in _core.pyi but not exported" for n in sorted(declared - exported)] + [
        f"_core.{n} is exported but not in _core.pyi" for n in sorted(exported - declared)
    ]


def size_record(wheel: Path, archive: zipfile.ZipFile) -> str:
    extension = next(i for i in archive.infolist() if i.filename.endswith(EXTENSION_SUFFIXES))
    unpacked = sum(i.file_size for i in archive.infolist())
    mib = 1024 * 1024
    return (
        f"{wheel.name}: wheel {wheel.stat().st_size / mib:.1f} MiB, unpacked {unpacked / mib:.1f}"
        f" MiB, {extension.filename} {extension.file_size / mib:.1f} MiB"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--no-import", action="store_true", help="skip the dir(_core) check")
    args = parser.parse_args()
    with zipfile.ZipFile(args.wheel) as archive:
        errors = check_tag(args.wheel, archive) + check_contents(archive)
        if not args.no_import:
            errors += check_core_surface()
        record = size_record(args.wheel, archive)
    print(record)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as out:
            out.write(f"- {record}\n")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
