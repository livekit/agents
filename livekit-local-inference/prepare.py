"""Prepare one offline build projection from reviewed source and vendor artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import TYPE_CHECKING, TypeGuard


if TYPE_CHECKING:
    from collections.abc import Mapping


HERE = Path(__file__).resolve().parent
SOURCE_COMMIT = "c749f5a77488ca9bfb63b142c112c09a44349689"
PLATFORMS = ("darwin-arm64", "linux-x64-gnu", "linux-arm64-gnu")


@dataclass(frozen=True, slots=True)
class Artifact:
    package: str
    archive: str
    archive_sha256: str
    member: str
    binary_sha256: str
    notices: tuple[tuple[str, str], ...]


class Arguments(argparse.Namespace):
    artifacts: Path
    archives: Path
    platform: str
    output: Path


def _mapping(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _record(value: object) -> TypeGuard[dict[str, object]]:
    return _mapping(value) and all(isinstance(key, str) for key in value)


def _sequence(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _text(record: Mapping[str, object], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        message = f"Artifact requires a nonempty {name}"
        raise ValueError(message)
    return value


def _digest(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        message = "Artifact requires a lowercase SHA-256 identity"
        raise ValueError(message)
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _artifact(index: Path, platform: str) -> Artifact:
    value: object = json.loads(index.read_text())
    if not _record(value) or not _sequence(rows := value.get("packages")):
        message = "Artifact index requires package records"
        raise ValueError(message)
    name = f"@livekit/local-inference-{platform}"
    selected = [row for row in rows if _record(row) and row.get("name") == name]
    if len(selected) != 1:
        message = "Artifact index must select exactly one native package"
        raise ValueError(message)
    row = selected[0]
    if row.get("version") != "0.2.7" or row.get("metadata_dependencies") != {}:
        message = "Native input must be the reviewed zero-dependency 0.2.7 package"
        raise ValueError(message)
    member = _text(row, "metadata_main")
    if Path(member).name != member or not member.endswith(".node"):
        message = "Native input requires one package-relative addon"
        raise ValueError(message)
    if not _record(files := row.get("extracted_files_sha256")):
        message = "Native input requires extracted-file identities"
        raise ValueError(message)
    wrappers = [
        item for item in rows if _record(item) and item.get("name") == "@livekit/local-inference"
    ]
    if len(wrappers) != 1 or wrappers[0].get("version") != "0.2.7":
        message = "Artifact index requires the matching wrapper's license identities"
        raise ValueError(message)
    if not _record(notices := wrappers[0].get("extracted_files_sha256")):
        message = "Artifact index requires the original license identities"
        raise ValueError(message)
    return Artifact(
        name,
        Path(_text(row, "archive")).name,
        _digest(_text(row, "archive_sha256")),
        member,
        _digest(_text(files, member)),
        tuple((name, _digest(_text(notices, name))) for name in ("LICENSE", "MODEL_LICENSE")),
    )


def _copy_sources(output: Path) -> dict[str, str]:
    paths = [
        HERE / name
        for name in (
            "pyproject.toml",
            "setup.py",
            "MANIFEST.in",
            "ruff.toml",
            "pyrightconfig.json",
            "engine-artifacts.json",
            "LICENSE",
            "MODEL_LICENSE",
            "README.md",
            "native/header-provenance.json",
        )
    ]
    for directory in ("src", "native", "tests"):
        paths.extend(
            path
            for path in (HERE / directory).rglob("*")
            if path.is_file()
            and (
                path.suffix in {".py", ".pyi", ".cpp", ".h", ".md"}
                or path.name in {"py.typed", "NODE-LICENSE"}
            )
        )
    identities: dict[str, str] = {}
    for path in sorted(paths):
        if path.is_symlink():
            message = "Build projection must contain regular source files"
            raise ValueError(message)
        target = output / path.relative_to(HERE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        identities[path.relative_to(HERE).as_posix()] = _sha256(target)
    return identities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=HERE / "engine-artifacts.json")
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--platform", choices=PLATFORMS, required=True)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(namespace=Arguments())
    artifact = _artifact(args.artifacts, args.platform)
    archive = args.archives / artifact.archive
    if _sha256(archive) != artifact.archive_sha256:
        message = "Native archive differs from its reviewed identity"
        raise ValueError(message)
    for name, expected in artifact.notices:
        if _sha256(HERE / name) != expected:
            message = f"License notice differs from its reviewed identity: {name}"
            raise ValueError(message)
    args.output.mkdir(parents=True)
    files = _copy_sources(args.output)
    vendor = args.output / "src/livekit/local_inference/_vendor"
    vendor.mkdir(exist_ok=True)
    binary = vendor / "engine.node"
    with tarfile.open(archive, "r:gz") as source:
        member = source.getmember(f"package/{artifact.member}")
        if not member.isfile():
            message = "Native archive member must be a regular file"
            raise ValueError(message)
        stream = source.extractfile(member)
        if stream is None:
            message = "Native archive has no readable addon"
            raise ValueError(message)
        with stream, binary.open("xb") as destination:
            shutil.copyfileobj(stream, destination)
    if _sha256(binary) != artifact.binary_sha256:
        message = "Native addon differs from its reviewed identity"
        raise ValueError(message)
    metadata: dict[str, str] = {
        "package": artifact.package,
        "version": "0.2.7",
        "sha256": artifact.binary_sha256,
        "source_commit": SOURCE_COMMIT,
        "license": "Apache-2.0",
        "platform": args.platform,
    }
    metadata["minimum_macos" if args.platform == "darwin-arm64" else "minimum_glibc"] = (
        "11.0" if args.platform == "darwin-arm64" else "2.34"
    )
    (vendor / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (args.output / "SOURCE_MANIFEST.json").write_text(
        json.dumps(
            {
                "format": 1,
                "platform": args.platform,
                "source_files_sha256": files,
                "preparer_sha256": _sha256(Path(__file__)),
                "artifact_index_sha256": _sha256(args.artifacts),
                "archive": {"file": artifact.archive, "sha256": artifact.archive_sha256},
                "addon": {"member": artifact.member, "sha256": artifact.binary_sha256},
                "vendor_metadata_sha256": _sha256(vendor / "metadata.json"),
                "engine_built_from_source": False,
                "input_scope": "Official archived engine; source build covers host and facade only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "source": str(args.output),
                "platform": args.platform,
                "artifact_sha256": artifact.binary_sha256,
                "source_files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
