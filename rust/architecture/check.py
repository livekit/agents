"""Check resolved Cargo boundaries and tracked Rust file lengths without writing baselines."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

CORES = {
    f"lk-agents-{name}"
    for name in ("core", "voice", "worker", "inference", "storage", "a2a", "media")
}
PURE = CORES | {"lk-agents-abi"}
BANNED = {
    "async-trait",
    "async-std",
    "smol",
    "reqwest",
    "hyper",
    "hyper-util",
    "mio",
    "tungstenite",
    "livekit",
    "livekit-ffi",
    "livekit-api",
    "lk-agents-runtime",
}
GENERATED = re.compile(r"^\s*//[!/ ]*(?:@generated\b|GENERATED\b|Automatically generated\b)", re.I)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def banned(name: str) -> bool:
    return name in BANNED or any(
        name == p or name.startswith(p + "-") for p in ("tokio", "futures", "pyo3")
    )


def dependency_violations(metadata: dict) -> set[str]:
    """Inspect every resolved edge (normal, build and dev), including optional features."""
    packages = {p["id"]: p["name"] for p in metadata["packages"]}
    resolve = metadata.get("resolve")
    if not resolve or not resolve.get("nodes"):
        raise ValueError("cargo metadata has no resolved dependency graph")
    nodes = {n["id"]: [d["pkg"] for d in n["deps"]] for n in resolve["nodes"]}
    members = metadata["workspace_members"]
    if not members or not any(packages[p] in PURE for p in members):
        raise ValueError("cargo metadata contains no subject cores or ABI")
    if set(nodes) != set(packages) or any(d not in nodes for ds in nodes.values() for d in ds):
        raise ValueError("cargo metadata has an incomplete resolved graph")
    violations = set()
    for root in members:
        name = packages[root]
        if name in PURE:
            visited = {(root, name)}
            pending = [(dep, name) for dep in nodes[root]]
            while pending:
                dep, owner = pending.pop()
                # Shared legacy debt belongs to the manifest introducing it. A new pure
                # consumer of ABI must not need another waiver for ABI's existing dependency.
                owner = packages[dep] if packages[dep] in PURE else owner
                if (dep, owner) in visited:
                    continue
                visited.add((dep, owner))
                if banned(packages[dep]):
                    violations.add(f"dependency:{owner}:{packages[dep]}")
                pending.extend((child, owner) for child in nodes[dep])
        # Workspace edges also enforce layering independently of external dependencies.
        if name not in PURE | {"lk-agents-fuzz", "lk-agents-runtime", "lk-agents-py"}:
            continue
        allowed = {
            "lk-agents-abi": set(),
            "lk-agents-core": {"lk-agents-abi"},
            "lk-agents-voice": {"lk-agents-abi", "lk-agents-core"},
            "lk-agents-a2a": {"lk-agents-abi", "lk-agents-storage"},
            "lk-agents-fuzz": PURE,
            "lk-agents-runtime": PURE,
            "lk-agents-py": {"lk-agents-runtime", "lk-agents-abi"},
        }.get(name, {"lk-agents-abi"})
        for dep in nodes[root]:
            target = packages[dep]
            if target.startswith("lk-agents-") and target != name and target not in allowed:
                # Proc macros/build tooling do not define a product dependency direction.
                if target not in {"lk-agents-host-derive", "lk-agents-bindgen"}:
                    violations.add(f"direction:{name}:{target}")
    return violations


def length_violations(root: Path, paths: list[str]) -> dict[str, int]:
    violations = {}
    for path in paths:
        text = (root / path).read_text()
        # A header is outside Rust tokens. Searching later lines could match a raw string.
        if text.splitlines() and GENERATED.match(text.splitlines()[0]):
            continue
        lines = len(text.splitlines())
        if lines > 1250:
            violations[path] = lines
    return violations


def tracked_rust_paths(root: Path) -> list[str]:
    paths = git(root, "ls-files", "-z", "--", "*.rs").strip("\0").split("\0")
    deleted = set(git(root, "ls-files", "--deleted", "-z", "--", "*.rs").split("\0"))
    return [path for path in paths if path and path not in deleted]


def ratchet(actual, baseline) -> list[str]:
    """Identity sets only shrink; numeric ceilings may also decrease."""
    errors = [f"new violation: {identity}" for identity in sorted(set(actual) - set(baseline))]
    if isinstance(actual, Mapping) and isinstance(baseline, Mapping):
        errors.extend(
            f"grew: {key}: {baseline[key]} -> {actual[key]}"
            for key in sorted(set(actual) & set(baseline))
            if actual[key] > baseline[key]
        )
    return errors


def check_history(root: Path, path: str, current) -> list[str]:
    """Compare with every committed version, preventing removed identities from returning."""
    if git(root, "rev-parse", "--is-shallow-repository").strip() == "true":
        raise ValueError("baseline history requires a full checkout (fetch-depth: 0)")
    errors = []
    for revision in git(root, "rev-list", "--full-history", "HEAD", "--", path).splitlines():
        try:
            previous = json.loads(git(root, "show", f"{revision}:{path}"))
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"baseline was deleted in {revision}: {path}") from exc
        errors.extend(f"{path} versus {revision[:12]}: {e}" for e in ratchet(current, previous))
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    command = [
        "cargo",
        "metadata",
        "--manifest-path",
        str(root / "rust/Cargo.toml"),
        "--locked",
        "--all-features",
        "--format-version",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    metadata = json.loads(result.stdout)
    expected = {name for name in PURE if (root / "rust/crates" / name / "Cargo.toml").exists()}
    members = set(metadata["workspace_members"])
    found = {p["name"] for p in metadata["packages"] if p["id"] in members}
    if missing := expected - found:
        raise ValueError(f"cargo metadata omitted existing subject crates: {sorted(missing)}")
    paths = tracked_rust_paths(root)
    if not paths or paths == [""]:
        raise ValueError("no tracked Rust files found")
    errors = []
    for name, actual in (
        ("dependencies", dependency_violations(metadata)),
        ("lengths", length_violations(root, paths)),
    ):
        path = f"rust/architecture/{name}-baseline.json"
        baseline = json.loads((root / path).read_text())
        errors.extend(ratchet(actual, baseline))
        errors.extend(check_history(root, path, baseline))
        print(f"{name}: {len(actual)} existing identities; baseline {len(baseline)}")
    for name in ("export-placement", "export-metadata"):
        path = f"rust/crates/lk-agents-codegen/{name}-baseline.json"
        errors.extend(check_history(root, path, json.loads((root / path).read_text())))
    for error in errors:
        print(error, file=sys.stderr)
    return bool(errors)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"architecture: {error}")
