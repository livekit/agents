"""Check Cargo boundaries, Rust sources, compiler lints and shrink-only baselines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

CARGO_RUNNER = os.environ.get("CARGO_RUNNER", "cargo")

CORES = {f"lk-agents-{name}" for name in ("core", "voice", "media")}
PURE = CORES | {"lk-agents-abi"}
# Media's worker pool is its one threaded module (decisions 84, 85): outside the Clippy policy.
UNRESTRICTED = (
    "rust/crates/lk-agents-media/src/pool.rs",
    "rust/crates/lk-agents-media/src/pool_tests.rs",
)
# Async I/O crates (decisions 69, 77, 82): exempt from the purity bans; their edges are still
# checked.
ASYNC = {
    "lk-agents-inference",
    "lk-agents-worker",
    "lk-agents-net",
    "lk-agents-storage",
    "lk-agents-cli",
}
# The I/O policy's crates (rust/architecture/io/clippy.toml): everything that runs beside the host.
IO = ASYNC | {"lk-agents-runtime", "lk-agents-py"}
# The documented crate graph (rust-core.md "Crates and dependencies"): each checked crate's allowed
# workspace dependencies. Build tools are unchecked.
GRAPH = {
    "lk-agents-abi": set(),
    "lk-agents-core": {"lk-agents-abi"},
    "lk-agents-voice": {"lk-agents-abi", "lk-agents-core", "lk-agents-media"},
    "lk-agents-media": {"lk-agents-abi"},
    # inference -> core: chat formatting, strict tool schemas and JSON repair.
    "lk-agents-inference": {"lk-agents-abi", "lk-agents-core", "lk-agents-media", "lk-agents-net"},
    # net -> core: the delegation A2A client executes core::a2a decisions. net -> media: the
    # blocking I/O pool is an instance of media's worker pool.
    "lk-agents-net": {"lk-agents-abi", "lk-agents-core", "lk-agents-media"},
    "lk-agents-worker": {"lk-agents-abi", "lk-agents-net"},
    "lk-agents-cli": {"lk-agents-abi", "lk-agents-worker", "lk-agents-net"},
    "lk-agents-storage": {"lk-agents-abi", "lk-agents-net"},
    # The DST runs the runtime's real executors over faked ports (decision 92); providers and the
    # network are fakes at their ABI boundary, never inference's, net's or storage's code.
    "lk-agents-fuzz": PURE | {"lk-agents-runtime"},
    "lk-agents-runtime": PURE | ASYNC,
    "lk-agents-py": {"lk-agents-runtime", "lk-agents-abi"},
}
BANNED = {
    "async-trait",
    "async-std",
    "smol",
    "reqwest",
    "rusqlite",
    "hyper",
    "hyper-util",
    "mio",
    "tungstenite",
    "livekit",
    "livekit-ffi",
    "livekit-api",
    "lk-agents-runtime",
}
# The platforms the extension ships wheels for (.github/workflows/build.yml). Closures are resolved
# per platform: an edge only another target takes (chrono -> js-sys on wasm32) is never built.
PLATFORMS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)
# Workspace crates whose compiler and Clippy output must stay warning-free. This list only grows.
WARNING_CLEAN = {
    f"lk-agents-{name}"
    for name in (
        "abi",
        "core",
        "voice",
        "runtime",
        "py",
        "fuzz",
        "media",
        "net",
        "storage",
        "inference",
        "worker",
        "cli",
        "codegen",
        "bindgen",
        "host-derive",
    )
}
# The exact first line lk-agents-codegen writes; prose that merely starts with "Generated" is not it.
GENERATED = "//! GENERATED FILE -- DO NOT EDIT."


def generated(text: str) -> bool:
    # A header is outside Rust tokens. Searching later lines could match a raw string.
    return text.split("\n", 1)[0].rstrip() == GENERATED


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
    resolved = {node["id"]: node["deps"] for node in resolve["nodes"]}
    nodes = {node: [edge["pkg"] for edge in edges] for node, edges in resolved.items()}
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
        if name not in GRAPH:
            continue
        allowed = GRAPH[name]
        for edge in resolved[root]:
            dep = edge["pkg"]
            target = packages[dep]
            if target.startswith("lk-agents-") and target != name and target not in allowed:
                # Proc macros/build tooling do not define a product dependency direction.
                if target not in {"lk-agents-host-derive", "lk-agents-bindgen"}:
                    violations.add(f"direction:{name}:{target}")
    return violations


# Python-binding code lives in the extension crate alone (decision 74); the generator crates write
# it there as text and are not bound by it.
HOST_LANGUAGE = "lk-agents-py"
GENERATORS = {"lk-agents-codegen", "lk-agents-bindgen", "lk-agents-host-derive"}
PYO3 = re.compile(r"\bpyo3")


# Calling into host code (rust-core.md "HostInvoker"): only HostInvoker's Python entry calls a host
# callable, enters a context or awaits a coroutine. Every other pyo3 file doing so is an identity.
# The interpreter-entry responsibilities, grouped into flat Python binding modules.
HOST_ENTRY_FILES = {
    "rust/crates/lk-agents-py/src/py_calls.rs",
    "rust/crates/lk-agents-py/src/py_streams.rs",
    "rust/crates/lk-agents-py/src/py_logging.rs",
}
HOST_CALL = re.compile(
    r"\.call(?:0|1|_method|_method0|_method1)?\("
    r"|\binto_future\(|(?<!py_calls::)future_into_py"
)


def integration_test_paths(root: Path) -> set[str]:
    """Resolve declared Cargo integration targets, including sibling `_tests.rs` files."""
    targets = set()
    for manifest in (root / "rust/crates").glob("*/Cargo.toml"):
        crate = manifest.parent
        config = tomllib.loads(manifest.read_text())
        for target in config.get("test", []):
            if "path" not in target:
                continue
            path = (crate / target["path"]).resolve()
            if path.is_relative_to(crate.resolve()):
                targets.add(path.relative_to(root.resolve()).as_posix())
    return targets


def host_call_violations(root: Path, paths: list[str]) -> set[str]:
    """Every tracked Rust file naming pyo3 (generated output included) that calls into host code
    outside HostInvoker's Python entry; the generators write such calls as text and are exempt, as
    are a crate's integration tests, which set up the interpreter they drive."""
    violations = set()
    integration_tests = integration_test_paths(root)
    for path in paths:
        parts = Path(path).parts
        if parts[:2] != ("rust", "crates") or len(parts) < 4 or parts[2] in GENERATORS:
            continue
        # The entry itself, and integration tests driving a live interpreter.
        if path in HOST_ENTRY_FILES or path in integration_tests:
            continue
        text = NOT_CODE.sub(blank, (root / path).read_text())
        if PYO3.search(text) and HOST_CALL.search(text):
            violations.add(f"host-call:{path}")
    return violations


# The runtime executors the DST runs (decision 92): they spawn and sleep only through the injected
# Spawner and Clock, and every `select!` is `biased;`, so the DST's scheduler decides their order.
EXECUTORS = tuple(
    f"rust/crates/lk-agents-runtime/src/{module}/"
    for module in ("voice", "room_io", "avatar", "rtc")
)
TOKIO_RUNTIME = {
    "tokio-spawn": re.compile(r"\btokio::(?:task::)?spawn"),
    "tokio-time": re.compile(r"\btokio::time\b"),
    "tokio-net": re.compile(r"\btokio::net\b"),
    "tokio-runtime": re.compile(r"\btokio::runtime\b|\bHandle::(?:current|try_current)\b"),
    "unbiased-select": re.compile(r"\bselect!\s*\{(?!\s*biased\s*;)"),
}
TEST_MODULE = re.compile(r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*mod\s+\w+\s*([;{])")


def without_test_modules(text: str) -> str:
    """The text without its inline `#[cfg(test)] mod .. { .. }` bodies (brace-matched); an
    out-of-line `mod ..;` names a test file, which is skipped by name."""
    kept, at = [], 0
    while match := TEST_MODULE.search(text, at):
        kept.append(text[at : match.start()])
        at = match.end()
        if match.group(1) == "{":
            depth = 1
            while depth and at < len(text):
                depth += {"{": 1, "}": -1}.get(text[at], 0)
                at += 1
    kept.append(text[at:])
    return "".join(kept)


def executor_violations(root: Path, paths: list[str]) -> dict[str, int]:
    """Per executor file and kind, the count of tokio runtime uses and unbiased selects outside
    comments, strings and its tests (a test file or an inline `#[cfg(test)]` module)."""
    violations = {}
    for path in paths:
        if not path.startswith(EXECUTORS) or any(
            part.removesuffix(".rs").endswith("tests") for part in Path(path).parts
        ):
            continue
        text = without_test_modules(NOT_CODE.sub(blank, (root / path).read_text()))
        for kind, pattern in TOKIO_RUNTIME.items():
            if count := len(pattern.findall(text)):
                violations[f"executor:{path}:{kind}"] = count
    return violations


def pyo3_violations(metadata: dict, root: Path, paths: list[str]) -> set[str]:
    """A pyo3-family dependency of any other workspace crate (every feature), and every tracked
    Rust file of another crate naming pyo3, generated output included."""
    packages = {p["id"]: p["name"] for p in metadata["packages"]}
    nodes = {n["id"]: [d["pkg"] for d in n["deps"]] for n in metadata["resolve"]["nodes"]}
    exempt = GENERATORS | {HOST_LANGUAGE}
    violations = set()
    for member in metadata["workspace_members"]:
        name = packages[member]
        if name in exempt:
            continue
        for dep in nodes.get(member, []):
            target = packages[dep]
            if target == "pyo3" or target.startswith("pyo3-"):
                violations.add(f"pyo3-dependency:{name}:{target}")
    for path in paths:
        parts = Path(path).parts
        if parts[:2] != ("rust", "crates") or len(parts) < 4 or parts[2] in exempt:
            continue
        if PYO3.search((root / path).read_text()):
            violations.add(f"pyo3-file:{path}")
    return violations


# A pyo3 path (`pyo3::types::PyDict`, `pyo3_async_runtimes`) and an `async_trait` use, counted per
# file, enclosing item and path: a new use in a file or item already carrying others still grows.
PYO3_USE = re.compile(r"\bpyo3\w*(?:::\w+)*")
ASYNC_TRAIT_USE = re.compile(r"\basync_trait(?:::async_trait)?\b")
# Comments, strings and char literals, blanked before reading items so their text is no use.
NOT_CODE = re.compile(
    r"//[^\n]*|/\*.*?\*/|b?r(#*)\".*?\"\1|b?\"(?:\\.|[^\\\"])*\"|'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^\\'\n])'",
    re.S,
)
ATTRIBUTE = re.compile(r"#!?\[[^\]]*\]")
ITEM = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?(?:(?:const|async|unsafe|default|extern\s+\"\w+\")\s+)*"
    r"(?:(fn|struct|enum|trait|mod|union)\s+(\w+)|macro_rules!\s*(\w+)|(impl\b.*))",
    re.S,
)


def blank(match: re.Match) -> str:
    return re.sub(r"[^\n]", " ", match.group(0))


def item_uses(text: str, pattern: re.Pattern) -> dict[str, int]:
    """`pattern`'s matches outside comments and strings, keyed by the enclosing named items
    (`impl Trait for X::method`) and the matched text, with their counts."""
    code = NOT_CODE.sub(blank, text)
    uses: dict[str, int] = {}
    stack: list[str | None] = []
    events = sorted(
        [(m.start(), m.group(0)) for m in re.finditer(r"[{};]", code)]
        + [(m.start(), m) for m in pattern.finditer(code)],
        key=lambda event: event[0],
    )

    def record(names: list[str], found: list[str]) -> None:
        for text in found:
            key = f"{'::'.join(names) or '<file>'}:{text}"
            uses[key] = uses.get(key, 0) + 1

    # A use belongs to the item whose header it is in (a signature, an attribute), else to the
    # enclosing items; it is placed when its header ends.
    header_start, pending = 0, []
    for at, event in events:
        if isinstance(event, re.Match):
            pending.append(event.group(0))
            continue
        names = [name for name in stack if name]
        if event == "{":
            header = " ".join(ATTRIBUTE.sub(" ", code[header_start:at]).split())
            item = ITEM.match(header)
            name = None
            if item and item.group(4):
                name = " ".join(item.group(4).split())
            elif item:
                name = item.group(2) or item.group(3)
            stack.append(name)
            record(names + [name] if name else names, pending)
        else:
            record(names, pending)
            if event == "}" and stack:
                stack.pop()
        header_start, pending = at + 1, []
    record([name for name in stack if name], pending)
    return uses


def use_violations(root: Path, paths: list[str]) -> dict[str, int]:
    """Every pyo3 use outside the extension and generator crates, and every async-trait use in a
    subject core or ABI, by file, enclosing item and path; each count is a ceiling."""
    violations = {}
    for path in paths:
        parts = Path(path).parts
        if parts[:2] != ("rust", "crates") or len(parts) < 4:
            continue
        checks = []
        if parts[2] not in GENERATORS | {HOST_LANGUAGE}:
            checks.append(("pyo3", PYO3_USE))
        if parts[2] in PURE:
            checks.append(("async-trait", ASYNC_TRAIT_USE))
        if not checks:
            continue
        text = (root / path).read_text()
        for kind, pattern in checks:
            if not pattern.search(text):
                continue
            for key, count in item_uses(text, pattern).items():
                violations[f"{kind}-use:{path}:{key}"] = count
    return violations


def length_violations(root: Path, paths: list[str]) -> dict[str, int]:
    violations = {}
    for path in paths:
        text = (root / path).read_text()
        if generated(text):
            continue
        lines = len(text.splitlines())
        if lines > 1250:
            violations[path] = lines
    return violations


# `#[path = ".."]`, bare or inside `cfg_attr`: a module file the attribute names.
PATH_ATTR = re.compile(r'#\s*\[\s*(?:cfg_attr\s*\([^\]]*?,\s*)?path\s*=\s*"([^"]+)"')


def path_violations(root: Path, paths: list[str]) -> set[str]:
    """A `#[path]` module outside its own crate's tree (`src/`, `tests/`, ..): the file then compiles
    in another crate, beyond its directory's dependency and Clippy policy."""
    violations = set()
    crates = (root / "rust/crates").resolve()
    for path in paths:
        parts = Path(path).parts
        if parts[:2] != ("rust", "crates") or len(parts) < 5:
            continue
        own = crates / parts[2]
        source = root / path
        for target in PATH_ATTR.findall(source.read_text()):
            resolved = (source.parent / target).resolve()
            if not resolved.is_relative_to(own):
                violations.add(f"{path} -> {os.path.relpath(resolved, root.resolve())}")
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


def stale(actual, baseline) -> list[str]:
    """A baseline row no current violation matches would let that violation return unseen."""
    return [f"stale baseline row: {identity}" for identity in sorted(set(baseline) - set(actual))]


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


DEPENDENCY_BASELINES = {
    "legacy": "rust/architecture/dependencies-baseline.json",
    "sqlite": "rust/architecture/dependencies-sqlite-baseline.json",
}


def baseline_family(identity: str) -> str:
    """Newly enforced families cannot receive exemptions from an older ratchet."""
    if identity.startswith("dependency:") and identity.endswith(":rusqlite"):
        return "sqlite"
    return "legacy"


def read_baselines(root: Path, paths: dict[str, str]) -> tuple[set[str], list[str]]:
    """Load independent first baselines, retaining each family's complete history."""
    baseline = set()
    errors = []
    for family, path in paths.items():
        entries = json.loads((root / path).read_text())
        if not isinstance(entries, list) or any(not isinstance(entry, str) for entry in entries):
            raise ValueError(f"{path}: expected an array of identities")
        if len(set(entries)) != len(entries):
            raise ValueError(f"{path}: duplicate identities")
        errors.extend(
            f"{path}: identity belongs to another baseline family: {entry}"
            for entry in entries
            if baseline_family(entry) != family
        )
        errors.extend(check_history(root, path, entries))
        baseline.update(entries)
    return baseline, errors


def cargo_metadata(manifest: Path, *args: str) -> dict:
    """The all-feature resolved graph; Cargo's own errors reach stderr."""
    command = [
        CARGO_RUNNER,
        "metadata",
        "--manifest-path",
        str(manifest),
        "--locked",
        "--all-features",
    ]
    result = subprocess.run(
        [*command, "--format-version", "1", *args], check=True, stdout=subprocess.PIPE, text=True
    )
    return json.loads(result.stdout)


def shipped_dependency_violations(manifest: Path) -> set[str]:
    """Dependency violations of the closures the shipped platforms build."""
    return set().union(
        *(
            dependency_violations(cargo_metadata(manifest, "--filter-platform", platform))
            for platform in PLATFORMS
        )
    )


def check_boundaries(metadata: dict) -> int:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "rust/Cargo.toml"
    expected = {name for name in PURE if (root / "rust/crates" / name / "Cargo.toml").exists()}
    members = set(metadata["workspace_members"])
    found = {p["name"] for p in metadata["packages"] if p["id"] in members}
    if missing := expected - found:
        raise ValueError(f"cargo metadata omitted existing subject crates: {sorted(missing)}")
    paths = tracked_rust_paths(root)
    if not paths or paths == [""]:
        raise ValueError("no tracked Rust files found")
    errors = []
    baseline, history_errors = read_baselines(root, DEPENDENCY_BASELINES)
    actual = shipped_dependency_violations(manifest)
    errors.extend(ratchet(actual, baseline))
    errors.extend(stale(actual, baseline))
    errors.extend(history_errors)
    print(f"dependencies: {len(actual)} existing identities; baseline {len(baseline)}")
    for name, actual in (
        ("lengths", length_violations(root, paths)),
        ("paths", path_violations(root, paths)),
        ("pyo3", pyo3_violations(metadata, root, paths)),
        ("host-calls", host_call_violations(root, paths)),
        ("executors", executor_violations(root, paths)),
        ("uses", use_violations(root, paths)),
    ):
        path = f"rust/architecture/{name}-baseline.json"
        baseline = json.loads((root / path).read_text())
        errors.extend(ratchet(actual, baseline))
        errors.extend(stale(actual, baseline))
        errors.extend(check_history(root, path, baseline))
        print(f"{name}: {len(actual)} existing identities; baseline {len(baseline)}")
    for name in ("export-placement",):
        path = f"rust/ai_slop/lk-agents-codegen/{name}-baseline.json"
        errors.extend(check_history(root, path, json.loads((root / path).read_text())))
    for error in errors:
        print(error, file=sys.stderr)
    return bool(errors)


def warning_violations(records: list[dict], names: Mapping[str, str]) -> list[str]:
    """Compiler and Clippy warnings of workspace crates, from a complete transcript."""
    if (
        not records
        or records[-1].get("reason") != "build-finished"
        or records[-1].get("success") is not True
    ):
        raise ValueError("cargo check transcript is empty, incomplete or unsuccessful")
    built = {names.get(r["package_id"]) for r in records if r.get("reason") == "compiler-artifact"}
    if missing := WARNING_CLEAN - built:
        raise ValueError(
            f"cargo check transcript is missing warning-clean crates: {sorted(missing)}"
        )
    return [
        record["message"].get("rendered") or record["message"]["message"]
        for record in records
        if record.get("reason") == "compiler-message"
        and record["message"]["level"] == "warning"
        and names.get(record["package_id"]) in WARNING_CLEAN
    ]


def check_workspace(metadata: dict, target: Path) -> int:
    """One Clippy pass checks compilation and warnings across the workspace."""
    root = Path(__file__).resolve().parents[2]
    names = {p["id"]: p["name"] for p in metadata["packages"]}
    process = subprocess.run(
        [
            CARGO_RUNNER,
            "clippy",
            "--manifest-path",
            str(root / "rust/Cargo.toml"),
            "--locked",
            "--workspace",
            "--all-targets",
            "--features",
            "lk-agents-py/ffmpeg",
            "--no-deps",
            "--target-dir",
            str(target),
            "--message-format=json",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
    )
    records = [json.loads(line) for line in process.stdout.splitlines()]
    for record in records:
        if record.get("reason") != "compiler-message":
            continue
        message = record["message"]
        if message["level"] == "error":
            sys.stderr.write(message.get("rendered") or message["message"])
    if process.returncode:
        return process.returncode
    errors = warning_violations(records, names)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"warnings: {len(errors)} in {len(WARNING_CLEAN)} warning-clean crates")
    return bool(errors)


LINTS = ("disallowed_methods", "disallowed_types", "disallowed_macros")


class Policy(NamedTuple):
    """A Clippy configuration over a crate set. A purity site also needs its source exemption;
    an I/O site is held by its baseline alone."""

    name: str
    crates: frozenset[str]
    conf: str
    baseline: str
    modes: tuple[str, ...]
    exempted: bool


POLICIES = (
    Policy(
        "purity",
        frozenset(PURE),
        "rust/architecture",
        "rust/architecture/clippy-baseline.json",
        ("default", "all"),
        True,
    ),
    Policy(
        "io",
        frozenset(IO),
        "rust/architecture/io",
        "rust/architecture/io/clippy-baseline.json",
        ("default", "all"),
        False,
    ),
)
PURITY = POLICIES[0]


def policed(path: str, policy: Policy = PURITY) -> bool:
    """A handwritten source of one of the policy's crates."""
    parts = Path(path).parts
    return len(parts) > 3 and parts[2] in policy.crates and not path.startswith(UNRESTRICTED)


def source_paths(root: Path, policy: Policy = PURITY) -> list[str]:
    paths = (
        git(root, "ls-files", "-c", "-o", "--exclude-standard", "-z", "--", "rust/crates/*/*.rs")
        .strip("\0")
        .split("\0")
    )
    result = []
    for path in paths:
        if not path or not policed(path, policy):
            continue
        if Path(path).parts[3] in {"build.rs", "build"}:
            continue
        if not (root / path).is_file():
            continue
        text = (root / path).read_text()
        if not generated(text):
            result.append(path)
    if not result:
        raise ValueError("no handwritten core Rust sources found")
    return result


def inventory(root: Path, helper: Path, paths: list[str]) -> set[str]:
    process = subprocess.run(
        [str(helper), "--architecture", "inventory", *paths],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    return set(process.stdout.splitlines())


def target_directory(root: Path) -> Path:
    """Use Cargo's target selection for helpers and compiler fixtures alike."""
    result = subprocess.run(
        [
            CARGO_RUNNER,
            "metadata",
            "--no-deps",
            "--format-version=1",
            "--locked",
            "--manifest-path",
            str(root / "rust/Cargo.toml"),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    return Path(json.loads(result.stdout)["target_directory"])


def build_helper(root: Path, target: Path | None = None) -> Path:
    process = subprocess.run(
        [
            CARGO_RUNNER,
            "build",
            "--quiet",
            "--profile",
            "check-tools",
            "--manifest-path",
            str(root / "rust/Cargo.toml"),
            "--locked",
            "-p",
            "lk-agents-codegen",
            "--bin",
            "lk-agents-codegen",
            "--message-format=json",
            *(["--target-dir", str(target)] if target is not None else []),
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    sys.stderr.write(process.stderr)
    if process.returncode:
        for line in process.stdout.splitlines():
            record = json.loads(line)
            if record.get("reason") == "compiler-message":
                sys.stderr.write(record["message"].get("rendered") or str(record["message"]))
        raise subprocess.CalledProcessError(process.returncode, process.args)
    for line in process.stdout.splitlines():
        record = json.loads(line)
        if (
            record.get("reason") == "compiler-artifact"
            and record["target"]["name"] == "lk-agents-codegen"
            and record.get("executable")
        ):
            return Path(record["executable"])
    raise ValueError("Cargo did not report the codegen executable")


def command(
    root: Path, crates: list[str], mode: str = "all", target: Path | None = None
) -> list[str]:
    args = [
        CARGO_RUNNER,
        "clippy",
        "--no-deps",
        "--manifest-path",
        str(root / "rust/Cargo.toml"),
        "--locked",
        "--all-targets",
    ]
    if target is not None:
        args.extend(["--target-dir", str(target)])
    for crate in crates:
        args.extend(["-p", crate])
    if mode == "all":
        args.append("--all-features")
    elif mode != "default":
        raise ValueError(f"unknown Clippy feature mode: {mode}")
    return [
        *args,
        "--message-format=json",
        "--",
        # Only the policy's lints decide; a crate's other Clippy findings are not this gate's.
        "-A",
        "clippy::all",
        *(arg for lint in LINTS for arg in ("--force-warn", f"clippy::{lint}")),
    ]


def compiler_locations(
    messages: list[dict], root: Path, policy: Policy = PURITY
) -> list[tuple[str, int, int, str, str]]:
    """Resolve macro-expansion spans back to the handwritten invoking core item."""
    root = root.resolve()
    sites = set()
    for message in messages:
        code = (message.get("code") or {}).get("code", "")
        if code not in {f"clippy::{lint}" for lint in LINTS}:
            continue
        candidates = []
        unrestricted = []
        for span in message["spans"]:
            if not span["is_primary"]:
                continue
            in_worker_pool = False
            while span:
                path = Path(span["file_name"])
                if not path.is_absolute():
                    path = root / "rust" / path
                try:
                    relative = path.resolve().relative_to(root).as_posix()
                except ValueError:
                    relative = ""
                in_worker_pool |= relative.startswith(UNRESTRICTED)
                if relative.startswith("rust/crates/") and policed(relative, policy):
                    candidates.append(
                        (
                            relative,
                            span["line_start"],
                            span["column_start"],
                            code,
                            message["message"],
                        )
                    )
                    break
                span = (span.get("expansion") or {}).get("span")
            unrestricted.append(in_worker_pool)
        if not candidates:
            if unrestricted and all(unrestricted):
                continue
            spans = ", ".join(span["file_name"] for span in message["spans"] if span["is_primary"])
            raise ValueError(
                f"unmapped core policy diagnostic {code}: {message['message']}: "
                f"{spans or 'no primary span'}"
            )
        sites.update(candidates)
    return sorted(sites)


def site_inventory(root: Path, helper: Path, locations: list[tuple]) -> set[str]:
    """Identity = lint, lexical item, offending source line and local occurrence."""
    from collections import Counter, defaultdict

    by_file = defaultdict(list)
    for site in locations:
        by_file[site[0]].append(site)
    identities = set()
    occurrences = Counter()
    for path, sites in by_file.items():
        result = subprocess.run(
            [str(helper), "--architecture", "locations", path, *(f"{s[1]}:{s[2]}" for s in sites)],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        scopes = result.stdout.splitlines()
        if len(scopes) != len(sites):
            raise ValueError(f"{path}: incomplete AST location inventory")
        lines = (root / path).read_text().splitlines()
        for (_, line, _, lint, message), scope in zip(sites, scopes, strict=True):
            item = scope.split("\t", 1)[1]
            identity = f"site:{path}:{item}:{lint}:{message}:{lines[line - 1].strip()}"
            occurrences[identity] += 1
            identities.add(f"{identity}:{occurrences[identity]}")
    return identities


def exemption_errors(
    root: Path, helper: Path, records: list[dict], policy: Policy = PURITY
) -> list[str]:
    """Each compiler site must retain an effective source exemption."""
    from collections import defaultdict

    features = {
        record["package_id"]: record["features"]
        for record in records
        if record.get("reason") == "compiler-artifact"
    }
    by_file = defaultdict(set)
    for record in records:
        if record.get("reason") != "compiler-message" or "custom-build" in record["target"]["kind"]:
            continue
        message = record["message"]
        if (message.get("code") or {}).get("code") not in {f"clippy::{lint}" for lint in LINTS}:
            continue
        expanded = any(s.get("expansion") for s in message["spans"] if s["is_primary"])
        crate_root = record["target"]["src_path"]
        package = record["package_id"]
        if package not in features:
            raise ValueError(f"{crate_root}: missing compiler feature inventory")
        for path, line, column, lint, _ in compiler_locations([message], root, policy):
            by_file[(path, crate_root, package)].add((line, column, lint, expanded))
    groups = defaultdict(dict)
    for (path, crate_root, package), sites in by_file.items():
        groups[(crate_root, package)][path] = sorted(sites)
    errors = []
    for (crate_root, package), files in sorted(groups.items()):
        env = {
            **os.environ,
            "ARCHITECTURE_FEATURES": ",".join(features[package]),
            "ARCHITECTURE_CRATE_ROOT": crate_root,
        }
        rows = [
            "\t".join(
                [
                    path,
                    *(
                        f"{line}:{column}:{lint.removeprefix('clippy::')}:{int(expanded)}"
                        for line, column, lint, expanded in sites
                    ),
                ]
            )
            for path, sites in sorted(files.items())
        ]
        process = subprocess.run(
            [str(helper), "--architecture", "coverage-batch"],
            input="\n".join(rows) + "\n",
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        sites = [(path, *site) for path, entries in sorted(files.items()) for site in entries]
        verdicts = process.stdout.splitlines()
        if len(verdicts) != len(sites) or any(v not in {"allowed", "denied"} for v in verdicts):
            raise ValueError(f"{crate_root}: incomplete exemption coverage")
        errors.extend(
            f"{path}:{line}:{column}: {lint} requires an effective source exemption"
            for (path, line, column, lint, _), verdict in zip(sites, verdicts, strict=True)
            if verdict != "allowed"
        )
    return errors


def compiler_messages(output: str | list[dict]) -> list[dict]:
    records = (
        [json.loads(line) for line in output.splitlines()] if isinstance(output, str) else output
    )
    if (
        not records
        or records[-1].get("reason") != "build-finished"
        or records[-1].get("success") is not True
        or not any(r.get("reason") == "compiler-artifact" for r in records)
    ):
        raise ValueError("Clippy transcript is empty, incomplete or unsuccessful")
    return [
        record["message"]
        for record in records
        if record.get("reason") == "compiler-message"
        and "custom-build" not in record.get("target", {}).get("kind", [])
    ]


def check_clippy(helper: Path, target: Path) -> int:
    root = Path(__file__).resolve().parents[2]
    return max(check_policy(root, helper, policy, target / policy.name) for policy in POLICIES)


def library(root: Path, crate: str) -> str:
    """The crate's library target name (lk-agents-py's is `_core`)."""
    manifest = tomllib.loads((root / "rust/crates" / crate / "Cargo.toml").read_text())
    return manifest.get("lib", {}).get("name", crate.replace("-", "_"))


def check_policy(root: Path, helper: Path, policy: Policy, target: Path) -> int:
    crates = sorted(name for name in policy.crates if (root / "rust/crates" / name).is_dir())
    errors: list[str] = []
    snapshots = {
        path: hashlib.sha256((root / path).read_bytes()).digest()
        for path in source_paths(root, policy)
    }
    actual = {f"allow:{identity}" for identity in inventory(root, helper, list(snapshots))}
    baseline, history_errors = read_baselines(root, {"legacy": policy.baseline})
    errors.extend(ratchet(actual, baseline))
    errors.extend(history_errors)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    env = os.environ.copy()
    env["CLIPPY_CONF_DIR"] = str(root / policy.conf)
    locations = set()
    for mode in policy.modes:
        process = subprocess.run(
            command(root, crates, mode, target / mode),
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
        )
        records = [json.loads(line) for line in process.stdout.splitlines()]
        if process.returncode:
            for record in records:
                if (
                    record.get("reason") == "compiler-message"
                    and record["message"]["level"] == "error"
                ):
                    sys.stderr.write(record["message"].get("rendered") or str(record["message"]))
            return process.returncode
        messages = compiler_messages(records)
        artifacts = {
            record["target"]["name"]
            for record in records
            if record.get("reason") == "compiler-artifact"
            and {"lib", "rlib"} & set(record.get("target", {}).get("kind", []))
        }
        missing = {library(root, name) for name in crates} - artifacts
        if missing:
            raise ValueError(
                f"Clippy {policy.name} {mode} transcript is missing artifacts: {sorted(missing)}"
            )
        locations.update(compiler_locations(messages, root, policy))
        if policy.exempted:
            errors.extend(exemption_errors(root, helper, records, policy))
        print(f"clippy {policy.name} {mode}: successful complete compiler pass")
    current = {
        path: hashlib.sha256((root / path).read_bytes()).digest()
        for path in source_paths(root, policy)
    }
    changed = sorted(
        path
        for path in snapshots.keys() | current.keys()
        if snapshots.get(path) != current.get(path)
    )
    if changed:
        raise ValueError("source changed during Clippy captures; rerun: " + ", ".join(changed))
    actual.update(site_inventory(root, helper, sorted(locations)))
    errors.extend(ratchet(actual, baseline))
    errors.extend(stale(actual, baseline))
    for error in errors:
        print(error, file=sys.stderr)
    print(
        f"clippy {policy.name}: {sum(s.startswith('site:') for s in actual)} sites; "
        f"{sum(s.startswith('allow:') for s in actual)} exemptions; baseline {len(baseline)}"
    )
    return int(bool(errors))


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    metadata = cargo_metadata(root / "rust/Cargo.toml")
    target = Path(metadata["target_directory"])
    # A sibling stays usable when mbx has collected the default target's symlink destination.
    target = target.with_name(target.name + "-checks")
    if check_boundaries(metadata) or check_workspace(metadata, target / "workspace"):
        return 1
    # Clippy's different feature/lint sets otherwise replace each other's Cargo fingerprints.
    # Keep their mutable state private; the compiler cache still shares equivalent artifacts.
    helper = build_helper(root, target / "tools")
    return (
        check_clippy(helper, target)
        or subprocess.run([str(helper), "--check"], cwd=root).returncode
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"architecture: {error}")
