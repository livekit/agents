"""Check Cargo boundaries, Rust sources, compiler lints and shrink-only baselines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import tomllib

CORES = {f"lk-agents-{name}" for name in ("core", "voice", "media")}
PURE = CORES | {"lk-agents-abi"}
# Media's worker pool is its one threaded module (decisions 84, 85): outside the Clippy policy.
UNRESTRICTED = (
    "rust/crates/lk-agents-media/src/workers.rs",
    "rust/crates/lk-agents-media/src/workers/",
)
# Async I/O crates (decisions 69, 77, 82): exempt from the purity bans; their edges are still
# checked.
ASYNC = {"lk-agents-inference", "lk-agents-worker", "lk-agents-net", "lk-agents-storage"}
TEST_SUPPORT = "lk-agents-test-support"
# The documented crate graph (rust-core.md "Crates and dependencies"): each checked crate's allowed
# workspace dependencies. Every crate's tests may also use test support. Build tools are unchecked.
GRAPH = {
    "lk-agents-abi": set(),
    "lk-agents-core": {"lk-agents-abi"},
    "lk-agents-voice": {"lk-agents-abi", "lk-agents-core", "lk-agents-media"},
    "lk-agents-media": {"lk-agents-abi"},
    # inference -> core (chat formatting, schemas, JSON repair) is pending owner approval.
    "lk-agents-inference": {"lk-agents-abi", "lk-agents-media", "lk-agents-net"}
    | {"lk-agents-core"},
    # net -> core (the A2A client executes core::a2a decisions) is pending owner approval.
    "lk-agents-net": {"lk-agents-abi"} | {"lk-agents-core"},
    "lk-agents-worker": {"lk-agents-abi", "lk-agents-net"},
    "lk-agents-storage": {"lk-agents-abi", "lk-agents-net"},
    # The realtime and duplex fakes still run inference's realtime machine (decision 70); the DST
    # runs the runtime's real executors over faked ports (decision 92).
    "lk-agents-fuzz": PURE
    | {TEST_SUPPORT}
    | {"lk-agents-inference", "lk-agents-net", "lk-agents-runtime"},
    TEST_SUPPORT: PURE,
    "lk-agents-bench-steps": PURE,
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
            dev = edge.get("dep_kinds") and all(k.get("kind") == "dev" for k in edge["dep_kinds"])
            if target == TEST_SUPPORT and dev:
                continue
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
HOST_ENTRY = "rust/crates/lk-agents-py/src/host/"
HOST_CALL = re.compile(
    r"\.call(?:0|1|_method|_method0|_method1)?\(|call_soon_threadsafe|run_coroutine_threadsafe"
    r"|\binto_future\(|(?<!host::)future_into_py"
)


def host_call_violations(root: Path, paths: list[str]) -> set[str]:
    """Every tracked Rust file naming pyo3 (generated output included) that calls into host code
    outside HostInvoker's Python entry; the generators write such calls as text and are exempt, as
    are a crate's integration tests, which set up the interpreter they drive."""
    violations = set()
    for path in paths:
        parts = Path(path).parts
        if parts[:2] != ("rust", "crates") or len(parts) < 4 or parts[2] in GENERATORS:
            continue
        # The entry itself, and integration tests driving a live interpreter.
        if path.startswith(HOST_ENTRY) or parts[3] == "tests":
            continue
        text = (root / path).read_text()
        if PYO3.search(text) and HOST_CALL.search(text):
            violations.add(f"host-call:{path}")
    return violations


# The runtime executors the DST runs (decision 92): they spawn and sleep only through the injected
# Spawner and Clock, and every `select!` is `biased;`, so the DST's scheduler decides their order.
EXECUTORS = tuple(
    f"rust/crates/lk-agents-runtime/src/{module}/" for module in ("voice", "room_io", "avatar", "rtc")
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
    its tests (a test file or an inline `#[cfg(test)]` module)."""
    violations = {}
    for path in paths:
        if not path.startswith(EXECUTORS) or any(
            part.removesuffix(".rs").endswith("tests") for part in Path(path).parts
        ):
            continue
        text = without_test_modules((root / path).read_text())
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
        own = crates / parts[2] / parts[3]
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
CLIPPY_BASELINES = {
    "legacy": "rust/architecture/clippy-baseline.json",
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


def check_boundaries() -> int:
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
    baseline, history_errors = read_baselines(root, DEPENDENCY_BASELINES)
    actual = dependency_violations(metadata)
    errors.extend(ratchet(actual, baseline))
    errors.extend(history_errors)
    print(f"dependencies: {len(actual)} existing identities; baseline {len(baseline)}")
    for name, actual in (
        ("lengths", length_violations(root, paths)),
        ("paths", path_violations(root, paths)),
        ("pyo3", pyo3_violations(metadata, root, paths)),
        ("host-calls", host_call_violations(root, paths)),
        ("executors", executor_violations(root, paths)),
    ):
        path = f"rust/architecture/{name}-baseline.json"
        baseline = json.loads((root / path).read_text())
        errors.extend(ratchet(actual, baseline))
        errors.extend(check_history(root, path, baseline))
        print(f"{name}: {len(actual)} existing identities; baseline {len(baseline)}")
    for name in ("export-placement", "export-metadata", "in-place-proxies"):
        path = f"rust/crates/lk-agents-codegen/{name}-baseline.json"
        errors.extend(check_history(root, path, json.loads((root / path).read_text())))
    for error in errors:
        print(error, file=sys.stderr)
    return bool(errors)


LINTS = ("disallowed_methods", "disallowed_types", "disallowed_macros")
# All configs carry the same compiler-resolved policy; aliases cannot bypass Clippy resolution.
REQUIRED = {
    "disallowed-methods": {
        "tokio::runtime::Handle::spawn",
        "tokio::runtime::Handle::spawn_blocking",
        "tokio::runtime::Runtime::spawn",
        "tokio::runtime::Runtime::spawn_blocking",
        "tokio::task::LocalSet::spawn_local",
        "tokio::task::JoinSet::spawn",
        "tokio::task::JoinSet::spawn_on",
        "tokio::task::JoinSet::spawn_local",
        "tokio::task::JoinSet::spawn_local_on",
        "tokio::task::JoinSet::spawn_blocking",
        "tokio::task::JoinSet::spawn_blocking_on",
        "rand::random_iter",
        "rand::random_bool",
        "rand::random_ratio",
        "rand::make_rng",
        "rand::Rng::sample",
        "rand::Rng::sample_iter",
        "rand::Rng::fill",
        "rand::Rng::random_bool",
        "rand::Rng::random_ratio",
        "rand_core::TryRngCore::try_next_u32",
        "rand_core::TryRngCore::try_next_u64",
        "rand_core::TryRngCore::try_fill_bytes",
        "rand::SeedableRng::from_os_rng",
        "rand::SeedableRng::from_entropy",
        "rand_core::SeedableRng::from_os_rng",
        "rand_core::SeedableRng::from_entropy",
        "std::thread::Scope::spawn",
        "std::time::Instant::elapsed",
        "std::time::SystemTime::elapsed",
        "tokio::time::Instant::elapsed",
        "rand::Rng::random",
        "rand::Rng::random_range",
        "rand::Rng::gen",
        "rand::Rng::gen_range",
        "rand::RngCore::next_u32",
        "rand::RngCore::next_u64",
        "rand::RngCore::fill_bytes",
        "rand_core::RngCore::next_u32",
        "rand_core::RngCore::next_u64",
        "rand_core::RngCore::fill_bytes",
        "getrandom::fill",
        "std::thread::spawn",
        "std::thread::Builder::spawn",
        "std::time::Instant::now",
        "std::time::SystemTime::now",
        "std::collections::hash_map::RandomState::new",
        "std::process::id",
        "std::env::var",
        "std::env::var_os",
        "std::env::vars",
        "std::env::vars_os",
        "uuid::Uuid::new_v4",
        "rand::random",
        "rand::random_range",
        "rand::rng",
        "rand::thread_rng",
        "tokio::spawn",
        "tokio::task::spawn",
        "tokio::task::spawn_blocking",
        "tokio::task::spawn_local",
        "tokio::time::sleep",
        "tokio::time::sleep_until",
        "tokio::time::interval",
        "tokio::time::interval_at",
        "tokio::time::timeout",
        "tokio::time::timeout_at",
        "tokio::time::Instant::now",
        "tokio::sync::mpsc::unbounded_channel",
        "std::sync::mpsc::channel",
        "futures::channel::mpsc::unbounded",
        "futures_channel::mpsc::unbounded",
        "crossbeam::channel::unbounded",
        "crossbeam_channel::unbounded",
    },
    "disallowed-types": {
        "parking_lot::ReentrantMutex",
        "lock_api::Mutex",
        "lock_api::RwLock",
        "lock_api::ReentrantMutex",
        "rand::rngs::ThreadRng",
        "rand::rngs::OsRng",
        "tokio::sync::mpsc::UnboundedSender",
        "tokio::sync::mpsc::UnboundedReceiver",
        "futures::channel::mpsc::UnboundedSender",
        "futures::channel::mpsc::UnboundedReceiver",
        "futures_channel::mpsc::UnboundedSender",
        "futures_channel::mpsc::UnboundedReceiver",
        "std::sync::mpsc::Sender",
        "std::sync::mpsc::Receiver",
        "tokio::sync::Mutex",
        "tokio::sync::RwLock",
        "std::collections::HashMap",
        "std::collections::HashSet",
        "std::sync::Mutex",
        "std::sync::RwLock",
        "parking_lot::Mutex",
        "parking_lot::RwLock",
        "std::sync::atomic::AtomicBool",
        "std::sync::atomic::AtomicU8",
        "std::sync::atomic::AtomicU16",
        "std::sync::atomic::AtomicU32",
        "std::sync::atomic::AtomicU64",
        "std::sync::atomic::AtomicUsize",
        "std::sync::atomic::AtomicI8",
        "std::sync::atomic::AtomicI16",
        "std::sync::atomic::AtomicI32",
        "std::sync::atomic::AtomicI64",
        "std::sync::atomic::AtomicIsize",
        "std::sync::atomic::AtomicPtr",
    },
    "disallowed-macros": {
        "tokio::select",
        "tokio::join",
        "tokio::try_join",
        "futures::select",
        "futures::join",
        "futures::try_join",
    },
}


def configuration_errors(root: Path, crates: list[str]) -> list[str]:
    errors = []
    for crate in crates:
        path = root / "rust/crates" / crate / "clippy.toml"
        if path.with_name(".clippy.toml").exists():
            errors.append(f"{path}: shadowing .clippy.toml is forbidden")
        config = tomllib.loads(path.read_text())
        for key, required in REQUIRED.items():
            actual = {
                entry if isinstance(entry, str) else entry["path"] for entry in config.get(key, [])
            }
            errors.extend(f"{path}: missing {key}: {name}" for name in sorted(required - actual))
    return errors


def policed(path: str) -> bool:
    """A subject-core or ABI source the Clippy policy covers."""
    parts = Path(path).parts
    return len(parts) > 3 and parts[2] in PURE and not path.startswith(UNRESTRICTED)


def source_paths(root: Path) -> list[str]:
    paths = (
        git(root, "ls-files", "-c", "-o", "--exclude-standard", "-z", "--", "rust/crates/*/*.rs")
        .strip("\0")
        .split("\0")
    )
    result = []
    for path in paths:
        if not path or not policed(path):
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
            "cargo",
            "metadata",
            "--no-deps",
            "--format-version=1",
            "--locked",
            "--config",
            str(root / "rust/.cargo/config.toml"),
            "--manifest-path",
            str(root / "rust/Cargo.toml"),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    return Path(json.loads(result.stdout)["target_directory"])


def build_helper(root: Path) -> Path:
    subprocess.run(
        [
            "cargo",
            "build",
            "--config",
            str(root / "rust/.cargo/config.toml"),
            "--quiet",
            "--manifest-path",
            str(root / "rust/Cargo.toml"),
            "--locked",
            "-p",
            "lk-agents-codegen",
            "--bin",
            "lk-agents-codegen",
        ],
        cwd=root,
        check=True,
    )
    return target_directory(root) / "debug/lk-agents-codegen"


def command(root: Path, crates: list[str], mode: str = "all") -> list[str]:
    args = [
        "cargo",
        "clippy",
        "--no-deps",
        "--config",
        str(root / "rust/.cargo/config.toml"),
        "--manifest-path",
        str(root / "rust/Cargo.toml"),
        "--locked",
        "--all-targets",
    ]
    for crate in crates:
        args.extend(["-p", crate])
    if mode == "all":
        args.append("--all-features")
    elif mode == "python":
        features = []
        for crate in crates:
            manifest = tomllib.loads((root / "rust/crates" / crate / "Cargo.toml").read_text())
            if "python" in manifest.get("features", {}):
                features.append(f"{crate}/python")
        if features:
            args.extend(["--features", ",".join(features)])
    elif mode != "default":
        raise ValueError(f"unknown Clippy feature mode: {mode}")
    return [
        *args,
        "--message-format=json",
        "--",
        *(arg for lint in LINTS for arg in ("--force-warn", f"clippy::{lint}")),
    ]


def compiler_locations(messages: list[dict], root: Path) -> list[tuple[str, int, int, str, str]]:
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
                if relative.startswith("rust/crates/") and policed(relative):
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


def exemption_errors(root: Path, helper: Path, records: list[dict]) -> list[str]:
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
        for path, line, column, lint, _ in compiler_locations([message], root):
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


def check_clippy() -> int:
    root = Path(__file__).resolve().parents[2]
    crates = sorted(name for name in PURE if (root / "rust/crates" / name).is_dir())
    errors = configuration_errors(root, crates)
    if errors:
        sys.exit("\n".join(errors))
    helper = build_helper(root)
    snapshots = {
        path: hashlib.sha256((root / path).read_bytes()).digest() for path in source_paths(root)
    }
    actual = {f"allow:{identity}" for identity in inventory(root, helper, list(snapshots))}
    baseline, history_errors = read_baselines(root, CLIPPY_BASELINES)
    errors.extend(ratchet(actual, baseline))
    errors.extend(history_errors)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    env = os.environ.copy()
    env["CLIPPY_CONF_DIR"] = str(root / "rust/crates/lk-agents-abi")
    locations = set()
    for mode in ("default", "python", "all"):
        process = subprocess.run(
            command(root, crates, mode),
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
        )
        sys.stderr.write(process.stderr)
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
            and "lib" in record.get("target", {}).get("kind", [])
        }
        missing = {name.replace("-", "_") for name in crates} - artifacts
        if missing:
            raise ValueError(
                f"Clippy {mode} transcript is missing core artifacts: {sorted(missing)}"
            )
        sites = compiler_locations(messages, root)
        errors.extend(exemption_errors(root, helper, records))
        locations.update(sites)
        print(f"clippy {mode}: successful complete compiler pass")
    current = {
        path: hashlib.sha256((root / path).read_bytes()).digest() for path in source_paths(root)
    }
    changed = sorted(
        path
        for path in snapshots.keys() | current.keys()
        if snapshots.get(path) != current.get(path)
    )
    if changed:
        raise ValueError("core source changed during Clippy captures; rerun: " + ", ".join(changed))
    actual.update(site_inventory(root, helper, sorted(locations)))
    errors.extend(ratchet(actual, baseline))
    for error in errors:
        print(error, file=sys.stderr)
    print(
        f"clippy: {sum(s.startswith('site:') for s in actual)} sites; "
        f"{sum(s.startswith('allow:') for s in actual)} exemptions; baseline {len(baseline)}"
    )
    return bool(errors)


def main() -> int:
    return check_boundaries() or check_clippy()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"architecture: {error}")
