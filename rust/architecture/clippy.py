"""Run compiler-resolved core purity lints and ratchet explicit Rust AST exemptions."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import tomllib
from check import GENERATED, PURE, check_history, git, ratchet

LINTS = ("disallowed_methods", "disallowed_types", "disallowed_macros")
BASELINE = "rust/architecture/clippy-baseline.json"
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


def source_paths(root: Path) -> list[str]:
    paths = (
        git(root, "ls-files", "-c", "-o", "--exclude-standard", "-z", "--", "rust/crates/*/*.rs")
        .strip("\0")
        .split("\0")
    )
    result = []
    for path in paths:
        if not path or Path(path).parts[2] not in PURE:
            continue
        if Path(path).parts[3] in {"build.rs", "build"}:
            continue
        if not (root / path).is_file():
            continue
        text = (root / path).read_text()
        if not (text.splitlines() and GENERATED.match(text.splitlines()[0])):
            result.append(path)
    if not result:
        raise ValueError("no handwritten core Rust sources found")
    return result


def inventory(root: Path, helper: Path, paths: list[str]) -> set[str]:
    process = subprocess.run(
        [str(helper), "inventory", *paths], cwd=root, text=True, capture_output=True, check=True
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
            "architecture-allows",
        ],
        cwd=root,
        check=True,
    )
    return target_directory(root) / "debug/architecture-allows"


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
        for span in message["spans"]:
            if not span["is_primary"]:
                continue
            while span:
                path = Path(span["file_name"])
                if not path.is_absolute():
                    path = root / "rust" / path
                try:
                    relative = path.resolve().relative_to(root).as_posix()
                except ValueError:
                    relative = ""
                if relative.startswith("rust/crates/") and Path(relative).parts[2] in PURE:
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
        if not candidates:
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
            [str(helper), "locations", path, *(f"{s[1]}:{s[2]}" for s in sites)],
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
            [str(helper), "coverage-batch"],
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


def main() -> int:
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
    baseline = json.loads((root / BASELINE).read_text())
    errors.extend(ratchet(actual, baseline))
    errors.extend(check_history(root, BASELINE, baseline))
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    env = os.environ.copy()
    env["CLIPPY_CONF_DIR"] = str(root / "rust/crates/lk-agents-abi")
    locations = set()
    for mode in ("default", "python", "all"):
        started = time.monotonic()
        process = subprocess.run(
            command(root, crates, mode),
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
        )
        elapsed = time.monotonic() - started
        timing = f"{time.strftime('%H:%M:%S')} s0-architecture clippy-{mode} rc={process.returncode} {elapsed:.2f}s"
        print(timing, flush=True)
        log = Path(os.environ.get("ARCHITECTURE_TIMINGS_DIR", "/tmp/plan-align")) / "timings.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            stream.write(timing + "\n")
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


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"clippy architecture: {error}")
