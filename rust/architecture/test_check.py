"""Counterexamples for the Cargo-closure and tracked-file architecture contracts."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from check import (
    check_history,
    dependency_violations,
    executor_violations,
    host_call_violations,
    length_violations,
    path_violations,
    pyo3_violations,
    ratchet,
    read_baselines,
    tracked_rust_paths,
    use_violations,
    warning_violations,
)


def metadata(edges):
    names = sorted(set(edges) | {name for deps in edges.values() for name in deps})
    return {
        "packages": [{"id": name, "name": name} for name in names],
        "workspace_members": [name for name in names if name.startswith("lk-agents-")],
        "resolve": {
            "nodes": [
                {"id": name, "deps": [{"pkg": dep} for dep in edges.get(name, [])]}
                for name in names
            ]
        },
    }


# A scratch repository must not inherit the caller's repository or private index.
_GIT_FREE_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


class DependencyContracts(unittest.TestCase):
    def test_cargo_all_features_exposes_optional_sdk_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers=["voice", "sdk"]\nresolver="3"\n'
            )
            for directory, name in (("voice", "lk-agents-voice"), ("sdk", "livekit-ffi")):
                crate = root / directory
                (crate / "src").mkdir(parents=True)
                (crate / "src/lib.rs").write_text("")
                manifest = f'[package]\nname="{name}"\nversion="0.0.0"\nedition="2024"\n'
                if directory == "voice":
                    manifest += '[dependencies]\nlivekit-ffi={path="../sdk", optional=true}\n'
                (crate / "Cargo.toml").write_text(manifest)
            command = ["cargo", "metadata", "--offline", "--format-version", "1"]
            default = json.loads(
                subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout
            )
            enabled = json.loads(
                subprocess.run(
                    command + ["--all-features"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            self.assertEqual(dependency_violations(default), set())
            self.assertIn("dependency:lk-agents-voice:livekit-ffi", dependency_violations(enabled))

    def test_all_feature_resolved_closure_rejects_sdk_and_async_dependencies(self):
        for banned in (
            "tokio",
            "livekit",
            "livekit-ffi",
            "futures-util",
            "pyo3",
            "reqwest",
            "rusqlite",
        ):
            with self.subTest(banned=banned):
                actual = dependency_violations(
                    metadata(
                        {"lk-agents-voice": ["optional-adapter"], "optional-adapter": [banned]}
                    )
                )
                self.assertIn(f"dependency:lk-agents-voice:{banned}", actual)

    def test_direction_rejects_core_to_runtime_even_without_async_dependencies(self):
        actual = dependency_violations(metadata({"lk-agents-core": ["lk-agents-runtime"]}))
        self.assertIn("direction:lk-agents-core:lk-agents-runtime", actual)

    def test_test_support_is_confined_to_test_dependencies(self):
        for consumer in ("lk-agents-voice", "lk-agents-runtime", "lk-agents-py"):
            with self.subTest(consumer=consumer):
                graph = metadata(
                    {
                        consumer: ["lk-agents-test-support"],
                        "lk-agents-test-support": ["lk-agents-abi"],
                    }
                )
                violation = f"direction:{consumer}:lk-agents-test-support"
                self.assertIn(violation, dependency_violations(graph))
                node = next(node for node in graph["resolve"]["nodes"] if node["id"] == consumer)
                node["deps"][0]["dep_kinds"] = [{"kind": "dev", "target": None}]
                self.assertNotIn(violation, dependency_violations(graph))
        graph = metadata(
            {
                "lk-agents-fuzz": ["lk-agents-test-support"],
                "lk-agents-test-support": ["lk-agents-abi"],
            }
        )
        self.assertNotIn(
            "direction:lk-agents-fuzz:lk-agents-test-support", dependency_violations(graph)
        )

    def test_the_dst_may_run_the_runtime_executors(self):
        graph = metadata({"lk-agents-fuzz": ["lk-agents-runtime"], "lk-agents-abi": []})
        self.assertNotIn("direction:lk-agents-fuzz:lk-agents-runtime", dependency_violations(graph))

    def test_the_dst_may_run_real_storage(self):
        graph = metadata({"lk-agents-fuzz": ["lk-agents-storage"], "lk-agents-abi": []})
        self.assertNotIn("direction:lk-agents-fuzz:lk-agents-storage", dependency_violations(graph))
        graph = metadata({"lk-agents-fuzz": ["lk-agents-inference"], "lk-agents-abi": []})
        self.assertIn("direction:lk-agents-fuzz:lk-agents-inference", dependency_violations(graph))

    def test_direction_allows_only_the_documented_graph(self):
        for crate, target in (
            ("lk-agents-net", "lk-agents-media"),
            ("lk-agents-test-support", "lk-agents-runtime"),
            ("lk-agents-bench-steps", "lk-agents-inference"),
            ("lk-agents-media", "lk-agents-core"),
        ):
            with self.subTest(edge=f"{crate} -> {target}"):
                actual = dependency_violations(metadata({crate: [target], "lk-agents-abi": []}))
                self.assertIn(f"direction:{crate}:{target}", actual)

    def test_shared_abi_debt_keeps_its_owning_manifest_identity(self):
        actual = dependency_violations(
            metadata({"lk-agents-core": ["lk-agents-abi"], "lk-agents-abi": ["tokio"]})
        )
        self.assertEqual(actual, {"dependency:lk-agents-abi:tokio"})
        changed = dependency_violations(
            metadata({"lk-agents-core": ["lk-agents-abi", "tokio"], "lk-agents-abi": ["tokio"]})
        )
        self.assertIn("dependency:lk-agents-core:tokio", changed)

    def test_record_dependency_and_documented_composition_are_allowed(self):
        self.assertEqual(
            dependency_violations(
                metadata(
                    {
                        "lk-agents-voice": ["lk-agents-core", "lk-agents-abi", "lk-agents-media"],
                        "lk-agents-inference": ["lk-agents-abi", "lk-agents-net", "tokio"],
                        "lk-agents-worker": ["lk-agents-abi", "lk-agents-net", "reqwest"],
                        "lk-agents-net": ["lk-agents-abi", "tokio"],
                        "lk-agents-storage": ["lk-agents-abi", "lk-agents-net", "rusqlite"],
                        "lk-agents-runtime": [
                            "lk-agents-inference",
                            "lk-agents-worker",
                            "lk-agents-storage",
                        ],
                        "lk-agents-core": ["lk-agents-abi", "livekit-protocol"],
                        "lk-agents-fuzz": ["lk-agents-voice", "lk-agents-abi"],
                    }
                )
            ),
            set(),
        )

    def test_async_crates_never_reach_the_runtime_or_each_other_upward(self):
        self.assertEqual(
            dependency_violations(
                metadata(
                    {
                        "lk-agents-core": ["lk-agents-abi"],
                        "lk-agents-inference": ["lk-agents-runtime"],
                        "lk-agents-net": ["lk-agents-inference"],
                    }
                )
            ),
            {
                "direction:lk-agents-inference:lk-agents-runtime",
                "direction:lk-agents-net:lk-agents-inference",
            },
        )

    def test_incomplete_metadata_cannot_pass(self):
        with self.assertRaises(ValueError):
            dependency_violations({"packages": [], "workspace_members": [], "resolve": None})
        with self.assertRaises(ValueError):
            dependency_violations(
                metadata({"lk-agents-voice": ["tokio"]}) | {"resolve": {"nodes": []}}
            )


class LengthContracts(unittest.TestCase):
    @patch.dict(os.environ, _GIT_FREE_ENV, clear=True)
    def test_git_selection_checks_tracked_files_and_retires_working_deletions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
            for name in ("keep.rs", "removed.rs", "untracked.rs"):
                (root / name).write_text("// handwritten\n" * 1251)
            subprocess.run(
                ["git", "-C", str(root), "add", "keep.rs", "removed.rs"],
                check=True,
                capture_output=True,
            )
            self.assertEqual(set(tracked_rust_paths(root)), {"keep.rs", "removed.rs"})
            (root / "removed.rs").unlink()
            self.assertEqual(tracked_rust_paths(root), ["keep.rs"])
            self.assertEqual(length_violations(root, tracked_rust_paths(root)), {"keep.rs": 1251})

    def test_1251_line_handwritten_file_fails_with_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "new.rs"
            path.write_text("// handwritten\n" * 1251)
            self.assertEqual(length_violations(root, ["new.rs"]), {"new.rs": 1251})
            path.write_text("// handwritten\n" * 1250)
            self.assertEqual(length_violations(root, ["new.rs"]), {})

    def test_only_the_exact_generator_header_exempts_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "output.rs"
            path.write_text("//! GENERATED FILE -- DO NOT EDIT.\n" + "\n" * 1251)
            self.assertEqual(length_violations(root, ["output.rs"]), {})
            for first in (
                'const NOTE: &str = "GENERATED";',
                "//! Generated host views of native synthesis streams.",
                "// @generated by lk-agents-codegen",
                "//! GENERATED",
            ):
                path.write_text(first + "\n" + "\n" * 1251)
                self.assertEqual(length_violations(root, ["output.rs"]), {"output.rs": 1252}, first)

    def test_marker_inside_raw_string_cannot_exempt_handwritten_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "handwritten.rs").write_text(
                'const NOTE: &str = r#"\n//! GENERATED FILE -- DO NOT EDIT.\n"#;\n' + "\n" * 1251
            )
            self.assertEqual(length_violations(root, ["handwritten.rs"]), {"handwritten.rs": 1254})

    def test_same_count_different_location_and_existing_file_growth_fail(self):
        self.assertTrue(ratchet({"new.rs": 1251}, {"old.rs": 1500}))
        self.assertTrue(ratchet({"old.rs": 1501}, {"old.rs": 1500}))
        self.assertFalse(ratchet({"old.rs": 1499}, {"old.rs": 1500}))

    def test_missing_tracked_file_cannot_silently_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                length_violations(Path(tmp), ["missing.rs"])


class BaselineContracts(unittest.TestCase):
    def test_same_count_replacement_is_rejected(self):
        self.assertTrue(ratchet({"new"}, {"old"}))
        self.assertFalse(ratchet(set(), {"old"}))

    @patch("check.check_history", return_value=[])
    def test_first_sqlite_baseline_cannot_receive_retired_core_exemptions(self, history):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {"legacy": "legacy.json", "sqlite": "sqlite.json"}
            legacy = "dependency:lk-agents-voice:tokio"
            sqlite = "dependency:lk-agents-storage:rusqlite"
            for family, identity in (("legacy", legacy), ("sqlite", sqlite)):
                (root / paths[family]).write_text(json.dumps([identity]))
            baseline, errors = read_baselines(root, paths)
            self.assertFalse(errors)
            self.assertEqual(baseline, {legacy, sqlite})
            self.assertEqual(history.call_count, 2)
            # Removing an older exemption cannot transfer it into a new family's first baseline.
            (root / paths["legacy"]).write_text("[]")
            (root / paths["sqlite"]).write_text(json.dumps([legacy]))
            _, errors = read_baselines(root, paths)
            self.assertTrue(any(legacy in error for error in errors))

    @patch.dict(os.environ, _GIT_FREE_ENV, clear=True)
    def test_committed_baseline_cannot_be_expanded_or_reintroduced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Inherited GIT_* variables (hooks, commit scripts) would aim these commands at the
            # outer repository, so the scratch repository gets a clean environment.
            env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
            env.update(
                GIT_AUTHOR_NAME="Test",
                GIT_AUTHOR_EMAIL="test@example.com",
                GIT_COMMITTER_NAME="Test",
                GIT_COMMITTER_EMAIL="test@example.com",
            )

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                ).stdout

            git("init", "-q")
            path = root / "baseline.json"
            path.write_text(json.dumps(["old"]))
            git("add", "baseline.json")
            git("commit", "-qm", "bootstrap")
            self.assertFalse(check_history(root, "baseline.json", {"old"}))
            self.assertTrue(check_history(root, "baseline.json", {"old", "new"}))
            main_branch = git("symbolic-ref", "--short", "HEAD").strip()
            git("checkout", "-qb", "retirement")
            path.write_text("[]")
            git("add", "baseline.json")
            git("commit", "-qm", "shrink")
            self.assertTrue(check_history(root, "baseline.json", {"old"}))
            git("checkout", "-q", main_branch)
            git("commit", "-qm", "unrelated", "--allow-empty")
            git("merge", "-q", "-s", "ours", "-m", "merge retirement", "retirement")
            self.assertTrue(check_history(root, "baseline.json", {"old"}))


if __name__ == "__main__":
    unittest.main()


class PathContracts(unittest.TestCase):
    # Catches: a crate compiling another crate's file through `#[path]` (it then escapes that
    # crate's dependency and Clippy policy), bare or under `cfg_attr`; a module file inside the
    # crate's own tree stays allowed.
    def test_a_path_module_outside_its_crate_tree_is_a_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "rust/crates/lk-agents-runtime/src"
            worker = root / "rust/crates/lk-agents-worker/src"
            runtime.mkdir(parents=True)
            worker.mkdir(parents=True)
            (worker / "python.rs").write_text("")
            (runtime / "inner.rs").write_text("")
            (runtime / "lib.rs").write_text(
                '#[path = "inner.rs"]\nmod inner;\n'
                '#[path = "../../lk-agents-worker/src/python.rs"]\nmod python;\n'
                '#[cfg_attr(feature = "x", path = "../../lk-agents-worker/src/python.rs")]\nmod other;\n'
            )
            self.assertEqual(
                path_violations(root, ["rust/crates/lk-agents-runtime/src/lib.rs"]),
                {
                    "rust/crates/lk-agents-runtime/src/lib.rs"
                    " -> rust/crates/lk-agents-worker/src/python.rs"
                },
            )


class HostCallContracts(unittest.TestCase):
    """Every Rust-to-host call goes through HostInvoker's Python entry."""

    def test_a_host_call_outside_the_entry_is_a_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                # A generated proxy calling the host object itself, bypassing the entry.
                "rust/crates/lk-agents-py/src/generated/llm.rs": 'use ::pyo3::prelude::*;\nfn f(o: &Bound<\'_, PyAny>) { o.call_method1("chat", ()); }\n',
                "rust/crates/lk-agents-py/src/turn.rs": 'use pyo3::prelude::*;\nfn f(l: &Bound<\'_, PyAny>) { l.call_method1("call_soon_threadsafe", ()); }\n',
                "rust/crates/lk-agents-runtime/src/bridge.rs": "use pyo3_async_runtimes::tokio::future_into_py;\n",
                # The entry, a generated call through it, a test harness, a generator, plain Rust.
                "rust/crates/lk-agents-py/src/host/mod.rs": "use pyo3::prelude::*;\nfn f(o: &Bound<'_, PyAny>) { o.call0(); }\n",
                "rust/crates/lk-agents-py/src/generated/stt.rs": "use ::pyo3::prelude::*;\nfn f() { crate::host::future_into_py(py, fut); }\n",
                "rust/crates/lk-agents-py/tests/agent_host.rs": "use pyo3::prelude::*;\nfn f(o: &Bound<'_, PyAny>) { o.call1(()); }\n",
                "rust/crates/lk-agents-codegen/src/proxy.rs": '"::pyo3 o.call_method1("\n',
                "rust/crates/lk-agents-voice/src/lib.rs": "fn f(c: &C) { c.call(1); }\n",
            }
            for path, text in files.items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(text)
            self.assertEqual(
                host_call_violations(root, list(files)),
                {
                    "host-call:rust/crates/lk-agents-py/src/generated/llm.rs",
                    "host-call:rust/crates/lk-agents-py/src/turn.rs",
                    "host-call:rust/crates/lk-agents-runtime/src/bridge.rs",
                },
            )


class ExecutorContracts(unittest.TestCase):
    """Decision 92: the DST's scheduler runs the runtime executors."""

    def test_tokio_runtime_use_and_unbiased_selects_are_counted_outside_tests_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "rust/crates/lk-agents-runtime/src/room_io/owner.rs": (
                    "fn f() { tokio::spawn(a); tokio::select! { x = a => {} };"
                    " tokio::select! { biased; x = a => {} } }\n"
                    "#[cfg(test)]\nmod tests { fn t() { tokio::time::sleep(d); } }\n"
                ),
                # Code after an out-of-line test module still counts.
                "rust/crates/lk-agents-runtime/src/rtc/room.rs": (
                    "#[cfg(test)]\nmod tests;\nfn f() { tokio::spawn(a); }\n"
                ),
                "rust/crates/lk-agents-runtime/src/avatar/output.rs": (
                    "use tokio::time::sleep;\nfn f() { Handle::current(); }\n"
                ),
                "rust/crates/lk-agents-runtime/src/voice/api_tests/playback.rs": "fn t() { tokio::spawn(a); }\n",
                "rust/crates/lk-agents-runtime/src/room_io/input_tests.rs": "fn t() { tokio::spawn(a); }\n",
                "rust/crates/lk-agents-runtime/src/avatar/runner.rs": (
                    "fn f() { tokio::spawn(a); tokio::select! { x = a => {} };"
                    " tokio::select! { biased; x = a => {} } }\n"
                ),
                # Comments, doc comments and strings name `tokio::spawn` without using it.
                "rust/crates/lk-agents-runtime/src/voice/reply.rs": (
                    "/// Spawned through the injected Spawner, never `tokio::spawn`.\n"
                    'fn f() { // not tokio::time\n let s = "tokio::spawn"; /* select! { x } */ }\n'
                ),
                # Outside the executors: the process runtime owns the Tokio runtime.
                "rust/crates/lk-agents-runtime/src/process.rs": "fn f() { tokio::spawn(a); }\n",
            }
            for path, text in files.items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(text)
            owner = "executor:rust/crates/lk-agents-runtime/src/room_io/owner.rs"
            avatar = "executor:rust/crates/lk-agents-runtime/src/avatar/output.rs"
            self.assertEqual(
                executor_violations(root, list(files)),
                {
                    "executor:rust/crates/lk-agents-runtime/src/avatar/runner.rs:tokio-spawn": 1,
                    "executor:rust/crates/lk-agents-runtime/src/avatar/runner.rs:unbiased-select": 1,
                    f"{owner}:tokio-spawn": 1,
                    f"{owner}:unbiased-select": 1,
                    f"{avatar}:tokio-time": 1,
                    f"{avatar}:tokio-runtime": 1,
                    "executor:rust/crates/lk-agents-runtime/src/rtc/room.rs:tokio-spawn": 1,
                },
            )


class Pyo3Contracts(unittest.TestCase):
    """Decision 74: pyo3 is the extension crate's alone."""

    def test_uses_are_counted_per_file_item_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "rust/crates/lk-agents-abi/src/err.rs": (
                    "use pyo3::prelude::*; // pyo3::Python in a comment\n"
                    "impl FromHostErr for E {\n"
                    "    fn from_host(py: pyo3::Python<'_>) -> Self {\n"
                    "        let s = \"pyo3::fake\"; let c = '{';\n"
                    "        if s.is_empty() { pyo3::Python::attach(|_| ()); }\n"
                    "        pyo3::Python::attach(|_| ())\n"
                    "    }\n"
                    "}\n"
                ),
                "rust/crates/lk-agents-voice/src/host.rs": (
                    "#[async_trait::async_trait]\npub trait Host<'a> { async fn f(&'a self); }\n"
                ),
                "rust/crates/lk-agents-inference/src/net.rs": "#[async_trait::async_trait]\ntrait Free {}\n",
                "rust/crates/lk-agents-py/src/lib.rs": "use pyo3::prelude::*;\n",
            }
            for path, text in files.items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(text)
            uses = use_violations(root, list(files))
            err = "pyo3-use:rust/crates/lk-agents-abi/src/err.rs"
            self.assertEqual(
                uses,
                {
                    f"{err}:<file>:pyo3::prelude": 1,
                    f"{err}:impl FromHostErr for E::from_host:pyo3::Python": 1,
                    f"{err}:impl FromHostErr for E::from_host:pyo3::Python::attach": 2,
                    "async-trait-use:rust/crates/lk-agents-voice/src/host.rs:Host:async_trait::async_trait": 1,
                },
            )
            # Another use in an item already carrying one grows its count; removing a use
            # elsewhere does not pay for it.
            grown = dict(
                uses, **{f"{err}:impl FromHostErr for E::from_host:pyo3::Python::attach": 3}
            )
            del grown[f"{err}:<file>:pyo3::prelude"]
            self.assertEqual(
                ratchet(grown, uses),
                [f"grew: {err}:impl FromHostErr for E::from_host:pyo3::Python::attach: 2 -> 3"],
            )

    def test_pyo3_outside_the_extension_crate_is_a_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "rust/crates/lk-agents-runtime/src/host.rs": "use pyo3::prelude::*;\n",
                "rust/crates/lk-agents-abi/src/python/generated/llm.rs": "//! GENERATED\npyo3::Python::attach(|_| ());\n",
                "rust/crates/lk-agents-py/src/lib.rs": "use pyo3::prelude::*;\n",
                "rust/crates/lk-agents-codegen/src/proxy.rs": '"::pyo3::Python::attach"\n',
                "rust/crates/lk-agents-voice/src/lib.rs": "pub fn plain() {}\n",
            }
            for path, text in files.items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(text)
            edges = {
                "lk-agents-runtime": ["pyo3", "tokio"],
                "lk-agents-worker": ["pyo3-async-runtimes"],
                "lk-agents-py": ["pyo3", "pyo3-log"],
                "lk-agents-codegen": [],
            }
            self.assertEqual(
                pyo3_violations(metadata(edges), root, list(files)),
                {
                    "pyo3-dependency:lk-agents-runtime:pyo3",
                    "pyo3-dependency:lk-agents-worker:pyo3-async-runtimes",
                    "pyo3-file:rust/crates/lk-agents-runtime/src/host.rs",
                    "pyo3-file:rust/crates/lk-agents-abi/src/python/generated/llm.rs",
                },
            )


class WarningContracts(unittest.TestCase):
    NAMES = {f"id:{name}": name for name in ("lk-agents-core", "lk-agents-voice")}

    def transcript(self, *warnings, success=True):
        artifacts = [
            {"reason": "compiler-artifact", "package_id": f"id:{name}"}
            for name in ("lk-agents-core", "lk-agents-voice")
        ]
        messages = [
            {
                "reason": "compiler-message",
                "package_id": f"id:{name}",
                "message": {"level": "warning", "message": "unused", "rendered": name},
            }
            for name in warnings
        ]
        return [*messages, *artifacts, {"reason": "build-finished", "success": success}]

    @patch("check.WARNING_CLEAN", {"lk-agents-core"})
    def test_only_warning_clean_crates_fail_on_a_warning(self):
        self.assertEqual(warning_violations(self.transcript("lk-agents-voice"), self.NAMES), [])
        self.assertEqual(
            warning_violations(self.transcript("lk-agents-core"), self.NAMES), ["lk-agents-core"]
        )

    @patch("check.WARNING_CLEAN", {"lk-agents-core", "lk-agents-net"})
    def test_incomplete_or_failed_transcripts_cannot_pass(self):
        for records in ([], self.transcript(success=False), self.transcript()):
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                warning_violations(records, self.NAMES)
