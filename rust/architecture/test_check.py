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
    length_violations,
    ratchet,
    tracked_rust_paths,
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
        for banned in ("tokio", "livekit", "livekit-ffi", "futures-util", "pyo3", "reqwest"):
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
                        "lk-agents-voice": ["lk-agents-core", "lk-agents-abi"],
                        "lk-agents-core": ["lk-agents-abi", "livekit-protocol"],
                        "lk-agents-a2a": ["lk-agents-storage"],
                        "lk-agents-fuzz": ["lk-agents-voice", "lk-agents-abi"],
                    }
                )
            ),
            set(),
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

    def test_generated_marker_exempts_output_but_not_a_string_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "output.rs"
            path.write_text("// @generated by lk-agents-codegen\n" + "\n" * 1251)
            self.assertEqual(length_violations(root, ["output.rs"]), {})
            path.write_text('const NOTE: &str = "GENERATED";\n' + "\n" * 1251)
            self.assertEqual(length_violations(root, ["output.rs"]), {"output.rs": 1252})

    def test_marker_inside_raw_string_cannot_exempt_handwritten_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "handwritten.rs").write_text(
                'const NOTE: &str = r#"\n// @generated by test\n"#;\n' + "\n" * 1251
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
