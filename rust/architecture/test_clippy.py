"""Deliberately failing compiler fixtures for every core-purity construct family."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import check as clippy
from check import ratchet

ROOT = Path(__file__).resolve().parents[2]
HELPER: Path


class ClippyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global HELPER
        HELPER = clippy.build_helper(ROOT)

    def test_pool_diagnostics_do_not_hide_a_policed_macro_caller(self):
        pool = {
            "file_name": "crates/lk-agents-media/src/pool.rs",
            "line_start": 1,
            "column_start": 1,
            "is_primary": True,
        }
        message = {
            "code": {"code": "clippy::disallowed_types"},
            "message": "use of a disallowed type `std::sync::Mutex`",
            "spans": [pool],
        }
        self.assertEqual(clippy.compiler_locations([message], ROOT), [])
        caller = {**pool, "file_name": "crates/lk-agents-media/src/stream_buffer.rs"}
        pool["expansion"] = {"span": caller}
        locations = clippy.compiler_locations([message], ROOT)
        self.assertEqual([site[0] for site in locations], ["rust/" + caller["file_name"]])

    def test_new_and_broadened_exemptions_fail_even_when_counts_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.rs"
            old = '#[allow(clippy::disallowed_methods, reason = "legacy")]\nfn old() {}\n'
            path.write_text(old)
            original = clippy.inventory(ROOT, HELPER, [str(path)])
            variants = [
                old.replace("fn old", "fn new"),
                old.replace("disallowed_methods", "all"),
                old.replace("allow(", "expect("),
                '#[cfg_attr(any(), allow(clippy::disallowed_methods, reason = "legacy"))]\nfn old() {}\n',
                '#![allow(warnings, reason = "legacy")]\nfn old() {}\n',
            ]
            for text in variants:
                with self.subTest(text=text):
                    path.write_text(text)
                    self.assertTrue(ratchet(clippy.inventory(ROOT, HELPER, [str(path)]), original))

    def test_combined_allow_can_shrink_without_changing_sibling_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.rs"
            path.write_text(
                '#[allow(clippy::disallowed_methods, clippy::disallowed_types, reason = "legacy")] fn old() {}'
            )
            original = clippy.inventory(ROOT, HELPER, [str(path)])
            path.write_text(
                '#[allow(clippy::disallowed_types, reason = "updated explanation")] fn old() {}'
            )
            self.assertFalse(ratchet(clippy.inventory(ROOT, HELPER, [str(path)]), original))

    def test_exemption_identity_includes_trait_method_variant_and_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.rs"
            for old in (
                '#[allow(clippy::disallowed_types, reason = "legacy")] trait T { #[allow(clippy::disallowed_types, reason = "legacy")] fn old(); fn new(); }',
                'enum E { Old { #[allow(clippy::disallowed_types, reason = "legacy")] a: u8 }, New { a: u8 } }',
                'fn f(#[allow(clippy::disallowed_types, reason = "legacy")] old: u8, new: u8) {}',
            ):
                path.write_text(old)
                original = clippy.inventory(ROOT, HELPER, [str(path)])
                changed = (
                    old.replace("fn old", "fn renamed")
                    .replace("Old {", "Renamed {")
                    .replace("old: u8", "renamed: u8")
                )
                path.write_text(changed)
                self.assertTrue(ratchet(clippy.inventory(ROOT, HELPER, [str(path)]), original))

    def test_tuple_field_exemptions_cannot_move_to_a_different_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.rs"
            path.write_text(
                'struct T(#[allow(clippy::disallowed_types, reason = "legacy")] u8, u8);'
            )
            original = clippy.inventory(ROOT, HELPER, [str(path)])
            path.write_text(
                'struct T(u8, #[allow(clippy::disallowed_types, reason = "legacy")] u8);'
            )
            self.assertTrue(ratchet(clippy.inventory(ROOT, HELPER, [str(path)]), original))

    def test_macro_body_exemptions_are_scanned_and_string_literals_are_opaque(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.rs"
            path.write_text(
                'macro_rules! m { () => { #[allow(clippy::disallowed_methods, reason = "legacy")] fn hidden() {} }; }'
            )
            found = clippy.inventory(ROOT, HELPER, [str(path)])
            self.assertEqual(len(found), 1)
            self.assertIn("macro macro_rules m", next(iter(found)))
            path.write_text('fn plain() { let text = r#"#[allow(clippy::disallowed_methods)]"#; }')
            self.assertFalse(clippy.inventory(ROOT, HELPER, [str(path)]))

    def test_core_exemptions_require_a_nonempty_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.rs"
            for reason in ("", ', reason = "   "'):
                path.write_text(f"#[allow(clippy::disallowed_types{reason})] fn missing() {{}}")
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    clippy.inventory(ROOT, HELPER, [str(path)])
                self.assertIn("missing", error.exception.stderr)
                self.assertIn("requires a nonempty reason", error.exception.stderr)

    def test_build_environment_reads_belong_to_build_tools(self):
        message = {"code": {"code": "clippy::disallowed_methods"}, "message": "env read"}
        records = [
            {"reason": "compiler-message", "target": {"kind": [kind]}, "message": message}
            for kind in ("custom-build", "lib")
        ]
        records.extend(
            [{"reason": "compiler-artifact"}, {"reason": "build-finished", "success": True}]
        )
        self.assertEqual(clippy.compiler_messages("\n".join(map(json.dumps, records))), [message])
        self.assertNotIn("rust/crates/lk-agents-media/build.rs", clippy.source_paths(ROOT))
        self.assertNotIn(
            "rust/crates/lk-agents-media/build/blingfire.rs", clippy.source_paths(ROOT)
        )

    def test_symlinked_validation_root_preserves_policy_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "physical"
            source = root / "rust/crates/lk-agents-voice/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text("fn old() {}\n")
            alias = Path(directory) / "alias"
            alias.symlink_to(root, target_is_directory=True)
            message = {
                "code": {"code": "clippy::disallowed_methods"},
                "message": "banned call",
                "spans": [
                    {
                        "is_primary": True,
                        "file_name": "crates/lk-agents-voice/src/lib.rs",
                        "line_start": 1,
                        "column_start": 1,
                        "expansion": None,
                    }
                ],
            }
            self.assertEqual(
                clippy.compiler_locations([message], alias),
                [
                    (
                        "rust/crates/lk-agents-voice/src/lib.rs",
                        1,
                        1,
                        "clippy::disallowed_methods",
                        "banned call",
                    )
                ],
            )
            message["spans"] = []
            with self.assertRaisesRegex(ValueError, "no primary span"):
                clippy.compiler_locations([message], alias)
            message["spans"] = [
                {
                    "is_primary": True,
                    "file_name": "generated-unknown.rs",
                    "line_start": 1,
                    "column_start": 1,
                    "expansion": None,
                }
            ]
            with self.assertRaisesRegex(ValueError, "unmapped core policy"):
                clippy.compiler_locations([message], alias)
            message["spans"][0]["file_name"] = "crates/lk-agents-media/src/pool.rs"
            self.assertEqual(clippy.compiler_locations([message], alias), [])

    def test_empty_partial_and_failed_compiler_transcripts_fail(self):
        for records in (
            [],
            [{"reason": "compiler-artifact"}],
            [{"reason": "build-finished", "success": True}],
            [{"reason": "compiler-artifact"}, {"reason": "build-finished", "success": False}],
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                clippy.compiler_messages("\n".join(map(json.dumps, records)))

    def test_architecture_clippy_selects_only_restricted_packages(self):
        packages = sorted(clippy.PURE)
        command = clippy.command(ROOT, packages)
        selected = [command[i + 1] for i, arg in enumerate(command) if arg == "-p"]
        self.assertEqual(selected, packages)
        self.assertNotIn("--workspace", command)

    def test_compiler_rejects_each_banned_construct_family_and_allowed_item_growth(self):
        self.assertIn("--all-features", clippy.command(ROOT, ["lk-agents-abi"]))
        self.assertNotIn("--all-features", clippy.command(ROOT, ["lk-agents-abi"], "default"))
        self.assertIn("lk-agents-abi/python", clippy.command(ROOT, ["lk-agents-abi"], "python"))
        source = ROOT / "rust/architecture/fixtures.rs"
        env = {**os.environ, "CLIPPY_CONF_DIR": str(ROOT / "rust/architecture")}
        command = [
            "cargo",
            "clippy",
            "--no-deps",
            "--config",
            str(ROOT / "rust/.cargo/config.toml"),
            "--manifest-path",
            str(ROOT / "rust/Cargo.toml"),
            "--locked",
            "-p",
            "lk-agents-codegen",
            "--example",
            "architecture-fixtures",
            "--message-format=json",
            "--",
            *(a for lint in clippy.LINTS for a in ("--force-warn", f"clippy::{lint}")),
        ]
        process = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        messages = clippy.compiler_messages(process.stdout)
        diagnostic_text = "\n".join(m["message"] for m in messages)
        for name in (
            "std::thread::spawn",
            "std::time::Instant::now",
            "std::time::SystemTime::now",
            "RandomState::new",
            "std::process::id",
            "std::env::var",
            "rand::random",
            "next_u64",
            "rand::Rng::random",
            "uuid::Uuid::new_v4",
            "mpsc::channel",
            "unbounded",
            "HashMap",
            "HashSet",
            "Mutex",
            "RwLock",
            "AtomicU64",
            "tokio::task::spawn",
            "tokio::select",
            "tokio::join",
            "tokio::time::sleep",
        ):
            with self.subTest(name=name):
                self.assertIn(name, diagnostic_text)
        locations = set()
        for message in messages:
            lint = (message.get("code") or {}).get("code", "")
            if lint not in {f"clippy::{lint}" for lint in clippy.LINTS}:
                continue
            for span in message["spans"]:
                if not span["is_primary"]:
                    continue
                while span:
                    if Path(span["file_name"]).resolve() == source.resolve() or span[
                        "file_name"
                    ].endswith("architecture/fixtures.rs"):
                        locations.add(
                            (
                                str(source),
                                span["line_start"],
                                span["column_start"],
                                lint,
                                message["message"],
                            )
                        )
                        break
                    span = (span.get("expansion") or {}).get("span")
        self.assertTrue(locations)
        rejected_lines = {site[1] for site in locations}
        for line, text in enumerate(source.read_text().splitlines(), 1):
            if text.strip().startswith(
                (
                    "let ",
                    "tokio::spawn",
                    "tokio::select!",
                    "tokio::join!",
                    "tokio::time::sleep",
                    "std::thread::spawn",
                    "input:",
                    "pub async fn function_exemption",
                )
            ):
                with self.subTest(source_line=text):
                    self.assertIn(line, rejected_lines)
        sites = clippy.site_inventory(ROOT, HELPER, sorted(locations))
        # Both calls under the same legacy allowance must be captured; retaining only
        # the first in a baseline must reject growth without needing a second build.
        old_sites = {s for s in sites if ":fn old:" in s}
        self.assertEqual(len(old_sites), 2)
        self.assertTrue(ratchet(sites, sites - {sorted(old_sites)[-1]}))
        macro_sites = {s for s in sites if ":fn macro_growth:" in s}
        self.assertEqual(len(macro_sites), 2)
        self.assertTrue(ratchet(sites, sites - {sorted(macro_sites)[-1]}))
        ordered = sorted(locations)
        scopes = subprocess.run(
            [
                str(HELPER),
                "--architecture",
                "locations",
                str(source),
                *(f"{s[1]}:{s[2]}" for s in ordered),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        expected = {
            "fn old": "allowed",
            "fn missing_exemption": "denied",
            "fn argument_exemption": "denied",
            "fn async_argument_exemption": "denied",
            "fn function_exemption": "allowed",
        }
        cases = [
            (site, scope.split("\t")[1].split("/")[0])
            for site, scope in zip(ordered, scopes, strict=True)
            if scope.split("\t")[1].split("/")[0] in expected
        ]
        self.assertEqual({owner for _, owner in cases}, set(expected))
        request = "\t".join(
            [
                str(source),
                *(
                    f"{line}:{column}:{lint.removeprefix('clippy::')}:0"
                    for (_, line, column, lint, _), _ in cases
                ),
            ]
        )
        result = subprocess.run(
            [str(HELPER), "--architecture", "coverage-batch"],
            input=request + "\n",
            cwd=ROOT,
            env={**os.environ, "ARCHITECTURE_FEATURES": "", "ARCHITECTURE_CRATE_ROOT": str(source)},
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.splitlines(), [expected[owner] for _, owner in cases])
        # Baseline membership cannot replace the missing source exemption.
        self.assertFalse(ratchet(sites, sites))


class IoPolicyContracts(unittest.TestCase):
    """The I/O policy refuses a raw lock: only lk_agents_net::lock waits without the GIL."""

    def test_the_io_policy_rejects_each_raw_lock(self):
        io = next(policy for policy in clippy.POLICIES if policy.name == "io")
        env = {**os.environ, "CLIPPY_CONF_DIR": str(ROOT / io.conf)}
        command = [
            "cargo",
            "clippy",
            "--no-deps",
            "--config",
            str(ROOT / "rust/.cargo/config.toml"),
            "--manifest-path",
            str(ROOT / "rust/Cargo.toml"),
            "--locked",
            "-p",
            "lk-agents-codegen",
            "--example",
            "io-fixtures",
            "--message-format=json",
            "--",
            *(a for lint in clippy.LINTS for a in ("--force-warn", f"clippy::{lint}")),
        ]
        process = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        diagnostics = [m["message"] for m in clippy.compiler_messages(process.stdout)]
        for name in ("std::sync::Mutex", "std::sync::RwLock", "parking_lot::Mutex", "parking_lot::RwLock"):
            with self.subTest(name=name):
                self.assertTrue(any(f"`{name}`" in d for d in diagnostics), diagnostics)


class CompilerLocationContracts(unittest.TestCase):
    def test_worker_exemption_keeps_other_primary_spans_visible(self):
        def span(path, expansion=None):
            return {
                "is_primary": True,
                "file_name": path,
                "line_start": 1,
                "column_start": 1,
                "expansion": expansion,
            }

        worker = span("crates/lk-agents-media/src/pool.rs")
        core = span("crates/lk-agents-core/src/lib.rs")
        message = {
            "code": {"code": "clippy::disallowed_types"},
            "message": "banned type",
            "spans": [worker, span("unmapped.rs")],
        }
        with self.assertRaisesRegex(ValueError, "unmapped core policy"):
            clippy.compiler_locations([message], ROOT)
        expected = [
            (
                "rust/crates/lk-agents-core/src/lib.rs",
                1,
                1,
                "clippy::disallowed_types",
                "banned type",
            )
        ]
        message["spans"] = [worker, core]
        self.assertEqual(clippy.compiler_locations([message], ROOT), expected)


if __name__ == "__main__":
    unittest.main()
