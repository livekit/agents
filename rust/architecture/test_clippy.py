"""Deliberately failing compiler fixtures for every core-purity construct family."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import clippy
from check import ratchet

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "rust/target/debug/architecture-allows"


class ClippyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            HELPER.exists()
            and HELPER.stat().st_mtime >= (ROOT / "rust/architecture/allows.rs").stat().st_mtime
        ):
            return
        subprocess.run(
            [
                "cargo",
                "build",
                "--config",
                str(ROOT / "rust/.cargo/config.toml"),
                "--quiet",
                "--manifest-path",
                str(ROOT / "rust/Cargo.toml"),
                "--locked",
                "--target-dir",
                str(ROOT / "rust/target"),
                "-p",
                "lk-agents-codegen",
                "--bin",
                "architecture-allows",
            ],
            cwd=ROOT,
            check=True,
        )

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

    def test_empty_partial_and_failed_compiler_transcripts_fail(self):
        for records in (
            [],
            [{"reason": "compiler-artifact"}],
            [{"reason": "build-finished", "success": True}],
            [{"reason": "compiler-artifact"}, {"reason": "build-finished", "success": False}],
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                clippy.compiler_messages("\n".join(map(json.dumps, records)))

    def test_configuration_removal_fails_with_crate_and_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rust/crates/lk-agents-abi/clippy.toml"
            path.parent.mkdir(parents=True)
            path.write_text(
                (ROOT / "rust/crates/lk-agents-abi/clippy.toml")
                .read_text()
                .replace('  { path = "std::thread::spawn",', '  { path = "not::the::banned_spawn",')
            )
            errors = clippy.configuration_errors(root, ["lk-agents-abi"])
            self.assertTrue(any("lk-agents-abi" in e and "std::thread::spawn" in e for e in errors))
            path.with_name(".clippy.toml").write_text("")
            self.assertTrue(
                any("shadowing" in e for e in clippy.configuration_errors(root, ["lk-agents-abi"]))
            )

    def test_compiler_rejects_each_banned_construct_family_and_allowed_item_growth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "Cargo.toml").write_text("""[package]
name = "architecture-negative-fixture"
version = "0.0.0"
edition = "2024"
[workspace]
[features]
native = []
python = []
[dependencies]
tokio = { version = "=1.53.1", features = ["rt", "macros", "time", "sync"] }
futures = "=0.3.34"
rand = "=0.9.5"
uuid = { version = "=1.28.0", features = ["v4"] }
crossbeam-channel = "=0.5.17"
parking_lot = "=0.12.5"
""")
            (root / "src/lib.rs").write_text("""#![allow(unused, deprecated)]
#![allow(clippy::disallowed_macros, reason = "legacy macro sites")]
use rand::{Rng, RngCore};
pub fn ambient(rng: &mut impl RngCore) {
    let _ = std::thread::spawn(|| {});
    let _ = std::time::Instant::now();
    let _ = std::time::SystemTime::now();
    let _ = std::collections::hash_map::RandomState::new();
    let _ = std::process::id();
    let _ = std::env::var("FIXTURE");
    let _ = rand::random::<u32>();
    let _ = rng.next_u64();
    let _ = rng.random::<u32>();
    let _ = uuid::Uuid::new_v4();
    let _ = std::sync::mpsc::channel::<()>();
    let _ = tokio::sync::mpsc::unbounded_channel::<()>();
    let _ = futures::channel::mpsc::unbounded::<()>();
    let _ = crossbeam_channel::unbounded::<()>();
    let _: std::collections::HashMap<u8, u8> = Default::default();
    let _: std::collections::HashSet<u8> = Default::default();
    let _ = std::sync::Mutex::new(());
    let _ = std::sync::RwLock::new(());
    let _ = tokio::sync::Mutex::new(());
    let _ = tokio::sync::RwLock::new(());
    let _ = parking_lot::Mutex::new(());
    let _ = parking_lot::RwLock::new(());
    let _ = std::sync::atomic::AtomicU64::new(0);
}
#[cfg(not(feature = "python"))]
pub fn default_only() {
    let _ = std::env::var("DEFAULT_ONLY");
}
#[cfg(feature = "native")]
pub fn optional_native() {
    let _ = std::thread::Builder::new().spawn(|| {});
}
pub fn through_handle(handle: &tokio::runtime::Handle) {
    let _ = handle.spawn(async {});
}
pub async fn tasks() {
    tokio::spawn(async {});
    tokio::select! { _ = async {} => {} }
    tokio::join!(async {}, async {});
    tokio::time::sleep(std::time::Duration::ZERO).await;
}
#[allow(clippy::disallowed_methods, reason = "legacy")]
pub fn old() { std::thread::spawn(|| {}); }
""")
            self.assertIn("--all-features", clippy.command(ROOT, ["lk-agents-abi"]))
            self.assertNotIn("--all-features", clippy.command(ROOT, ["lk-agents-abi"], "default"))
            self.assertIn("lk-agents-abi/python", clippy.command(ROOT, ["lk-agents-abi"], "python"))
            env = os.environ.copy()
            env["CLIPPY_CONF_DIR"] = str(ROOT / "rust/crates/lk-agents-abi")
            env["CARGO_TARGET_DIR"] = str(ROOT / "rust/target")
            command = [
                "cargo",
                "clippy",
                "--no-deps",
                "--config",
                str(ROOT / "rust/.cargo/config.toml"),
                "--all-features",
                "--manifest-path",
                str(root / "Cargo.toml"),
                "--message-format=json",
                "--",
                *(a for lint in clippy.LINTS for a in ("--force-warn", f"clippy::{lint}")),
            ]
            process = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            all_messages = clippy.compiler_messages(process.stdout)
            default_command = [arg for arg in command if arg != "--all-features"]
            default_process = subprocess.run(
                default_command, env=env, cwd=root, capture_output=True, text=True
            )
            self.assertEqual(
                default_process.returncode, 0, default_process.stderr + default_process.stdout
            )
            messages = all_messages + clippy.compiler_messages(default_process.stdout)
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
            rejected_lines = set()
            for message in messages:
                if (message.get("code") or {}).get("code") not in {
                    f"clippy::{lint}" for lint in clippy.LINTS
                }:
                    continue
                for span in message["spans"]:
                    if not span["is_primary"]:
                        continue
                    while span:
                        if span["file_name"].endswith("src/lib.rs"):
                            rejected_lines.add(span["line_start"])
                            break
                        span = (span.get("expansion") or {}).get("span")
            for line, text in enumerate((root / "src/lib.rs").read_text().splitlines(), 1):
                if text.strip().startswith(
                    ("let ", "tokio::spawn", "tokio::select!", "tokio::join!", "tokio::time::sleep")
                ):
                    with self.subTest(source_line=text):
                        self.assertIn(line, rejected_lines)

            def spawn_locations(diagnostics):
                return [
                    (
                        str(root / "src/lib.rs"),
                        span["line_start"],
                        span["column_start"],
                        "clippy::disallowed_methods",
                        message["message"],
                    )
                    for message in diagnostics
                    if "disallowed method `std::thread::spawn`" in message["message"]
                    for span in message["spans"]
                    if span["is_primary"]
                ]

            def macro_locations(diagnostics):
                result = set()
                for message in diagnostics:
                    if (message.get("code") or {}).get("code") != "clippy::disallowed_macros":
                        continue
                    for span in message["spans"]:
                        if not span["is_primary"]:
                            continue
                        while span:
                            if span["file_name"].endswith("src/lib.rs"):
                                result.add(
                                    (
                                        str(root / "src/lib.rs"),
                                        span["line_start"],
                                        span["column_start"],
                                        "clippy::disallowed_macros",
                                        message["message"],
                                    )
                                )
                                break
                            span = (span.get("expansion") or {}).get("span")
                return sorted(result)

            original_macro_locations = macro_locations(all_messages)
            original_macro_sites = clippy.site_inventory(ROOT, HELPER, original_macro_locations)
            original_sites = clippy.site_inventory(ROOT, HELPER, spawn_locations(all_messages))
            original_allows = clippy.inventory(ROOT, HELPER, [str(root / "src/lib.rs")])
            # Force-warn must defeat a pre-existing allow, preventing a second banned site there.
            source = root / "src/lib.rs"
            source.write_text(
                source.read_text()
                .replace(
                    "pub fn old() { std::thread::spawn(|| {}); }",
                    "pub fn old() { std::thread::spawn(|| {}); std::thread::spawn(|| {}); }",
                )
                .replace(
                    "tokio::select! { _ = async {} => {} }",
                    "tokio::select! { _ = async {} => {} } tokio::select! { _ = async {} => {} }",
                )
            )
            changed = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True)
            self.assertEqual(changed.returncode, 0, changed.stderr + changed.stdout)
            original_spawns = [
                m for m in all_messages if "disallowed method `std::thread::spawn`" in m["message"]
            ]
            changed_spawns = [
                m
                for m in clippy.compiler_messages(changed.stdout)
                if "disallowed method `std::thread::spawn`" in m["message"]
            ]
            self.assertEqual(len(changed_spawns), len(original_spawns) + 1)
            self.assertEqual(clippy.inventory(ROOT, HELPER, [str(source)]), original_allows)
            changed_sites = clippy.site_inventory(
                ROOT, HELPER, spawn_locations(clippy.compiler_messages(changed.stdout))
            )
            self.assertTrue(ratchet(changed_sites, original_sites))
            changed_macros = macro_locations(clippy.compiler_messages(changed.stdout))
            self.assertEqual(len(changed_macros), len(original_macro_locations) + 1)
            self.assertTrue(
                ratchet(clippy.site_inventory(ROOT, HELPER, changed_macros), original_macro_sites)
            )
            # The same legacy site remains permitted by its site identity after deleting the
            # explicit allow. The ordinary deny pass must still reject the missing exemption.
            old_call = "pub fn old() { std::thread::spawn(|| {}); }\n"
            source.write_text(
                '#[allow(clippy::disallowed_methods, reason = "legacy")]\n' + old_call
            )
            deny_command = ["-D" if arg == "--force-warn" else arg for arg in command]
            allowed = subprocess.run(
                deny_command, env=env, cwd=root, capture_output=True, text=True
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr + allowed.stdout)
            source.write_text(old_call)
            denied = subprocess.run(deny_command, env=env, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            denied_messages = [
                record["message"]
                for record in (json.loads(line) for line in denied.stdout.splitlines())
                if record.get("reason") == "compiler-message"
            ]
            retained_sites = clippy.site_inventory(ROOT, HELPER, spawn_locations(denied_messages))
            self.assertTrue(retained_sites)
            self.assertFalse(ratchet(retained_sites, original_sites))


if __name__ == "__main__":
    unittest.main()
