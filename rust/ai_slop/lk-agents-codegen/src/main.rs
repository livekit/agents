//! `cargo run -p lk-agents-codegen -- --write` (in `rust/`): resolve ABI declarations and emit bindings.
//! `--write` regenerates in place; `--check` exits 1 if any file would change (CI).
//!
//! Declaration discovery follows Rust modules. ABI owns the shared contracts; each language
//! backend owns its interpreter interfaces. Generated files are outputs, never discovery inputs.

mod accessors;
mod adapter;
mod architecture;
mod bases;
mod class_stub;
mod decl;
mod exceptions;
mod facade;
mod forward;
mod implementations;
mod imports;
mod native_enums;
mod parse;
mod placement;
mod proxy;
mod python;
mod records;
mod shape_error;
mod surface;
#[cfg(test)]
#[path = "codegen_tests.rs"]
mod tests;

use std::path::{Path, PathBuf};

/// The authoritative v2 Python project. Generation must not recreate the retired v1 package.
const PYTHON_PACKAGES: &[&str] = &["livekit-agents-next"];
const PYTHON_GENERATED: &str = "livekit/agents/_generated";
/// Where `_core.pyi` sits, beside the extension.
const PYTHON_CORE: &str = "livekit/agents";
/// The names `dir(_core)` returns, which the registration test of `lk-agents-py` reads.
const CORE_NAMES: &str = "rust/crates/lk-agents-py/src/generated/core_names.json";

fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--architecture") {
        return match architecture::run(&args[1..]) {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("architecture: {error}");
                std::process::ExitCode::FAILURE
            }
        };
    }
    let check = std::env::args().any(|a| a == "--check");
    let write = std::env::args().any(|a| a == "--write");
    if check == write {
        eprintln!("usage: lk-agents-codegen (--write | --check)");
        return std::process::ExitCode::from(2);
    }
    match run(check) {
        Ok(0) => std::process::ExitCode::SUCCESS,
        Ok(n) => {
            eprintln!(
                "\n{n} generated file(s) are out of date. Run `cargo run -p lk-agents-codegen -- --write` from `rust/` and commit the \
                 result."
            );
            std::process::ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("codegen: {e}");
            std::process::ExitCode::from(1)
        }
    }
}

fn run(check: bool) -> Result<usize, String> {
    let (root, python_packages) = repo_root()?;
    if check {
        placement::check(&root)?;
        architecture::host_members::check(&root)?;
    }
    sync(&root, &generate(&root, &python_packages)?, check)
}

/// Everything codegen writes, by path under the repo root.
struct Outputs {
    files: Vec<(PathBuf, String)>,
    /// Directories holding nothing but generated files: any other file there is obsolete.
    dirs: Vec<PathBuf>,
}

impl Outputs {
    fn add(&mut self, path: PathBuf, text: &str) {
        self.files.push((path, text.to_owned()));
    }
}

fn generate(root: &Path, python_packages: &[&str]) -> Result<Outputs, String> {
    let abi = decl::tree::Crate::read(root, "rust/crates/lk-agents-abi", "lk_agents_abi")?;
    let mut declared = decl::resolve::declarations(&abi)?;

    let mut o = Outputs { files: Vec::new(), dirs: Vec::new() };
    let python_backend = decl::tree::Crate::read(root, "rust/crates/lk-agents-py", "lk_agents_py")?;
    declared.local_interfaces = accessors::read(&python_backend)?;
    let implementations = implementations::emit(root, &declared)?;
    o.add(Path::new(adapter::DIR).join(&implementations.path), &implementations.text);
    let accessors = accessors::emit(&declared);
    o.add(Path::new(adapter::DIR).join(&accessors.path), &accessors.text);
    let conversions = records::conversions(&declared)?;
    o.add(Path::new(adapter::DIR).join(&conversions.path), &adapter::rust_names(&conversions.text));
    o.dirs.push(PathBuf::from(adapter::DIR));
    for out in adapter::emit(&declared)? {
        o.add(Path::new(adapter::DIR).join(&out.path), &adapter::rust_names(&out.text));
    }
    let (native_enums, enum_stub) = native_enums::emit(&declared)?;
    o.add(Path::new(adapter::DIR).join(&native_enums.path), &native_enums.text);
    let mut class_stubs = class_stub::emit(&declared)?;
    class_stubs.push(enum_stub);
    let mut models = records::python(&declared)?;
    models.extend(exceptions::python(&declared));
    models.extend(native_enums::flags(&declared));
    let bases = bases::read(root, &declared)?;
    let with_bases = python::emit(&declared, &bases)?;
    let (core_pyi, core_names) = surface::emit(class_stubs.iter())?;
    o.add(PathBuf::from(CORE_NAMES), &core_names.text);
    let mut forward = forward::emit(&declared)?;
    forward.extend(facade::emit(&declared)?);
    for package in python_packages {
        let python_root = Path::new(package).join(PYTHON_GENERATED);
        o.dirs.push(python_root.clone());
        // Authored plugin bases, constants and generated records share their declared module.
        let presentation: Vec<&python::Out> = with_bases.iter().chain(&forward).collect();
        for out in &presentation {
            let text = match models.iter().find(|m| m.path == out.path) {
                Some(m) => records::merge(&out.text, &m.text)
                    .map_err(|e| format!("_generated/{}: {e}", out.path))?,
                None => out.text.clone(),
            };
            o.add(python_root.join(&out.path), &text);
        }
        let merged = |m: &&python::Out| presentation.iter().any(|l| l.path == m.path);
        for out in class_stubs.iter().map(|s| &s.out).chain(models.iter().filter(|m| !merged(m))) {
            o.add(python_root.join(&out.path), &out.text);
        }
        o.add(Path::new(package).join(PYTHON_CORE).join(&core_pyi.path), &core_pyi.text);
        o.add(python_root.join("__init__.py"), INIT);
    }
    for (path, text) in &mut o.files {
        if path.extension().is_some_and(|ext| ext == "rs") {
            *text = rustfmt(root, text)?;
        }
    }
    Ok(o)
}

/// Writes the outputs under `root` (`--write`, which also deletes obsolete files), or counts the
/// files that differ, are missing or are obsolete without writing a byte (`--check`).
fn sync(root: &Path, outputs: &Outputs, check: bool) -> Result<usize, String> {
    let mut stale = 0;
    for (path, text) in &outputs.files {
        stale += put(&root.join(path), text, check)?;
    }
    for dir in &outputs.dirs {
        for entry in std::fs::read_dir(root.join(dir)).into_iter().flatten() {
            let path = entry.map_err(|e| e.to_string())?.path();
            // Interpreter caches live beside the generated modules.
            if path.is_dir() || outputs.files.iter().any(|(p, _)| root.join(p) == path) {
                continue;
            }
            if check {
                eprintln!("OBSOLETE  {}", path.display());
                stale += 1;
            } else {
                std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                println!("removed  {}", path.display());
            }
        }
    }
    Ok(stale)
}

/// The generated package's `__init__.py`, so the directory is never a namespace package.
const INIT: &str = "# GENERATED DIRECTORY -- every file here is written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`).\n\
#\n\
# Nothing re-exports from this package. The frozen public paths do the re-exporting\n\
# (`voice/events.py` does `from .._generated.events import ...`), because moving a\n\
# name here would move it in the public API too.\n";

/// Generated Rust is formatted as the workspace's `cargo fmt` would format it.
fn rustfmt(root: &Path, text: &str) -> Result<String, String> {
    use std::io::Write;
    let mut child = std::process::Command::new("rustfmt")
        .args(["--edition", "2024", "--emit", "stdout", "--config-path"])
        .arg(root.join("rust/rustfmt.toml"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("rustfmt: {e}"))?;
    child.stdin.take().expect("piped").write_all(text.as_bytes()).map_err(|e| e.to_string())?;
    let output = child.wait_with_output().map_err(|e| format!("rustfmt: {e}"))?;
    if !output.status.success() {
        return Err("rustfmt rejected generated Rust".into());
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}

/// Returns 1 when the file on disk differs from what would be written.
fn put(path: &Path, text: &str, check: bool) -> Result<usize, String> {
    let current = std::fs::read_to_string(path).ok();
    if current.as_deref() == Some(text) {
        return Ok(0);
    }
    if check {
        eprintln!("STALE  {}", path.display());
        return Ok(1);
    }
    std::fs::create_dir_all(path.parent().expect("has a parent")).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("wrote  {}", path.display());
    Ok(0)
}

/// From `CARGO_MANIFEST_DIR`, not the cwd, so output lands in the same place wherever it runs.
/// Returns the repo root and the Python projects under it (of [`PYTHON_PACKAGES`]).
fn repo_root() -> Result<(PathBuf, Vec<&'static str>), String> {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let packages: Vec<&'static str> = PYTHON_PACKAGES
            .iter()
            .copied()
            .filter(|d| p.join(d).join("pyproject.toml").exists())
            .collect();
        if p.join("rust/Cargo.toml").exists() && !packages.is_empty() {
            return Ok((p, packages));
        }
        if !p.pop() {
            return Err(format!(
                "could not find the repo root (no directory with rust/Cargo.toml and one of {PYTHON_PACKAGES:?} above the manifest)"
            ));
        }
    }
}

#[cfg(test)]
mod drift {
    use super::*;

    fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
        let mut out = Vec::new();
        let mut todo = vec![dir.to_owned()];
        while let Some(d) = todo.pop() {
            for entry in std::fs::read_dir(&d).expect("readable") {
                let path = entry.expect("entry").path();
                if path.is_dir() {
                    todo.push(path);
                } else {
                    out.push((path.clone(), std::fs::read(&path).expect("readable")));
                }
            }
        }
        out.sort();
        out
    }

    // `--check` fails on a changed, missing or obsolete file of every output family and writes
    // nothing while doing so.
    #[test]
    fn generated_surface_drift() {
        let (root, packages) = repo_root().expect("repo root");
        let outputs = generate(&root, &packages).expect("generates");
        assert!(
            outputs.files.iter().all(|(path, _)| !path.starts_with("livekit-agents"))
                && outputs.dirs.iter().all(|path| !path.starts_with("livekit-agents")),
            "generation must never restore the retired v1 package"
        );
        let tmp = std::env::temp_dir().join(format!("codegen-drift-{}", uuid::Uuid::new_v4()));
        for dir in &outputs.dirs {
            std::fs::create_dir_all(tmp.join(dir)).expect("dir");
        }
        sync(&tmp, &outputs, false).expect("writes");
        assert_eq!(sync(&tmp, &outputs, true), Ok(0));

        let families = [
            "rust/crates/lk-agents-py/src/generated/mod.rs",
            "livekit-agents-next/livekit/agents/_generated/storage_class.py",
            "livekit-agents-next/livekit/agents/_generated/stt.py",
            "livekit-agents-next/livekit/agents/_generated/language.py",
            "livekit-agents-next/livekit/agents/_generated/__init__.py",
            "livekit-agents-next/livekit/agents/_generated/_core_voice.pyi",
            "livekit-agents-next/livekit/agents/_generated/_core_llm.pyi",
            "livekit-agents-next/livekit/agents/_core.pyi",
            CORE_NAMES,
        ];
        for family in families {
            let path = tmp.join(family);
            let text = std::fs::read(&path).unwrap_or_else(|_| panic!("{family} is generated"));
            let mut changed = text.clone();
            changed.extend_from_slice(b"\n# drift\n");
            std::fs::write(&path, &changed).expect("mutates");
            let before = snapshot(&tmp);
            assert_eq!(sync(&tmp, &outputs, true), Ok(1), "{family} changed");
            assert_eq!(snapshot(&tmp), before, "--check wrote while {family} was changed");
            std::fs::remove_file(&path).expect("removes");
            assert_eq!(sync(&tmp, &outputs, true), Ok(1), "{family} missing");
            assert!(!path.exists(), "--check recreated {family}");
            std::fs::write(&path, &text).expect("restores");
        }
        for dir in &outputs.dirs {
            let stray = tmp.join(dir).join("obsolete_output.py");
            std::fs::write(&stray, "# GENERATED FILE -- DO NOT EDIT.\n").expect("adds");
            assert_eq!(sync(&tmp, &outputs, true), Ok(1), "{} obsolete", dir.display());
            assert!(stray.exists(), "--check deleted an obsolete file");
            std::fs::remove_file(&stray).expect("removes");
        }
        std::fs::remove_dir_all(&tmp).expect("cleans up");
    }
}
