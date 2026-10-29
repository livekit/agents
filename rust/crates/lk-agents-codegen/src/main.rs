//! `cargo run -p lk-agents-codegen -- --write` (in `rust/`): read every `#[derive(Host)]` declaration, write every host's view.
//! `--write` regenerates in place; `--check` exits 1 if any file would change (CI).
//!
//! Two readers during the migration to the declaration model: [`decl`] discovers the
//! `#[host(class)]` declarations of `lk-agents-abi` and writes their pyo3 adapters
//! ([`adapter`]) and stubs ([`class_stub`]); the families not yet moved are read from the explicit
//! source lists below.

mod adapter;
mod architecture;
mod bases;
mod class_stub;
mod decl;
mod forward;
mod imports;
mod ir;
mod parse;
mod placement;
mod proxy;
mod python;
mod records;
mod shape_error;
mod stub;
mod surface;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

/// Codegen inputs in output order: ABI records and the storage/A2A boundary records with Python
/// targets.
const SOURCES: &[&str] = &[
    "rust/crates/lk-agents-abi/src/a2a/metadata.rs",
    "rust/crates/lk-agents-abi/src/telemetry/attributes.rs",
    "rust/crates/lk-agents-abi/src/voice/events.rs",
    "rust/crates/lk-agents-abi/src/metrics/base.rs",
    "rust/crates/lk-agents-abi/src/metrics/usage/records.rs",
    "rust/crates/lk-agents-abi/src/voice/amd.rs",
    "rust/crates/lk-agents-abi/src/voice/session_options.rs",
    "rust/crates/lk-agents-abi/src/voice/session_effects.rs",
    "rust/crates/lk-agents-abi/src/worker/worker.rs",
    "rust/crates/lk-agents-abi/src/stt/stt.rs",
    "rust/crates/lk-agents-abi/src/llm/chat_context.rs",
    "rust/crates/lk-agents-abi/src/llm/realtime.rs",
    // The four pipeline events, in the order they shared one file.
    "rust/crates/lk-agents-abi/src/stt/events.rs",
    "rust/crates/lk-agents-abi/src/vad/events.rs",
    "rust/crates/lk-agents-abi/src/turn/events.rs",
    "rust/crates/lk-agents-abi/src/llm/events.rs",
    "rust/crates/lk-agents-abi/src/llm/llm.rs",
    "rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs",
    "rust/crates/lk-agents-abi/src/tts/tts.rs",
    "rust/crates/lk-agents-abi/src/llm/mcp.rs",
    "rust/crates/lk-agents-abi/src/voice/room_io.rs",
    "rust/crates/lk-agents-abi/src/llm/provider_format/image.rs",
    "rust/crates/lk-agents-abi/src/turn/keyterms.rs",
    "rust/crates/lk-agents-abi/src/llm/tool_executor.rs",
    "rust/crates/lk-agents-abi/src/turn/eot.rs",
    "rust/crates/lk-agents-abi/src/worker/job.rs",
    "rust/crates/lk-agents-abi/src/types/connect.rs",
    "rust/crates/lk-agents-abi/src/utils/audio.rs",
    "rust/crates/lk-agents-abi/src/tts/timed.rs",
    "rust/crates/lk-agents-abi/src/types/identity.rs",
    "rust/crates/lk-agents-abi/src/voice/rtc.rs",
    "rust/crates/lk-agents-abi/src/voice/calls.rs",
    "rust/crates/lk-agents-abi/src/worker/remote_session.rs",
    "rust/crates/lk-agents-abi/src/llm/fallback.rs",
    "rust/crates/lk-agents-abi/src/inference/fallback.rs",
    "rust/crates/lk-agents-abi/src/tts/markup/steering.rs",
    "rust/crates/lk-agents-abi/src/tts/markup/expr.rs",
    "rust/crates/lk-agents-abi/src/voice/workflows.rs",
    "rust/crates/lk-agents-abi/src/vad/vad.rs",
    "rust/crates/lk-agents-abi/src/inference/mod.rs",
    // The session schema (plan 13.5): the row models Python reads rows as, then everything else
    // that crosses `_core`'s storage boundary: the statements' requests and results, the run
    // records, the turn's context and output, the threads' and the streams' records.
    "rust/crates/lk-agents-abi/src/a2a/options.rs",
    "rust/crates/lk-agents-storage/src/schema.rs",
    "rust/crates/lk-agents-storage/src/db.rs",
    "rust/crates/lk-agents-storage/src/streams.rs",
    "rust/crates/lk-agents-storage/src/run.rs",
    "rust/crates/lk-agents-py/src/storage.rs",
];

/// `#[host(export ..)]` / `#[host(object ..)]` files. Any core crate may appear.
const EXPORT_SOURCES: &[&str] = &[
    "rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs",
    "rust/crates/lk-agents-voice/src/workflows/mod.rs",
    "rust/crates/lk-agents-abi/src/voice/background_audio.rs",
    "rust/crates/lk-agents-voice/src/tool_executor.rs",
    "rust/crates/lk-agents-abi/src/worker/endpoint/reset_codes.rs",
    "rust/crates/lk-agents-voice/src/keyterm_detection.rs",
    "rust/crates/lk-agents-voice/src/amd/classifier.rs",
    "rust/crates/lk-agents-voice/src/agent_activity/options.rs",
    "rust/crates/lk-agents-voice/src/speech_handle.rs",
    "rust/crates/lk-agents-voice/src/workflows/dtmf.rs",
    "rust/crates/lk-agents-abi/src/voice/workflows/warm_transfer.rs",
    "rust/crates/lk-agents-voice/src/workflows/prompts.rs",
    "rust/crates/lk-agents-abi/src/voice/workflows.rs",
    // One subject per file, reached through `agent_session`'s re-exports, in the order they were
    // declared in when they were one file.
    "rust/crates/lk-agents-voice/src/agent_session/mod.rs",
    "rust/crates/lk-agents-abi/src/utils/clock.rs",
    "rust/crates/lk-agents-voice/src/agent_session/api.rs",
    "rust/crates/lk-agents-voice/src/agent_session/api/host.rs",
    "rust/crates/lk-agents-abi/src/metrics/usage.rs",
    "rust/crates/lk-agents-abi/src/tts/stream_pacer.rs",
    "rust/crates/lk-agents-abi/src/worker/job.rs",
    "rust/crates/lk-agents-voice/src/remote_session.rs",
    "rust/crates/lk-agents-abi/src/tts/markup/mood.rs",
    "rust/crates/lk-agents-abi/src/llm/chat_context.rs",
    "rust/crates/lk-agents-voice/src/generation/mod.rs",
    "rust/crates/lk-agents-abi/src/voice/io.rs",
    "rust/crates/lk-agents-voice/src/io.rs",
    "rust/crates/lk-agents-abi/src/vad/vad.rs",
    "rust/crates/lk-agents-abi/src/voice/rtc.rs",
    "rust/crates/lk-agents-abi/src/voice/rtc/events.rs",
    "rust/crates/lk-agents-abi/src/voice/rtc/ports.rs",
    "rust/crates/lk-agents-abi/src/voice/room_owner.rs",
    "rust/crates/lk-agents-abi/src/tts/tts.rs",
    "rust/crates/lk-agents-abi/src/stt/stt.rs",
    "rust/crates/lk-agents-abi/src/llm/llm.rs",
    "rust/crates/lk-agents-abi/src/turn/detector.rs",
    "rust/crates/lk-agents-abi/src/types/retry.rs",
    "rust/crates/lk-agents-abi/src/llm/realtime.rs",
    "rust/crates/lk-agents-abi/src/tokenize/language/code.rs",
    "rust/crates/lk-agents-abi/src/media/images.rs",
    "rust/crates/lk-agents-voice/src/transcription/filters.rs",
    "rust/crates/lk-agents-voice/src/transcription/synchronizer.rs",
    "rust/crates/lk-agents-abi/src/llm/events.rs",
    "rust/crates/lk-agents-voice/src/agent_session/commands.rs",
    "rust/crates/lk-agents-abi/src/utils/host_ref.rs",
    "rust/crates/lk-agents-abi/src/llm/mcp.rs",
];

/// The authoritative v2 Python project. Generation must not recreate the retired v1 package.
const PYTHON_PACKAGES: &[&str] = &["livekit-agents-next"];
const PYTHON_GENERATED: &str = "livekit/agents/_generated";
/// Where `_core.pyi` sits, beside the extension.
const PYTHON_CORE: &str = "livekit/agents";
/// The names `dir(_core)` returns, which the registration test of `lk-agents-py` reads.
const CORE_NAMES: &str = "rust/crates/lk-agents-py/src/core_names.json";

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
    let mut decls = Vec::new();
    let mut constants = Vec::new();
    for src in SOURCES {
        let path = root.join(src);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("{src}: {e} (the SOURCES list names a file that is not there)"))?;
        decls.extend(parse::file(src, &text)?);
        constants.extend(parse::constants(src, &text)?);
    }
    parse::link(&mut decls)?;
    let abi = decl::tree::Crate::read(root, "rust/crates/lk-agents-abi", "lk_agents_abi")?;
    let mut declared = decl::resolve::declarations(&abi)?;
    records::link_legacy(&mut decls, &mut declared)?;
    resolve(&decls)?;
    // Aliases first, from every file: a signature may name `Object` or `Extra` from another file.
    let mut texts = Vec::new();
    for src in EXPORT_SOURCES {
        let text = std::fs::read_to_string(root.join(src))
            .map_err(|e| format!("{src}: {e} (EXPORT_SOURCES names a file that is not there)"))?;
        texts.push((*src, text));
    }
    let mut all_aliases = Vec::new();
    for src in SOURCES {
        all_aliases
            .extend(parse::aliases(&std::fs::read_to_string(root.join(src)).unwrap_or_default()));
    }
    for (_, text) in &texts {
        all_aliases.extend(parse::aliases(text));
    }
    let global = parse::global_aliases(&all_aliases);
    let mut exposed = ir::Exposed::default();
    for (src, text) in &texts {
        let x = parse::exposed_with(src, text, &global)?;
        if x.is_empty() {
            return Err(format!("{src}: in EXPORT_SOURCES but holds no #[host(..)] item"));
        }
        exposed.extend(x);
    }
    unlisted_exports(root)?;

    let mut o = Outputs { files: Vec::new(), dirs: Vec::new() };
    records::check_in_place(root, &declared)?;
    let conversions = records::conversions(&declared)?;
    o.add(Path::new(adapter::DIR).join(&conversions.path), &conversions.text);
    for (dir, outs) in
        [(adapter::DIR, adapter::emit(&declared)?), (records::DIR, records::rust(&declared)?)]
    {
        o.dirs.push(PathBuf::from(dir));
        for out in &outs {
            o.add(Path::new(dir).join(&out.path), &out.text);
        }
    }
    let class_stubs = class_stub::emit(&declared)?;
    let models = records::python(&declared)?;
    let bases = bases::emit(&decls, &exposed, &declared)?;
    let with_bases = python::emit_with(&decls, &bases, &constants)?;
    let stub = stub::emit(&decls, &exposed)?;
    let (core_pyi, core_names) = surface::emit(class_stubs.iter().chain(&stub))?;
    o.add(PathBuf::from(CORE_NAMES), &core_names.text);
    let forward = forward::emit(&decls, &exposed, &declared)?;
    for package in python_packages {
        let python_root = Path::new(package).join(PYTHON_GENERATED);
        o.dirs.push(python_root.clone());
        // A module holding both legacy and declared models is one file: a legacy model there may
        // name a declared one.
        let legacy: Vec<&python::Out> =
            with_bases.iter().chain(stub.iter().map(|s| &s.out)).chain(&forward).collect();
        for out in &legacy {
            let text = match models.iter().find(|m| m.path == out.path) {
                Some(m) => records::merge(&out.text, &m.text)
                    .map_err(|e| format!("_generated/{}: {e}", out.path))?,
                None => out.text.clone(),
            };
            o.add(python_root.join(&out.path), &text);
        }
        let merged = |m: &&python::Out| legacy.iter().any(|l| l.path == m.path);
        for out in class_stubs.iter().map(|s| &s.out).chain(models.iter().filter(|m| !merged(m))) {
            o.add(python_root.join(&out.path), &out.text);
        }
        o.add(Path::new(package).join(PYTHON_CORE).join(&core_pyi.path), &core_pyi.text);
        o.add(python_root.join("__init__.py"), INIT);
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

/// Every `Ty::Declared` names a type in the set. Checked after all files (forward references are
/// legal), and an error rather than `Any`.
fn resolve(decls: &[ir::Decl]) -> Result<(), String> {
    for d in decls {
        // Only rendered types must be declared: binding-only fields cross by their own converters.
        if d.target(ir::Host::Python).is_none() {
            continue;
        }
        // A `#[python(skip)]` field is on no host class, so the type it names never renders
        // (`TurnContext.boundary` is a core object the binding hands over beside the record).
        for f in d.fields().iter().filter(|f| !f.py_skip) {
            for name in declared(&f.ty) {
                if !decls.iter().any(|o| o.rust_name == name) {
                    return Err(format!(
                        "{}: {}.{} names `{name}`, which is not a #[derive(Host)] type. Declare \
                         it, or write it as #[python(ty = \"..\", from = \"..\")] if it is a type \
                         we do not own.",
                        d.source, d.rust_name, f.rust_name
                    ));
                }
            }
        }
        if let ir::Body::Union(members) = &d.body {
            for m in members {
                for name in declared(&m.ty) {
                    if !decls.iter().any(|o| o.rust_name == name) {
                        return Err(format!(
                            "{}: union {} names `{name}`, which is not a #[derive(Host)] type.",
                            d.source, d.rust_name
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn declared(t: &ir::Ty) -> Vec<String> {
    match t {
        ir::Ty::Declared(n) => vec![n.clone()],
        ir::Ty::Option(i) | ir::Ty::List(i) => declared(i),
        ir::Ty::Map(k, v) => [declared(k), declared(v)].concat(),
        ir::Ty::Tuple(items) => items.iter().flat_map(declared).collect(),
        _ => Vec::new(),
    }
}

/// Refuses an in-place export (`#[host(export)]`, `object`, `stream`, ..) in a file outside
/// [`EXPORT_SOURCES`]: its macro registers the name in `_core`, but no stub or `core_names.json`
/// entry would declare it. The ABI's `python` bridge (the macros' own home and fixtures) is exempt.
fn unlisted_exports(root: &Path) -> Result<(), String> {
    let output = std::process::Command::new("git")
        .args(["ls-files", "-c", "-o", "--exclude-standard", "-z", "--", "rust/crates/**/*.rs"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("git ls-files: {e}"))?;
    let paths = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    let mut unlisted = Vec::new();
    for path in paths.split('\0').filter(|p| !p.is_empty() && !EXPORT_SOURCES.contains(p)) {
        if path.starts_with("rust/crates/lk-agents-abi/src/python/")
            || ["codegen", "bindgen", "host-derive"]
                .iter()
                .any(|c| path.starts_with(&format!("rust/crates/lk-agents-{c}/")))
        {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(root.join(path)) else { continue };
        let exports = text.lines().map(str::trim_start).any(|l| {
            l.starts_with("#[")
                && ["host(export", "host(object("].iter().any(|h| l.contains(h))
        });
        if exports {
            unlisted.push(path);
        }
    }
    match unlisted.is_empty() {
        true => Ok(()),
        false => Err(format!(
            "{} hold in-place exports but are not in EXPORT_SOURCES (their `_core` names would \
             have no stub)",
            unlisted.join(", ")
        )),
    }
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
            "rust/crates/lk-agents-abi/src/python/generated/mod.rs",
            "livekit-agents-next/livekit/agents/_generated/storage_class.py",
            "livekit-agents-next/livekit/agents/_generated/stt.py",
            "livekit-agents-next/livekit/agents/_generated/language.py",
            "livekit-agents-next/livekit/agents/_generated/__init__.py",
            "livekit-agents-next/livekit/agents/_generated/_core_classes.pyi",
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
