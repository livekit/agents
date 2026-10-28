//! `cargo run -p lk-agents-codegen -- --write` (in `rust/`): read every `#[derive(Host)]` declaration, write every host's view.
//! `--write` regenerates in place; `--check` exits 1 if any file would change (CI).
//!
//! Two readers during the migration to the declaration model: [`decl`] discovers the
//! `#[host(class)]` declarations of `lk-agents-abi` and writes their pyo3 adapters
//! ([`adapter`]) and stubs ([`class_stub`]); the families not yet moved are read from the explicit
//! source lists below.

mod adapter;
mod bases;
mod class_stub;
mod decl;
mod forward;
mod ir;
mod parse;
mod placement;
mod proxy;
mod python;
mod stub;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

/// Codegen inputs in output order: ABI records and the storage/A2A boundary records with Python
/// targets.
const SOURCES: &[&str] = &[
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
    "rust/crates/lk-agents-abi/src/telemetry/report.rs",
    "rust/crates/lk-agents-abi/src/voice/room_io.rs",
    "rust/crates/lk-agents-abi/src/llm/provider_format/image.rs",
    "rust/crates/lk-agents-abi/src/turn/keyterms.rs",
    "rust/crates/lk-agents-abi/src/llm/tool_executor.rs",
    "rust/crates/lk-agents-abi/src/voice/run_result.rs",
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
    "rust/crates/lk-agents-storage/src/schema.rs",
    "rust/crates/lk-agents-storage/src/db.rs",
    "rust/crates/lk-agents-storage/src/streams.rs",
    "rust/crates/lk-agents-storage/src/run.rs",
    "rust/crates/lk-agents-a2a/src/lib.rs",
    "rust/crates/lk-agents-py/src/storage.rs",
];

/// `#[host(export ..)]` / `#[host(object ..)]` files. Any core crate may appear.
const EXPORT_SOURCES: &[&str] = &[
    "rust/crates/lk-agents-abi/src/utils/strings.rs",
    "rust/crates/lk-agents-abi/src/tts/markup/strip.rs",
    "rust/crates/lk-agents-abi/src/tts/markup/transcript.rs",
    "rust/crates/lk-agents-abi/src/tts/markup/summarize.rs",
    "rust/crates/lk-agents-abi/src/llm/schema/openai.rs",
    "rust/crates/lk-agents-abi/src/llm/provider_format/strict.rs",
    "rust/crates/lk-agents-abi/src/telemetry/report.rs",
    "rust/crates/lk-agents-voice/src/workflows/mod.rs",
    "rust/crates/lk-agents-voice/src/background_audio.rs",
    "rust/crates/lk-agents-voice/src/tool_executor.rs",
    "rust/crates/lk-agents-worker/src/auth.rs",
    "rust/crates/lk-agents-worker/src/config.rs",
    "rust/crates/lk-agents-worker/src/cli.rs",
    "rust/crates/lk-agents-inference/src/amd/detector.rs",
    "rust/crates/lk-agents-voice/src/keyterm_detection.rs",
    "rust/crates/lk-agents-voice/src/amd/classifier.rs",
    "rust/crates/lk-agents-voice/src/agent_activity/options.rs",
    "rust/crates/lk-agents-voice/src/speech_handle.rs",
    "rust/crates/lk-agents-voice/src/workflows/dtmf.rs",
    "rust/crates/lk-agents-voice/src/workflows/warm_transfer.rs",
    "rust/crates/lk-agents-voice/src/workflows/prompts.rs",
    "rust/crates/lk-agents-abi/src/voice/workflows.rs",
    "rust/crates/lk-agents-inference/src/llm/models.rs",
    // One subject per file, reached through `agent_session`'s re-exports, in the order they were
    // declared in when they were one file.
    "rust/crates/lk-agents-voice/src/agent_session/mod.rs",
    "rust/crates/lk-agents-abi/src/utils/clock.rs",
    "rust/crates/lk-agents-voice/src/agent_session/api.rs",
    "rust/crates/lk-agents-voice/src/agent_session/api/host.rs",
    "rust/crates/lk-agents-abi/src/telemetry/gen_ai.rs",
    "rust/crates/lk-agents-abi/src/evals/llm_judge.rs",
    "rust/crates/lk-agents-abi/src/evals/evaluator.rs",
    "rust/crates/lk-agents-abi/src/metrics/usage.rs",
    "rust/crates/lk-agents-abi/src/utils/filters.rs",
    "rust/crates/lk-agents-abi/src/utils/byte_stream.rs",
    "rust/crates/lk-agents-abi/src/tts/stream_pacer.rs",
    "rust/crates/lk-agents-abi/src/tokenize/xml.rs",
    "rust/crates/lk-agents-abi/src/tokenize/replace.rs",
    "rust/crates/lk-agents-media/src/stream_buffer.rs",
    "rust/crates/lk-agents-media/src/decoder.rs",
    "rust/crates/lk-agents-inference/src/eot/options.rs",
    "rust/crates/lk-agents-abi/src/worker/job.rs",
    "rust/crates/lk-agents-voice/src/remote_session.rs",
    "rust/crates/lk-agents-voice/src/remote_session/driver.rs",
    "rust/crates/lk-agents-voice/src/remote_session/console.rs",
    "rust/crates/lk-agents-abi/src/errors/api.rs",
    "rust/crates/lk-agents-abi/src/tts/markup/mod.rs",
    "rust/crates/lk-agents-abi/src/tts/markup/mood.rs",
    "rust/crates/lk-agents-abi/src/llm/chat_context.rs",
    "rust/crates/lk-agents-abi/src/llm/ctx_diff.rs",
    "rust/crates/lk-agents-abi/src/metrics/otel.rs",
    "rust/crates/lk-agents-abi/src/types/connect.rs",
    "rust/crates/lk-agents-voice/src/generation/mod.rs",
    "rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs",
    "rust/crates/lk-agents-abi/src/llm/provider_format/host.rs",
    "rust/crates/lk-agents-abi/src/tokenize/thinking.rs",
    "rust/crates/lk-agents-abi/src/tokenize/search.rs",
    "rust/crates/lk-agents-abi/src/voice/io.rs",
    "rust/crates/lk-agents-voice/src/io.rs",
    "rust/crates/lk-agents-voice/src/avatar.rs",
    "rust/crates/lk-agents-voice/src/avatar_runner.rs",
    "rust/crates/lk-agents-abi/src/vad/vad.rs",
    "rust/crates/lk-agents-rtc/src/traits.rs",
    "rust/crates/lk-agents-rtc/src/events.rs",
    "rust/crates/lk-agents-rtc/src/playout.rs",
    "rust/crates/lk-agents-rtc/src/wait.rs",
    "rust/crates/lk-agents-abi/src/voice/rtc.rs",
    "rust/crates/lk-agents-voice/src/room_io/rooms.rs",
    "rust/crates/lk-agents-abi/src/tts/tts.rs",
    "rust/crates/lk-agents-abi/src/stt/stt.rs",
    "rust/crates/lk-agents-abi/src/llm/llm.rs",
    "rust/crates/lk-agents-abi/src/turn/detector.rs",
    "rust/crates/lk-agents-abi/src/tts/stream_adapter.rs",
    "rust/crates/lk-agents-abi/src/stt/stream_adapter.rs",
    "rust/crates/lk-agents-abi/src/types/retry.rs",
    "rust/crates/lk-agents-abi/src/stt/retry.rs",
    "rust/crates/lk-agents-abi/src/telemetry/provider.rs",
    "rust/crates/lk-agents-inference/src/fallback/chain.rs",
    "rust/crates/lk-agents-inference/src/fallback/llm.rs",
    "rust/crates/lk-agents-inference/src/fallback/stt.rs",
    "rust/crates/lk-agents-inference/src/fallback/tts.rs",
    "rust/crates/lk-agents-inference/src/fallback/realtime.rs",
    "rust/crates/lk-agents-inference/src/stt/mod.rs",
    "rust/crates/lk-agents-inference/src/vad/mod.rs",
    "rust/crates/lk-agents-inference/src/tts/mod.rs",
    "rust/crates/lk-agents-abi/src/llm/realtime.rs",
    "rust/crates/lk-agents-inference/src/realtime/session.rs",
    "rust/crates/lk-agents-inference/src/realtime/dialect.rs",
    "rust/crates/lk-agents-inference/src/stt/options.rs",
    "rust/crates/lk-agents-inference/src/tts/wire.rs",
    "rust/crates/lk-agents-inference/src/interruption/event.rs",
    "rust/crates/lk-agents-inference/src/interruption/options.rs",
    "rust/crates/lk-agents-abi/src/tokenize/language/code.rs",
    "rust/crates/lk-agents-abi/src/utils/connection_pool.rs",
    "rust/crates/lk-agents-voice/src/room_io/_input.rs",
    "rust/crates/lk-agents-media/src/images/encode.rs",
    "rust/crates/lk-agents-abi/src/tokenize/word.rs",
    "rust/crates/lk-agents-abi/src/tokenize/sentence.rs",
    "rust/crates/lk-agents-abi/src/evals/group.rs",
    "rust/crates/lk-agents-worker/src/load.rs",
    "rust/crates/lk-agents-worker/src/hw/cpu.rs",
    "rust/crates/lk-agents-worker/src/hw/cgroup.rs",
    "rust/crates/lk-agents-abi/src/llm/tool_arguments.rs",
    "rust/crates/lk-agents-media/src/blingfire.rs",
    "rust/crates/lk-agents-voice/src/transcription/filters.rs",
    "rust/crates/lk-agents-voice/src/transcription/synchronizer.rs",
    "rust/crates/lk-agents-abi/src/llm/schema/defaults.rs",
    "rust/crates/lk-agents-media/src/images/serialize.rs",
    "rust/crates/lk-agents-inference/src/mcp/mod.rs",
    "rust/crates/lk-agents-inference/src/mcp/render.rs",
    "rust/crates/lk-agents-inference/src/mcp/session.rs",
];

/// The generated Python package, under every Python project the repo has: `python/livekit-agents`
/// (agents-rust), or `livekit-agents` and its successor `livekit-agents-next` (livekit/agents, where
/// `rust/` sits beside the packages).
const PYTHON_PACKAGES: &[&str] =
    &["python/livekit-agents", "livekit-agents", "livekit-agents-next"];
const PYTHON_GENERATED: &str = "livekit/agents/_generated";
/// The package whose plugin base classes stay hand-written: the reference the copy is held to.
const REFERENCE_PACKAGE: &str = "livekit-agents";

fn main() -> std::process::ExitCode {
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

    let mut stale = 0;
    let abi = decl::tree::Crate::read(&root, "rust/crates/lk-agents-abi", "lk_agents_abi")?;
    let declared = decl::resolve::declarations(&abi)?;
    stale += put_dir(&root.join(adapter::DIR), &adapter::emit(&declared)?, check)?;
    let class_stubs = class_stub::emit(&declared)?;
    let bases = bases::emit(&decls, &exposed)?;
    let with_bases = python::emit_with(&decls, &bases, &constants)?;
    let reference: Vec<ir::Decl> = decls
        .iter()
        .filter(|d| !d.target(ir::Host::Python).is_some_and(|t| t.with_bases))
        .cloned()
        .collect();
    let records = python::emit_with(&reference, &[], &constants)?;
    let stub = stub::emit(&decls, &exposed)?;
    let forward = forward::emit(&decls, &exposed)?;
    for package in python_packages {
        let python_root = root.join(package).join(PYTHON_GENERATED);
        // The reference keeps its hand-written base classes.
        let python = if package == REFERENCE_PACKAGE { &records } else { &with_bases };
        for out in python.iter().chain(&stub) {
            stale += put(&python_root.join(&out.path), &out.text, check)?;
        }
        // Every package: the records name these modules (`.language`), the reference's included.
        for out in &forward {
            stale += put(&python_root.join(&out.path), &out.text, check)?;
        }
        // The declared classes are the successor's: the reference keeps its own `_core`.
        if package != REFERENCE_PACKAGE {
            for out in &class_stubs {
                stale += put(&python_root.join(&out.path), &out.text, check)?;
            }
        }
        stale += put(&python_root.join("__init__.py"), INIT, check)?;
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

/// `outs` are the whole of `dir`: any other file there is obsolete, reported by `--check` and
/// deleted by `--write`.
fn put_dir(dir: &Path, outs: &[python::Out], check: bool) -> Result<usize, String> {
    let mut stale = 0;
    for out in outs {
        stale += put(&dir.join(&out.path), &out.text, check)?;
    }
    for entry in std::fs::read_dir(dir).into_iter().flatten() {
        let path = entry.map_err(|e| e.to_string())?.path();
        let known = outs.iter().any(|o| dir.join(&o.path) == path);
        if known {
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
    Ok(stale)
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
