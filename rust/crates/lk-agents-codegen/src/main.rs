//! `cargo run -p lk-agents-codegen -- --write` (in `rust/`): read every `#[derive(Host)]` declaration, write every host's view.
//! `--write` regenerates in place; `--check` exits 1 if any file would change (CI).
//! The source lists are explicit, not a glob, so generated output never depends on file placement.

mod ir;
mod napi;
mod node;
mod parse;
mod python;
mod stub;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_napi;

use std::path::{Path, PathBuf};

/// Codegen inputs in output order, each with the path the napi binding reaches its records by.
/// All in `lk-agents-abi` (`scripts/check_layout.py` enforces it), plus `lk-agents-storage`'s
/// schema, whose records have a Python target only.
const SOURCES: &[(&str, &str)] = &[
    ("rust/crates/lk-agents-abi/src/voice/events.rs", "lk_agents_abi::voice::events"),
    ("rust/crates/lk-agents-abi/src/metrics/base.rs", "lk_agents_abi::metrics::base"),
    (
        "rust/crates/lk-agents-abi/src/metrics/usage/records.rs",
        "lk_agents_abi::metrics::usage::records",
    ),
    (
        "rust/crates/lk-agents-abi/src/metrics/usage_collector/records.rs",
        "lk_agents_abi::metrics::usage_collector::records",
    ),
    ("rust/crates/lk-agents-abi/src/voice/amd.rs", "lk_agents_abi::voice::amd"),
    (
        "rust/crates/lk-agents-abi/src/voice/session_options.rs",
        "lk_agents_abi::voice::session_options",
    ),
    (
        "rust/crates/lk-agents-abi/src/voice/session_effects.rs",
        "lk_agents_abi::voice::session_effects",
    ),
    ("rust/crates/lk-agents-abi/src/worker/worker.rs", "lk_agents_abi::worker::worker"),
    ("rust/crates/lk-agents-abi/src/stt/stt.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/llm/chat_context.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/llm/realtime.rs", "lk_agents_abi::llm::realtime"),
    // The four pipeline events, in the order they shared one file.
    ("rust/crates/lk-agents-abi/src/stt/events.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/vad/events.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/turn/events.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/llm/events.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/llm/llm.rs", "lk_agents_abi"),
    (
        "rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs",
        "lk_agents_abi::llm::provider_format::tools",
    ),
    ("rust/crates/lk-agents-abi/src/tts/tts.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/llm/mcp.rs", "lk_agents_abi::llm::mcp"),
    ("rust/crates/lk-agents-abi/src/telemetry/report.rs", "lk_agents_abi::telemetry::report"),
    ("rust/crates/lk-agents-abi/src/voice/room_io.rs", "lk_agents_abi::voice::room_io"),
    (
        "rust/crates/lk-agents-abi/src/llm/provider_format/image.rs",
        "lk_agents_abi::llm::provider_format",
    ),
    ("rust/crates/lk-agents-abi/src/turn/keyterms.rs", "lk_agents_abi::turn::keyterms"),
    ("rust/crates/lk-agents-abi/src/llm/tool_executor.rs", "lk_agents_abi::llm::tool_executor"),
    ("rust/crates/lk-agents-abi/src/voice/run_result.rs", "lk_agents_abi::voice::run_result"),
    ("rust/crates/lk-agents-abi/src/turn/eot.rs", "lk_agents_abi::turn::eot"),
    ("rust/crates/lk-agents-abi/src/worker/job.rs", "lk_agents_abi::worker::job"),
    ("rust/crates/lk-agents-abi/src/types/connect.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/utils/audio.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/tts/timed.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/types/identity.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/voice/rtc.rs", "lk_agents_abi::voice::rtc"),
    ("rust/crates/lk-agents-abi/src/voice/calls.rs", "lk_agents_abi::voice::calls"),
    (
        "rust/crates/lk-agents-abi/src/worker/remote_session.rs",
        "lk_agents_abi::worker::remote_session",
    ),
    ("rust/crates/lk-agents-abi/src/llm/fallback.rs", "lk_agents_abi::llm::fallback"),
    ("rust/crates/lk-agents-abi/src/inference/fallback.rs", "lk_agents_abi::inference::fallback"),
    ("rust/crates/lk-agents-abi/src/tts/markup/steering.rs", "lk_agents_abi::tts::markup"),
    ("rust/crates/lk-agents-abi/src/tts/markup/expr.rs", "lk_agents_abi::tts::markup"),
    ("rust/crates/lk-agents-abi/src/voice/workflows.rs", "lk_agents_abi::voice::workflows"),
    ("rust/crates/lk-agents-abi/src/voice/audio.rs", "lk_agents_abi::voice::audio"),
    ("rust/crates/lk-agents-abi/src/vad/vad.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/inference/mod.rs", "lk_agents_abi::inference"),
    // The session schema (plan 13.5): the row models Python reads rows as, then everything else
    // that crosses `_core`'s storage boundary: the statements' requests and results, the run
    // records, the turn's context and output, the threads' and the streams' records.
    ("rust/crates/lk-agents-storage/src/schema.rs", "lk_agents_storage::schema"),
    ("rust/crates/lk-agents-storage/src/statements.rs", "lk_agents_storage::statements"),
    ("rust/crates/lk-agents-storage/src/streams.rs", "lk_agents_storage::streams"),
    ("rust/crates/lk-agents-storage/src/run/records.rs", "lk_agents_storage::run"),
    ("rust/crates/lk-agents-storage/src/run.rs", "lk_agents_storage::run"),
    ("rust/crates/lk-agents-storage/src/turn/message.rs", "lk_agents_storage::turn"),
    ("rust/crates/lk-agents-storage/src/turn.rs", "lk_agents_storage::turn"),
    ("rust/crates/lk-agents-storage/src/turn/host.rs", "lk_agents_storage::turn"),
    ("rust/crates/lk-agents-storage/src/threads.rs", "lk_agents_storage::threads"),
];

/// `#[host(export ..)]` / `#[host(object ..)]` files, with the path the binding reaches their
/// items by (a re-exporting root when the module is private). Any core crate may appear.
const EXPORT_SOURCES: &[(&str, &str)] = &[
    ("rust/crates/lk-agents-abi/src/utils/strings.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/tts/markup/strip.rs", "lk_agents_abi::tts::markup"),
    ("rust/crates/lk-agents-abi/src/tts/markup/transcript.rs", "lk_agents_abi::tts::markup"),
    ("rust/crates/lk-agents-abi/src/tts/markup/summarize.rs", "lk_agents_abi::tts::markup"),
    ("rust/crates/lk-agents-abi/src/llm/schema/openai.rs", "lk_agents_abi::llm::schema"),
    (
        "rust/crates/lk-agents-abi/src/llm/provider_format/strict.rs",
        "lk_agents_abi::llm::provider_format",
    ),
    ("rust/crates/lk-agents-abi/src/telemetry/report.rs", "lk_agents_abi::telemetry::report"),
    (
        "rust/crates/lk-agents-voice/src/workflows/builtins/credit_card.rs",
        "lk_agents_voice::workflows::builtins::credit_card",
    ),
    ("rust/crates/lk-agents-voice/src/tool_executor.rs", "lk_agents_voice::tool_executor"),
    ("rust/crates/lk-agents-voice/src/ivr/ivr_activity.rs", "lk_agents_voice::ivr::ivr_activity"),
    ("rust/crates/lk-agents-worker/src/auth.rs", "lk_agents_worker"),
    ("rust/crates/lk-agents-worker/src/config.rs", "lk_agents_worker"),
    ("rust/crates/lk-agents-inference/src/amd/detector.rs", "lk_agents_inference::amd"),
    ("rust/crates/lk-agents-voice/src/keyterm_detection.rs", "lk_agents_voice::keyterm_detection"),
    ("rust/crates/lk-agents-voice/src/amd/classifier.rs", "lk_agents_voice::amd::classifier"),
    ("rust/crates/lk-agents-voice/src/turn.rs", "lk_agents_voice::turn"),
    (
        "rust/crates/lk-agents-voice/src/agent_activity/options.rs",
        "lk_agents_voice::agent_activity::options",
    ),
    ("rust/crates/lk-agents-voice/src/workflows/handle.rs", "lk_agents_voice::workflows"),
    ("rust/crates/lk-agents-voice/src/speech_handle.rs", "lk_agents_voice::speech_handle"),
    (
        "rust/crates/lk-agents-voice/src/workflows/builtins/mod.rs",
        "lk_agents_voice::workflows::builtins",
    ),
    (
        "rust/crates/lk-agents-voice/src/workflows/builtins/dtmf.rs",
        "lk_agents_voice::workflows::builtins::dtmf",
    ),
    (
        "rust/crates/lk-agents-voice/src/workflows/builtins/warm_transfer.rs",
        "lk_agents_voice::workflows::builtins::warm_transfer",
    ),
    ("rust/crates/lk-agents-abi/src/voice/workflows.rs", "lk_agents_abi::voice::workflows"),
    ("rust/crates/lk-agents-abi/src/voice/host_policy.rs", "lk_agents_abi::voice::host_policy"),
    ("rust/crates/lk-agents-inference/src/llm/models.rs", "lk_agents_inference::llm::models"),
    ("rust/crates/lk-agents-inference/src/llm/mod.rs", "lk_agents_inference::llm"),
    (
        "rust/crates/lk-agents-voice/src/workflows/builtins/prompts.rs",
        "lk_agents_voice::workflows::builtins::prompts",
    ),
    (
        "rust/crates/lk-agents-voice/src/workflows/builtins/task_group.rs",
        "lk_agents_voice::workflows::builtins::task_group",
    ),
    // One subject per file, reached through `agent_session`'s re-exports, in the order they were
    // declared in when they were one file.
    ("rust/crates/lk-agents-voice/src/agent_session/mod.rs", "lk_agents_voice::agent_session"),
    ("rust/crates/lk-agents-voice/src/agent_session/commands.rs", "lk_agents_voice::agent_session"),
    ("rust/crates/lk-agents-voice/src/agent_session/time.rs", "lk_agents_voice::agent_session"),
    ("rust/crates/lk-agents-voice/src/agent_session/api.rs", "lk_agents_voice::agent_session"),
    ("rust/crates/lk-agents-voice/src/agent_session/api/host.rs", "lk_agents_voice::agent_session"),
    ("rust/crates/lk-agents-abi/src/telemetry/gen_ai.rs", "lk_agents_abi::gen_ai"),
    ("rust/crates/lk-agents-abi/src/evals/llm_judge.rs", "lk_agents_abi::evals"),
    ("rust/crates/lk-agents-abi/src/evals/evaluator.rs", "lk_agents_abi::evals"),
    ("rust/crates/lk-agents-abi/src/metrics/usage.rs", "lk_agents_abi::metrics"),
    ("rust/crates/lk-agents-abi/src/metrics/usage_collector.rs", "lk_agents_abi::metrics"),
    ("rust/crates/lk-agents-abi/src/utils/filters.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/utils/byte_stream.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/tts/stream_pacer.rs", "lk_agents_abi::tts::stream_pacer"),
    ("rust/crates/lk-agents-abi/src/tokenize/xml.rs", "lk_agents_abi::tokenize::xml"),
    ("rust/crates/lk-agents-abi/src/tokenize/replace.rs", "lk_agents_abi::tokenize"),
    ("rust/crates/lk-agents-media/src/stream_buffer.rs", "lk_agents_media"),
    ("rust/crates/lk-agents-media/src/decoder.rs", "lk_agents_media"),
    ("rust/crates/lk-agents-inference/src/eot/options.rs", "lk_agents_inference::eot::options"),
    ("rust/crates/lk-agents-abi/src/worker/job.rs", "lk_agents_abi::worker::job"),
    ("rust/crates/lk-agents-voice/src/remote_session.rs", "lk_agents_voice::remote_session"),
    (
        "rust/crates/lk-agents-voice/src/remote_session/driver.rs",
        "lk_agents_voice::remote_session::driver",
    ),
    ("rust/crates/lk-agents-abi/src/errors/host.rs", "lk_agents_abi::errors"),
    ("rust/crates/lk-agents-abi/src/errors/api.rs", "lk_agents_abi::errors"),
    ("rust/crates/lk-agents-abi/src/tts/markup/mod.rs", "lk_agents_abi::tts::markup"),
    ("rust/crates/lk-agents-abi/src/tts/markup/mood.rs", "lk_agents_abi::tts::markup"),
    ("rust/crates/lk-agents-abi/src/llm/chat_context.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/llm/ctx_diff.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/metrics/otel.rs", "lk_agents_abi::metrics"),
    ("rust/crates/lk-agents-abi/src/types/connect.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-voice/src/generation/mod.rs", "lk_agents_voice::generation"),
    (
        "rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs",
        "lk_agents_abi::llm::provider_format::tools",
    ),
    (
        "rust/crates/lk-agents-abi/src/llm/provider_format/host.rs",
        "lk_agents_abi::llm::provider_format::host",
    ),
    ("rust/crates/lk-agents-abi/src/tokenize/thinking.rs", "lk_agents_abi::tokenize::thinking"),
    ("rust/crates/lk-agents-abi/src/tokenize/search.rs", "lk_agents_abi::tokenize::search"),
    ("rust/crates/lk-agents-abi/src/voice/io.rs", "lk_agents_abi::voice::io"),
    ("rust/crates/lk-agents-voice/src/io.rs", "lk_agents_voice::io"),
    ("rust/crates/lk-agents-abi/src/vad/vad.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-rtc/src/traits.rs", "lk_agents_rtc"),
    ("rust/crates/lk-agents-rtc/src/events.rs", "lk_agents_rtc"),
    ("rust/crates/lk-agents-rtc/src/playout.rs", "lk_agents_rtc"),
    ("rust/crates/lk-agents-rtc/src/lib.rs", "lk_agents_rtc"),
    ("rust/crates/lk-agents-abi/src/voice/rtc.rs", "lk_agents_abi::voice::rtc"),
    ("rust/crates/lk-agents-voice/src/room_io/_output.rs", "lk_agents_voice::room_io::_output"),
    (
        "rust/crates/lk-agents-voice/src/room_io/_output/transcription.rs",
        "lk_agents_voice::room_io::_output::transcription",
    ),
    ("rust/crates/lk-agents-voice/src/room_io/rooms.rs", "lk_agents_voice::room_io::rooms"),
    (
        "rust/crates/lk-agents-voice/src/transcription/text_transforms.rs",
        "lk_agents_voice::transcription::text_transforms",
    ),
    ("rust/crates/lk-agents-voice/src/background_audio.rs", "lk_agents_voice::background_audio"),
    ("rust/crates/lk-agents-abi/src/errors/provider.rs", "lk_agents_abi::errors"),
    ("rust/crates/lk-agents-abi/src/tts/tts.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/stt/stt.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/llm/llm.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/turn/detector.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/tts/stream_adapter.rs", "lk_agents_abi"),
    ("rust/crates/lk-agents-abi/src/stt/stream_adapter.rs", "lk_agents_abi::stt::stream_adapter"),
    ("rust/crates/lk-agents-abi/src/types/retry.rs", "lk_agents_abi::types::retry"),
    ("rust/crates/lk-agents-abi/src/telemetry/provider.rs", "lk_agents_abi::telemetry::provider"),
    ("rust/crates/lk-agents-inference/src/fallback/chain.rs", "lk_agents_inference::fallback"),
    ("rust/crates/lk-agents-inference/src/fallback/llm.rs", "lk_agents_inference::fallback"),
    ("rust/crates/lk-agents-inference/src/fallback/stt.rs", "lk_agents_inference::fallback"),
    ("rust/crates/lk-agents-inference/src/fallback/tts.rs", "lk_agents_inference::fallback"),
    ("rust/crates/lk-agents-inference/src/fallback/realtime.rs", "lk_agents_inference::fallback"),
    (
        "rust/crates/lk-agents-inference/src/fallback/realtime_policy.rs",
        "lk_agents_inference::fallback",
    ),
    ("rust/crates/lk-agents-inference/src/stt/mod.rs", "lk_agents_inference::stt"),
    ("rust/crates/lk-agents-inference/src/vad/mod.rs", "lk_agents_inference::vad"),
    ("rust/crates/lk-agents-inference/src/tts/mod.rs", "lk_agents_inference::tts"),
    ("rust/crates/lk-agents-abi/src/llm/realtime.rs", "lk_agents_abi::llm::realtime"),
    ("rust/crates/lk-agents-inference/src/realtime/session.rs", "lk_agents_inference::realtime"),
    ("rust/crates/lk-agents-inference/src/realtime/dialect.rs", "lk_agents_inference::realtime"),
    ("rust/crates/lk-agents-inference/src/stt/options.rs", "lk_agents_inference::stt::options"),
    ("rust/crates/lk-agents-inference/src/tts/wire.rs", "lk_agents_inference::tts::wire"),
    (
        "rust/crates/lk-agents-inference/src/interruption/event.rs",
        "lk_agents_inference::interruption",
    ),
    (
        "rust/crates/lk-agents-inference/src/interruption/options.rs",
        "lk_agents_inference::interruption",
    ),
    (
        "rust/crates/lk-agents-abi/src/tokenize/language/code.rs",
        "lk_agents_abi::tokenize::language",
    ),
    (
        "rust/crates/lk-agents-abi/src/utils/connection_pool.rs",
        "lk_agents_abi::utils::connection_pool",
    ),
    ("rust/crates/lk-agents-voice/src/room_io/_input.rs", "lk_agents_voice::room_io::_input"),
    ("rust/crates/lk-agents-media/src/images/encode.rs", "lk_agents_media::images"),
    ("rust/crates/lk-agents-abi/src/tokenize/word.rs", "lk_agents_abi::tokenize"),
    ("rust/crates/lk-agents-abi/src/evals/group.rs", "lk_agents_abi::evals"),
    ("rust/crates/lk-agents-worker/src/load.rs", "lk_agents_worker"),
    ("rust/crates/lk-agents-worker/src/hw/cpu.rs", "lk_agents_worker"),
    ("rust/crates/lk-agents-worker/src/hw/cgroup.rs", "lk_agents_worker"),
    ("rust/crates/lk-agents-abi/src/llm/tool_arguments.rs", "lk_agents_abi::llm::tool_arguments"),
    ("rust/crates/lk-agents-media/src/blingfire.rs", "lk_agents_media::blingfire"),
    (
        "rust/crates/lk-agents-voice/src/transcription/filters.rs",
        "lk_agents_voice::transcription::filters",
    ),
    (
        "rust/crates/lk-agents-voice/src/transcription/synchronizer.rs",
        "lk_agents_voice::transcription::synchronizer",
    ),
    ("rust/crates/lk-agents-abi/src/llm/schema/defaults.rs", "lk_agents_abi::llm::schema"),
    ("rust/crates/lk-agents-abi/src/tts/emitter/mod.rs", "lk_agents_abi::tts::emitter"),
    ("rust/crates/lk-agents-media/src/images/serialize.rs", "lk_agents_media::images"),
    ("rust/crates/lk-agents-inference/src/mcp/mod.rs", "lk_agents_inference::mcp"),
    ("rust/crates/lk-agents-inference/src/mcp/render.rs", "lk_agents_inference::mcp::render"),
    ("rust/crates/lk-agents-inference/src/mcp/session.rs", "lk_agents_inference::mcp::session"),
];

/// The generated Python package, under whichever Python project the repo has: `python/livekit-agents`
/// (agents-rust) or `livekit-agents` (livekit/agents, where `rust/` sits beside the package).
const PYTHON_PACKAGES: &[&str] = &["python/livekit-agents", "livekit-agents"];
const PYTHON_GENERATED: &str = "livekit/agents/_generated";
const NODE_ROOT: &str = "node/src/generated";
const NODE_BINDING_ROOT: &str = "rust/crates/lk-agents-node/src/generated";

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
    let (root, python_package) = repo_root()?;
    let python_root = root.join(python_package).join(PYTHON_GENERATED);
    let mut decls = Vec::new();
    for (src, module) in SOURCES {
        let path = root.join(src);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("{src}: {e} (the SOURCES list names a file that is not there)"))?;
        decls.extend(parse::file(src, module, &text)?);
    }
    parse::link(&mut decls)?;
    resolve(&decls)?;
    // Aliases first, from every file: a signature may name `Object` or `Extra` from another file.
    let mut texts = Vec::new();
    for (src, module) in EXPORT_SOURCES {
        let text = std::fs::read_to_string(root.join(src))
            .map_err(|e| format!("{src}: {e} (EXPORT_SOURCES names a file that is not there)"))?;
        texts.push((*src, *module, text));
    }
    let mut all_aliases = Vec::new();
    for (src, _) in SOURCES {
        all_aliases
            .extend(parse::aliases(&std::fs::read_to_string(root.join(src)).unwrap_or_default()));
    }
    for (_, _, text) in &texts {
        all_aliases.extend(parse::aliases(text));
    }
    let global = parse::global_aliases(&all_aliases);
    let mut exposed = ir::Exposed::default();
    for (src, module, text) in &texts {
        let x = parse::exposed_with(src, module, text, &global)?;
        if x.is_empty() {
            return Err(format!("{src}: in EXPORT_SOURCES but holds no #[host(..)] item"));
        }
        exposed.extend(x);
    }

    let mut stale = 0;
    for out in python::emit(&decls)? {
        stale += put(&python_root.join(&out.path), &out.text, check)?;
    }
    // The TypeScript side exists in agents-rust only; a repo without `node/` gets none of it.
    if root.join("node").is_dir() {
        for out in node::emit(&decls)? {
            stale += put(&root.join(NODE_ROOT).join(&out.path), &out.text, check)?;
        }
    }
    if let Some(out) = stub::emit(&decls, &exposed)? {
        stale += put(&python_root.join(&out.path), &out.text, check)?;
    }
    for out in napi::emit(&decls, &exposed)? {
        stale += put(&root.join(NODE_BINDING_ROOT).join(&out.path), &out.text, check)?;
    }
    if !check {
        // Written every time so a fresh checkout imports, and the dir is never a namespace package.
        let init = python_root.join("__init__.py");
        std::fs::create_dir_all(init.parent().expect("has a parent")).map_err(|e| e.to_string())?;
        std::fs::write(
            &init,
            "# GENERATED DIRECTORY -- every file here is written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`).\n\
             #\n\
             # Nothing re-exports from this package. The frozen public paths do the re-exporting\n\
             # (`voice/events.py` does `from .._generated.events import ...`), because moving a\n\
             # name here would move it in the public API too.\n",
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(stale)
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

/// Every `Ty::Declared` names a type in the set. Checked after all files (forward references are
/// legal), and an error rather than `Any`.
fn resolve(decls: &[ir::Decl]) -> Result<(), String> {
    for d in decls {
        // Only rendered types must be declared: binding-only fields cross by their own converters.
        let renders_types =
            d.targets.iter().any(|t| matches!(t.host, ir::Host::Python | ir::Host::Node));
        if !renders_types {
            continue;
        }
        // A `#[python(skip)]` field is on no host class, so the type it names never renders
        // (`TurnContext.boundary` is a core object the binding hands over beside the record).
        let node = d.target(ir::Host::Node).is_some();
        for f in d.fields().iter().filter(|f| node || !f.py_skip) {
            for name in declared(&f.ty) {
                if !decls.iter().any(|o| o.rust_name == name) {
                    return Err(format!(
                        "{}: {}.{} names `{name}`, which is not a #[derive(Host)] type. Declare \
                         it, or write it as #[python(ty = \"..\", from = \"..\")] if it is a type \
                         we do not own (docs/PLAN-phase1.md §3.7).",
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
/// Returns the repo root and the Python project under it (one of [`PYTHON_PACKAGES`]).
fn repo_root() -> Result<(PathBuf, &'static str), String> {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if p.join("rust/Cargo.toml").exists()
            && let Some(package) =
                PYTHON_PACKAGES.iter().find(|d| p.join(d).join("pyproject.toml").exists())
        {
            return Ok((p, package));
        }
        if !p.pop() {
            return Err(format!(
                "could not find the repo root (no directory with rust/Cargo.toml and one of {PYTHON_PACKAGES:?} above the manifest)"
            ));
        }
    }
}
