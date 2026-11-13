//! Source rules over the git-tracked Rust files (AGENTS.md "Rust architecture rules"):
//!
//! 1. Handwritten files stay within [`MAX_LINES`] lines ("Layout and tests").
//! 2. In `lk-agents-py`, `#[pyclass]`, `#[pymethods]`, `#[pyfunction]` and `#[pymodule]` appear
//!    only in generated files and the `_core` module shell ("Bindings and the host").
//! 3. In `lk-agents-py`, Python is called only from [`HOST_ENTRY`] files and tests
//!    ("Bindings and the host").
//! 4. The async crates ([`ASYNC`]) use no Tokio runtime (`tokio::spawn`, `tokio::time`,
//!    `tokio::net`, `tokio::runtime`, `Handle::current`) and never detach work (`spawn_detached`):
//!    tasks run on the injected `Spawner` under an owner that joins or cancels them (`TaskSet`,
//!    `ScopedTask`, `owned::Background`), and time comes from the injected `Clock`. Only the
//!    primitives that implement ownership and time, and the boundary adapters, are exempt, each
//!    for what it names ([`EXEMPT`]). In the runtime's DST executor modules ([`EXECUTORS`]),
//!    every `select!` is also `biased;` ("Runtime executors and the DST", "Cancellation safety").
//! 5. In the async crates ([`SELECT_ORDERED`]), no `select!` branch that runs the producer
//!    (`run()`) precedes a branch reading its data (`next()`, `recv()`), so queued data is read
//!    before the end is acted on; branches are matched by method name ("Runtime executors and
//!    the DST").
//! 6. `Unacknowledged::sent`, the one way to an uncertain write, is called only where a write
//!    is handed to its transport: the storage connections and their fakes ([`SENT_WRITES`]);
//!    everything else passes on the `Uncertain` it was given ("Errors").
//! 7. No string literal says an input is closed outside the one shared terminal state
//!    ([`INPUT_STATE`]): a stream's input answers from its `InputState`, so after a provider's
//!    failure every input call returns that failure, never a generic close ("Errors"). A typed
//!    error's `#[error(..)]` message is a closed set, not a literal an input picks.
//! 8. A retry a request decided (`Step::Retry(backoff)`, `Retry::At(backoff)`) is announced
//!    (`backoff.announce(..)`, the one retry log), abandoned (`backoff.abandon()`) or handed on
//!    (the binding moved as a value) in its arm; `matches!` and `_` never swallow one, and a
//!    `match` over a request's steps with a wildcard arm names `Step::Retry`. Only
//!    [`REQUEST`], where retries are decided, is exempt ("Errors").
//!
//! 9. In the async crates, an awaited call's outcome is read: no `let _ = ..await` and no
//!    `if let Ok(..) = ..await { .. }` without `else`. A call whose failure is always safe to
//!    drop says so once, in its own signature, by returning `()` ("Errors").
//! 10. In `lk-agents-runtime`, no `log::error!` in a fallback (an `unwrap_or_else` closure or an
//!     `Err(..) =>` arm): a fallback is a decision, which the core makes from a `Result` input
//!     ("Sans-I/O cores").
//!
//! Rules 4 to 10 skip test files, benches, examples and `#[cfg(test)]` items. pyo3 and `#[host]`
//! placement need no rule here: the crate graph ([`crate::graph`]) leaves the compiler to reject
//! them.

use std::path::Path;
use std::process::Command;

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use quote::ToTokens;

use crate::tokens::{
    attribute, fat_arrow, group, ident, last_call, levels, line, path_segment, punct,
    select_futures, single_eq,
};

const MAX_LINES: usize = 1250;
/// Where a write is handed to its transport, the only places an outcome becomes uncertain
/// (rule 6).
const SENT_WRITES: &[&str] = &[
    "crates/lk-agents-storage/src/backends/",
    "crates/lk-agents-storage/src/testing.rs",
    "crates/lk-agents-fuzz/src/fakes/",
];
const GENERATED: &str = "//! GENERATED FILE -- DO NOT EDIT.";
/// The one place an input's refusal is worded (rule 7).
const INPUT_STATE: &str = "crates/lk-agents-abi/src/utils/stream.rs";
/// What a literal of a refused input says (rule 7), lowercased.
const INPUT_CLOSED: &[&str] = &["input is closed", "input closed"];

/// Where a request decides its retries (rule 8).
const REQUEST: &str = "crates/lk-agents-core/src/request/";
/// A request's steps other than `Retry` (rule 8).
const STEPS: &[&str] = &["Launch", "Stop", "GiveUp", "Unresolved", "CleanupIncomplete"];
/// What applies a retry in its arm (rule 8).
const APPLY_RETRY: &[&str] = &["announce", "abandon"];

/// Where Rust calls Python: HostInvoker's entry and log forwarding (rule 3).
const HOST_ENTRY: &[&str] = &["py_calls.rs", "py_streams.rs", "py_logging.rs"];
const HOST_CALLS: &[&str] =
    &["call", "call0", "call1", "call_method", "call_method0", "call_method1"];
const PYO3_MACROS: &[&str] = &["pyclass", "pymethods", "pyfunction", "pymodule"];

/// The runtime modules the DST runs on its own scheduler (rule 4).
const EXECUTORS: &[&str] = &["voice", "room_io", "avatar", "rtc"];
/// The crates that run beside the host, checked by rule 4.
const ASYNC: &[&str] = &[
    "lk-agents-cli",
    "lk-agents-inference",
    "lk-agents-net",
    "lk-agents-py",
    "lk-agents-runtime",
    "lk-agents-storage",
    "lk-agents-telemetry",
    "lk-agents-worker",
];
/// A use rule 4 bans: `spawn` (`tokio::spawn`, `tokio::task::spawn*`), `time`, `net`,
/// `runtime`, `Handle::current` or `spawn_detached`.
type Item = &'static str;
/// The files rule 4 exempts, each for the items it names and why.
const EXEMPT: &[(&str, &[Item], &str)] = &[
    (
        "crates/lk-agents-net/src/task.rs",
        &["runtime", "spawn", "Handle::current"],
        "implements Spawner on the Tokio runtime",
    ),
    ("crates/lk-agents-net/src/clock.rs", &["time"], "implements Clock on Tokio time"),
    (
        "crates/lk-agents-net/src/env.rs",
        &["runtime", "Handle::current"],
        "ProcessEnv lends the process runtime, the composition root of every async crate",
    ),
    (
        "crates/lk-agents-py/src/py_calls.rs",
        &["runtime"],
        "implements pyo3_async_runtimes' Runtime, the host's executor, on the process runtime",
    ),
    (
        "crates/lk-agents-runtime/src/process.rs",
        &["runtime"],
        "builds and owns the process runtime that ProcessEnv lends",
    ),
    (
        "crates/lk-agents-storage/src/http/server.rs",
        &["net"],
        "the local ThreadService's socket listener boundary adapter",
    ),
    ("crates/lk-agents-net/src/dns.rs", &["net"], "the name-resolution boundary adapter"),
    ("crates/lk-agents-net/src/websocket.rs", &["net"], "the WebSocket boundary adapter"),
    ("crates/lk-agents-cli/src/remote_cli.rs", &["net"], "the dev channel's TCP boundary adapter"),
    (
        "crates/lk-agents-worker/src/ipc/ipc_transport_unix.rs",
        &["net"],
        "the job channel's Unix socket boundary adapter",
    ),
    (
        "crates/lk-agents-worker/src/ipc/ipc_transport_windows.rs",
        &["net"],
        "the job channel's named pipe boundary adapter",
    ),
    (
        "crates/lk-agents-worker/src/server/http.rs",
        &["net"],
        "the health and metrics listeners' socket boundary adapter",
    ),
    (
        "crates/lk-agents-worker/src/server/public.rs",
        &["net"],
        "the public listener's socket boundary adapter",
    ),
    (
        "crates/lk-agents-worker/src/transport/webtransport.rs",
        &["net"],
        "the endpoint streams' QUIC and TCP boundary adapter",
    ),
];
/// Async crates checked by rule 5.
const SELECT_ORDERED: &[&str] = &[
    "lk-agents-inference",
    "lk-agents-net",
    "lk-agents-runtime",
    "lk-agents-storage",
    "lk-agents-worker",
];
const DATA_CALLS: &[&str] = &["next", "next_event", "next_audio", "recv", "recv_many"];
/// A branch driving the producer; `closed()` (the reader left) may come first.
const ENDING_CALLS: &[&str] = &["run"];

/// A tracked Rust file, relative to `rust/`.
struct Source<'a> {
    path: &'a str,
    crate_name: Option<&'a str>,
    text: String,
}

impl Source<'_> {
    fn error(&self, line: usize, message: &str) -> String {
        format!("rust/{}:{line}: {message}", self.path)
    }

    fn generated(&self) -> bool {
        self.text.lines().next().is_some_and(|first| first.trim_end() == GENERATED)
    }

    fn test(&self) -> bool {
        let file = self.path.rsplit('/').next().unwrap_or(self.path);
        file.ends_with("_tests.rs")
            || self.path.contains("/tests/")
            || self.path.contains("/benches/")
            || self.path.contains("/examples/")
    }
}

pub fn check(rust: &Path) -> Vec<String> {
    let mut errors = Vec::new();
    for path in tracked(rust) {
        let Ok(text) = std::fs::read_to_string(rust.join(&path)) else {
            continue;
        };
        let crate_name = path.strip_prefix("crates/").and_then(|rest| rest.split('/').next());
        errors.extend(check_source(&Source { path: &path, crate_name, text }));
    }
    errors
}

fn tracked(rust: &Path) -> Vec<String> {
    let output = Command::new("git")
        .args(["ls-files", "-z", "--", "crates/*.rs", "ai_slop/*.rs", "check/*.rs"])
        .current_dir(rust)
        .output()
        .expect("git runs");
    assert!(output.status.success(), "git ls-files failed");
    String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
        .collect()
}

fn check_source(source: &Source) -> Vec<String> {
    let mut errors = Vec::new();
    if !source.generated() && source.text.lines().count() > MAX_LINES {
        errors
            .push(source.error(1, &format!("over {MAX_LINES} lines: split it by responsibility")));
    }
    let Some(crate_name) = source.crate_name else {
        return errors;
    };
    let Ok(tokens) = source.text.parse::<TokenStream>() else {
        errors.push(source.error(1, "does not tokenize"));
        return errors;
    };
    if crate_name == "lk-agents-py" {
        bindings(source, tokens, &mut errors);
    }
    if source.text.contains("Unacknowledged::sent")
        && !source.test()
        && !SENT_WRITES.iter().any(|path| source.path.starts_with(path))
    {
        unacknowledged(source, without_tests(&source.text), &mut errors);
    }
    if !source.test() && source.path != INPUT_STATE {
        input_closed(source, without_tests(&source.text), &mut errors);
    }
    let steps = source.text.contains("Retry") || source.text.contains("Step::");
    if steps && !source.test() && !source.path.starts_with(REQUEST) {
        retries(source, without_tests(&source.text), &mut errors);
    }
    if ASYNC.contains(&crate_name) && !source.test() {
        let product = without_tests(&source.text);
        discarded(source, product.clone(), &mut errors);
        if crate_name == "lk-agents-runtime" {
            fallback_logged(source, product.clone(), &mut errors);
        }
        unowned(source, product.clone(), &mut errors);
        if executor(source.path) {
            biased(source, product.clone(), &mut errors);
        }
        if SELECT_ORDERED.contains(&crate_name) {
            select_order(source, product, &mut errors);
        }
    }
    errors
}

/// `.await` in `tokens`.
fn awaits(tokens: &[TokenTree]) -> bool {
    tokens
        .windows(2)
        .any(|w| punct(Some(&w[0]), '.') && ident(Some(&w[1])).as_deref() == Some("await"))
}

/// An awaited outcome dropped unread. Rule 9.
fn discarded(source: &Source, tokens: TokenStream, errors: &mut Vec<String>) {
    levels(tokens, &mut |level| {
        for (at, token) in level.iter().enumerate() {
            let statement = |from: usize| {
                let end =
                    (from..level.len()).find(|&i| punct(level.get(i), ';')).unwrap_or(level.len());
                &level[from..end]
            };
            let dropped = match ident(Some(token)).as_deref() {
                Some("let") => {
                    ident(level.get(at + 1)).as_deref() == Some("_")
                        && single_eq(level, at + 2)
                        && awaits(statement(at + 3))
                }
                Some("if") => {
                    ident(level.get(at + 1)).as_deref() == Some("let")
                        && ident(level.get(at + 2)).as_deref() == Some("Ok")
                        && group(level.get(at + 3), Delimiter::Parenthesis).is_some()
                        && single_eq(level, at + 4)
                        && {
                            let body = (at + 5..level.len())
                                .find(|&i| group(level.get(i), Delimiter::Brace).is_some());
                            body.is_some_and(|b| {
                                awaits(&level[at + 5..b])
                                    && ident(level.get(b + 1)).as_deref() != Some("else")
                            })
                        }
                }
                _ => false,
            };
            if dropped {
                errors.push(source.error(
                    line(token),
                    "an awaited outcome dropped unread: return it, report it, or make the callee \
                     return `()` if dropping is always right",
                ));
            }
        }
    });
}

/// `log::error!` in `tokens`, at any depth.
fn logs_error(tokens: TokenStream) -> Option<usize> {
    let mut found = None;
    levels(tokens, &mut |level| {
        for (at, token) in level.iter().enumerate() {
            if found.is_none()
                && ident(Some(token)).as_deref() == Some("log")
                && path_segment(level, at + 1).as_deref() == Some("error")
            {
                found = Some(line(token));
            }
        }
    });
    found
}

/// A fallback that logs an error and goes on: a decision taken in the runtime. Rule 10.
fn fallback_logged(source: &Source, tokens: TokenStream, errors: &mut Vec<String>) {
    levels(tokens, &mut |level| {
        for (at, token) in level.iter().enumerate() {
            let fallback = match ident(Some(token)).as_deref() {
                Some("unwrap_or_else") => {
                    group(level.get(at + 1), Delimiter::Parenthesis).map(|args| args.stream())
                }
                Some("Err")
                    if group(level.get(at + 1), Delimiter::Parenthesis).is_some()
                        && at + 2 < level.len()
                        && fat_arrow(level, at + 2) =>
                {
                    match group(level.get(at + 4), Delimiter::Brace) {
                        Some(body) => Some(body.stream()),
                        None => {
                            let end = (at + 4..level.len())
                                .find(|&i| punct(level.get(i), ','))
                                .unwrap_or(level.len());
                            Some(level[at + 4..end].iter().cloned().collect())
                        }
                    }
                }
                _ => None,
            };
            if let Some(logged) = fallback.and_then(logs_error) {
                errors.push(source.error(
                    logged,
                    "an error logged in a fallback: hand the `Result` to the core, which decides \
                     the fallback",
                ));
            }
        }
    });
}

/// `Unacknowledged::sent` outside a connection: a write called uncertain that no connection
/// sent. Rule 6.
fn unacknowledged(source: &Source, tokens: TokenStream, errors: &mut Vec<String>) {
    levels(tokens, &mut |level| {
        for (at, token) in level.iter().enumerate() {
            if ident(Some(token)).as_deref() == Some("Unacknowledged")
                && path_segment(level, at + 1).as_deref() == Some("sent")
            {
                errors.push(source.error(
                    line(token),
                    "Unacknowledged::sent outside a storage connection: pass on the Uncertain \
                     a sent write returned",
                ));
            }
        }
    });
}

/// A literal refusing input as closed, outside `InputState` and attributes. Rule 7.
fn input_closed(source: &Source, tokens: TokenStream, errors: &mut Vec<String>) {
    let mut attribute = false;
    for token in tokens {
        match &token {
            TokenTree::Group(group) if attribute && group.delimiter() == Delimiter::Bracket => {}
            TokenTree::Group(group) => input_closed(source, group.stream(), errors),
            TokenTree::Literal(literal) => {
                let text = literal.to_string().to_lowercase();
                if text.starts_with('"') && INPUT_CLOSED.iter().any(|said| text.contains(said)) {
                    errors.push(source.error(
                        line(&token),
                        "an input refused as closed: answer from the stream's InputState, so a \
                         provider's failure is what its input returns",
                    ));
                }
            }
            _ => {}
        }
        // `#[..]` and `#![..]`
        attribute = punct(Some(&token), '#') || (attribute && punct(Some(&token), '!'));
    }
}

/// A request's retry that its arm does not apply, or a step `match` whose wildcard would swallow
/// one. Rule 8.
fn retries(source: &Source, tokens: TokenStream, errors: &mut Vec<String>) {
    levels(tokens, &mut |level| {
        let mut steps = false;
        let mut retry = false;
        let mut wildcard = None;
        for (at, token) in level.iter().enumerate() {
            if ident(Some(token)).as_deref() == Some("_")
                && at + 1 < level.len()
                && fat_arrow(level, at + 1)
            {
                wildcard = Some(line(token));
            }
            let (Some(head), Some(variant)) = (ident(Some(token)), path_segment(level, at + 1))
            else {
                continue;
            };
            match (head.as_str(), variant.as_str()) {
                ("Step", "Retry") | ("Retry", "At") => {}
                ("Step", step) if STEPS.contains(&step) => {
                    steps = true;
                    continue;
                }
                _ => continue,
            }
            retry |= head == "Step";
            if !applied(level, at + 4) {
                errors.push(source.error(
                    line(token),
                    "a retry not applied: announce it (`backoff.announce(..)`, the one retry log), \
                     abandon it or hand it on in its arm",
                ));
            }
        }
        if let (true, false, Some(at)) = (steps, retry, wildcard) {
            errors.push(source.error(
                at,
                "a wildcard arm over a request's steps swallows its retries: name `Step::Retry`",
            ));
        }
    });
}

/// Whether the arm of the pattern binding a retry at `at` (its `(binding)` group) applies it.
fn applied(level: &[TokenTree], at: usize) -> bool {
    let Some(binding) = group(level.get(at), Delimiter::Parenthesis).and_then(|binding| {
        let tokens: Vec<TokenTree> = binding.stream().into_iter().collect();
        ident(tokens.first()).filter(|name| tokens.len() == 1 && name != "_")
    }) else {
        return false;
    };
    let mut body = None;
    for at in at + 1..level.len() {
        if punct(level.get(at), ';') {
            break;
        }
        if fat_arrow(level, at) {
            body = Some(match group(level.get(at + 2), Delimiter::Brace) {
                Some(block) => block.stream(),
                None => level
                    .get(at + 2..)
                    .unwrap_or_default()
                    .iter()
                    .take_while(|token| !punct(Some(token), ','))
                    .cloned()
                    .collect(),
            });
            break;
        }
        if single_eq(level, at) {
            body = level[at + 1..]
                .iter()
                .find_map(|token| group(Some(token), Delimiter::Brace))
                .map(proc_macro2::Group::stream);
            break;
        }
    }
    let mut applies = false;
    if let Some(body) = body {
        levels(body, &mut |level| {
            for (at, token) in level.iter().enumerate() {
                let name = ident(Some(token));
                applies |= name.as_deref().is_some_and(|name| APPLY_RETRY.contains(&name))
                    || (name.as_deref() == Some(binding.as_str())
                        && !punct(level.get(at + 1), '.'));
            }
        });
    }
    applies
}

/// pyo3 and `#[py*]` only in `lk-agents-py`; `#[host]` only in ABI; host calls only in the entry.
/// Rules 2 and 3, in `lk-agents-py`.
fn bindings(source: &Source, tokens: TokenStream, errors: &mut Vec<String>) {
    let mentions_pyo3 = source.text.contains("pyo3");
    let host_entry =
        HOST_ENTRY.iter().any(|f| source.path.ends_with(&format!("src/{f}"))) || source.test();
    levels(tokens, &mut |level| {
        for (at, token) in level.iter().enumerate() {
            if mentions_pyo3 && !host_entry && host_call(level, at) {
                errors.push(
                    source.error(line(token), "call Python through HostInvoker (py_calls.rs)"),
                );
            }
            let attr = punct(Some(token), '#')
                .then(|| {
                    group(
                        level.get(at + 1 + usize::from(punct(level.get(at + 1), '!'))),
                        Delimiter::Bracket,
                    )
                })
                .flatten();
            if let Some(name) = attr.and_then(|attr| attribute(attr).pop()) {
                pyo3_macro(source, &name, line(token), errors);
            }
        }
    });
}

fn host_call(level: &[TokenTree], at: usize) -> bool {
    let called = group(level.get(at + 1), Delimiter::Parenthesis).is_some();
    match ident(level.get(at)).as_deref() {
        Some(name) if HOST_CALLS.contains(&name) => {
            called && at > 0 && punct(level.get(at - 1), '.')
        }
        Some("into_future") => called,
        Some("future_into_py") => {
            !(at >= 3 && ident(level.get(at - 3)).as_deref() == Some("py_calls"))
        }
        _ => false,
    }
}

fn pyo3_macro(source: &Source, name: &str, line: usize, errors: &mut Vec<String>) {
    let shell = name == "pymodule" && source.path == "crates/lk-agents-py/src/lib.rs";
    if PYO3_MACROS.contains(&name) && !source.generated() && !shell {
        errors.push(source.error(line, &format!("#[{name}] belongs in generated output")));
    }
}

fn executor(path: &str) -> bool {
    EXECUTORS
        .iter()
        .any(|module| path.starts_with(&format!("crates/lk-agents-runtime/src/{module}/")))
}

/// What `path` may use of rule 4's items.
fn exempt(path: &str) -> &'static [Item] {
    EXEMPT.iter().find(|(file, ..)| *file == path).map_or(&[], |(_, items, _)| items)
}

/// No Tokio runtime and no detached work outside the primitives and adapters. Rule 4.
fn unowned(source: &Source, tokens: TokenStream, errors: &mut Vec<String>) {
    let allowed = exempt(source.path);
    let mut report = |token: &TokenTree, item: Item, message: &str| {
        if !allowed.contains(&item) {
            errors.push(source.error(line(token), message));
        }
    };
    levels(tokens, &mut |level| {
        for (at, token) in level.iter().enumerate() {
            match ident(Some(token)).as_deref() {
                Some("tokio") => {
                    if let Some(used) = tokio_runtime(level, at) {
                        let (item, fix) = match used.as_str() {
                            "time" => ("time", "wait and read time on the injected Clock"),
                            "net" => ("net", "sockets live in the boundary adapters"),
                            "runtime" => ("runtime", "use the injected Spawner"),
                            _ => ("spawn", "spawn on the injected Spawner under an owner"),
                        };
                        report(token, item, &format!("tokio::{used}: {fix}"));
                    }
                }
                Some("Handle")
                    if matches!(
                        path_segment(level, at + 1).as_deref(),
                        Some("current" | "try_current")
                    ) =>
                {
                    report(token, "Handle::current", "Handle::current: use the injected Spawner");
                }
                Some("spawn_detached") => report(
                    token,
                    "spawn_detached",
                    "spawn_detached: nothing joins it; run it through the owner's TaskSet or \
                     owned::Background, whose close joins it",
                ),
                _ => {}
            }
        }
    });
}

/// Only biased selects in the modules the DST schedules. Rule 4.
fn biased(source: &Source, tokens: TokenStream, errors: &mut Vec<String>) {
    levels(tokens, &mut |level| {
        for (at, token) in level.iter().enumerate() {
            if ident(Some(token)).as_deref() == Some("select") && punct(level.get(at + 1), '!') {
                let body = group(level.get(at + 2), Delimiter::Brace)
                    .map(|g| g.stream().into_iter().collect::<Vec<_>>());
                if body.is_some_and(|b| ident(b.first()).as_deref() != Some("biased")) {
                    errors.push(
                        source.error(line(token), "select! in a DST executor must be biased;"),
                    );
                }
            }
        }
    });
}

/// `tokio::spawn`, `tokio::time`, `tokio::net`, `tokio::runtime`, `tokio::task::spawn*`, also
/// inside a `use tokio::{..}` group.
fn tokio_runtime(level: &[TokenTree], at: usize) -> Option<String> {
    let runtime = |name: &str| matches!(name, "spawn" | "time" | "net" | "runtime");
    if let Some(next) = path_segment(level, at + 1) {
        if runtime(&next) {
            return Some(next);
        }
        let spawn = path_segment(level, at + 4).filter(|s| s.starts_with("spawn"));
        return (next == "task").then_some(spawn).flatten().map(|s| format!("task::{s}"));
    }
    let used = punct(level.get(at + 1), ':') && punct(level.get(at + 2), ':');
    let names = group(level.get(at + 3), Delimiter::Brace).filter(|_| used)?;
    names.stream().into_iter().filter_map(|t| ident(Some(&t))).find(|n| runtime(n))
}

/// Data branches before the branches that end the loop, so queued data is read first.
fn select_order(source: &Source, tokens: TokenStream, errors: &mut Vec<String>) {
    levels(tokens, &mut |level| {
        for (at, token) in level.iter().enumerate() {
            if ident(Some(token)).as_deref() != Some("select") || !punct(level.get(at + 1), '!') {
                continue;
            }
            let Some(body) = group(level.get(at + 2), Delimiter::Brace) else {
                continue;
            };
            let body: Vec<TokenTree> = body.stream().into_iter().collect();
            let mut ending = None;
            for call in select_futures(&body).iter().filter_map(|f| last_call(f)) {
                if ENDING_CALLS.contains(&call.as_str()) {
                    ending.get_or_insert(call);
                } else if let Some(end) =
                    ending.as_ref().filter(|_| DATA_CALLS.contains(&call.as_str()))
                {
                    let message =
                        format!("select! branch {end}() precedes {call}(): read queued data first");
                    errors.push(source.error(line(token), &message));
                }
            }
        }
    });
}

/// The file's tokens without `#[cfg(test)]` and `#[test]` items.
fn without_tests(text: &str) -> TokenStream {
    let Ok(mut file) = syn::parse_file(text) else {
        return text.parse().unwrap_or_default();
    };
    strip_tests(&mut file.items);
    file.into_token_stream()
}

fn strip_tests(items: &mut Vec<syn::Item>) {
    items.retain(|item| !test_item(item));
    for item in items {
        if let syn::Item::Mod(syn::ItemMod { content: Some((_, inner)), .. }) = item {
            strip_tests(inner);
        }
    }
}

fn test_item(item: &syn::Item) -> bool {
    let attrs: &[syn::Attribute] = match item {
        syn::Item::Fn(f) => &f.attrs,
        syn::Item::Mod(m) => &m.attrs,
        syn::Item::Impl(i) => &i.attrs,
        syn::Item::Use(u) => &u.attrs,
        syn::Item::Struct(s) => &s.attrs,
        syn::Item::Enum(e) => &e.attrs,
        syn::Item::Const(c) => &c.attrs,
        syn::Item::Static(s) => &s.attrs,
        _ => return false,
    };
    attrs.iter().any(|attr| {
        let text = attr.to_token_stream().to_string().replace(' ', "");
        text == "#[test]" || text.starts_with("#[cfg(test") || text.starts_with("#[cfg(all(test")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn errors(path: &str, text: &str) -> Vec<String> {
        let crate_name = path.strip_prefix("crates/").and_then(|rest| rest.split('/').next());
        check_source(&Source { path, crate_name, text: text.to_owned() })
    }

    const EXECUTOR: &str = "crates/lk-agents-runtime/src/voice/x.rs";

    #[test]
    fn clean_sources_pass() {
        assert!(
            errors(
                EXECUTOR,
                "async fn f() { select! { biased; x = rx.recv() => x, _ = t.closed() => () } }"
            )
            .is_empty()
        );
        assert!(
            errors(
                "crates/lk-agents-py/src/lib.rs",
                "use pyo3::prelude::*; #[pymodule] fn _core() {}"
            )
            .is_empty()
        );
        assert!(
            errors(EXECUTOR, "#[cfg(test)] mod tests { fn f() { tokio::spawn(x); } }").is_empty()
        );
    }

    #[test]
    fn handwritten_pyclass_fails() {
        assert_eq!(errors("crates/lk-agents-py/src/x.rs", "#[pyclass] struct A;").len(), 1);
    }

    #[test]
    fn python_calls_outside_the_entry_fail() {
        let text = "use pyo3::Python; fn f(o: &Py) { o.call0(); }";
        assert_eq!(errors("crates/lk-agents-py/src/x.rs", text).len(), 1);
        assert!(errors("crates/lk-agents-py/src/py_calls.rs", text).is_empty());
    }

    #[test]
    fn tokio_runtime_in_an_async_crate_fails() {
        assert_eq!(errors(EXECUTOR, "fn f() { tokio::spawn(x); }").len(), 1);
        assert_eq!(errors(EXECUTOR, "use tokio::{sync, time};").len(), 1);
        assert_eq!(errors(EXECUTOR, "fn f() { tokio::task::spawn_local(x); }").len(), 1);
        let storage = "crates/lk-agents-storage/src/thread/lease.rs";
        assert_eq!(errors(storage, "fn f() { tokio::time::sleep(d); }").len(), 1);
        assert_eq!(errors(storage, "fn f() { Handle::try_current(); }").len(), 1);
        assert!(errors("crates/lk-agents-core/src/x.rs", "fn f() { tokio::spawn(x); }").is_empty());
    }

    // Catches: an exemption leaking past what it names.
    #[test]
    fn an_exemption_allows_only_what_it_names() {
        let clock = "crates/lk-agents-net/src/clock.rs";
        assert!(errors(clock, "async fn f() { tokio::time::sleep(d).await }").is_empty());
        assert_eq!(errors(clock, "fn f() { tokio::spawn(x); }").len(), 1);
    }

    // Catches: work that outlives its caller detached from its owner's tasks, so nothing joins
    // it at the close.
    #[test]
    fn detached_work_in_an_async_crate_fails() {
        assert_eq!(errors(EXECUTOR, "fn f(s: &dyn Spawner) { s.spawn_detached(w); }").len(), 1);
        let storage = "crates/lk-agents-storage/src/storage.rs";
        assert_eq!(errors(storage, "fn f(s: &dyn Spawner) { s.spawn_detached(w); }").len(), 1);
    }

    // Catches: a write called uncertain that no connection sent, so a write that never left
    // could be resolved, or reported, as maybe applied.
    #[test]
    fn an_uncertain_write_outside_a_connection_fails() {
        let text = "fn f() -> E { E::Uncertain(Unacknowledged::sent(\"lost\")) }";
        assert_eq!(errors("crates/lk-agents-storage/src/thread/run.rs", text).len(), 1);
        let connection = "crates/lk-agents-storage/src/backends/sqlite/connection.rs";
        assert!(errors(connection, text).is_empty());
        assert!(errors("crates/lk-agents-core/src/a2a/x_tests.rs", text).is_empty());
    }

    #[test]
    fn unbiased_select_in_an_executor_fails() {
        assert_eq!(errors(EXECUTOR, "async fn f() { select! { x = rx.recv() => x } }").len(), 1);
    }

    #[test]
    fn ending_branch_before_data_fails() {
        let text =
            "async fn f() { select! { biased; e = pump.run() => e, x = source.next() => x } }";
        assert_eq!(errors("crates/lk-agents-inference/src/x.rs", text).len(), 1);
    }

    #[test]
    fn an_input_refused_as_closed_outside_its_state_fails() {
        let text = "fn f() -> E { ApiError::new(\"stream input is closed\").into() }";
        assert_eq!(errors("crates/lk-agents-runtime/src/voice/x.rs", text).len(), 1);
        let generated = "fn f() { new_err(\"Stream Input Closed\") }";
        assert_eq!(errors("crates/lk-agents-py/src/generated/x.rs", generated).len(), 1);
        assert!(errors(INPUT_STATE, text).is_empty());
        let typed = "enum E { #[error(\"decoder input is closed\")] Closed }";
        assert!(errors("crates/lk-agents-inference/src/x.rs", typed).is_empty());
        let test = "#[cfg(test)] mod tests { const S: &str = \"input is closed\"; }";
        assert!(errors("crates/lk-agents-net/src/x.rs", test).is_empty());
    }

    // Catches: a retry applied without its one log, or swallowed by a wildcard or `matches!`.
    #[test]
    fn a_retry_not_applied_in_its_arm_fails() {
        let path = "crates/lk-agents-inference/src/x.rs";
        let applied = [
            "fn f() { match r { Retry::At(b) => { let at = b.announce(s, &e); w(at) } _ => {} } }",
            "fn f() { if let Retry::At(b) = r { b.announce(s, &e); } }",
            "fn f() { match s { Step::Retry(b) if t => self.held = Some(b), _ => {} } }",
            "fn f() { match s { Step::Retry(b) => fail(b.abandon()), Step::Launch(i) => go(i), \
             _ => {} } }",
        ];
        for text in applied {
            assert!(errors(path, text).is_empty(), "{text}");
        }
        let swallowed = [
            "fn f() { match r { Retry::At(b) => wait(b.error()), _ => {} } }",
            "fn f() { if let Retry::At(_) = r { report(); } }",
            "fn f() { matches!(s, Step::Retry(_)); }",
            "fn f() { match s { Step::Launch(i) => go(i), _ => {} } }",
        ];
        for text in swallowed {
            assert_eq!(errors(path, text).len(), 1, "{text}");
        }
        let decided = "fn f() { match s { Step::Retry(b) => {} _ => {} } }";
        assert!(errors("crates/lk-agents-core/src/request/machine.rs", decided).is_empty());
    }

    // Catches: an awaited host, storage or network failure dropped with `let _ =` or an
    // `if let Ok` with no `else`.
    #[test]
    fn an_awaited_outcome_dropped_unread_fails() {
        let path = "crates/lk-agents-storage/src/voice.rs";
        assert_eq!(errors(path, "async fn f() { let _ = run.abandon(o).await; }").len(), 1);
        let unread = "async fn f() { if let Ok(s) = db.snapshot().await { use_it(s); } }";
        assert_eq!(errors(path, unread).len(), 1);
        let read = [
            "async fn f() { if let Ok(s) = db.snapshot().await { a(s) } else { b() } }",
            "async fn f() { let _ = tx.send(x); }",
            "async fn f() { notify(x).await; }",
        ];
        for text in read {
            assert!(errors(path, text).is_empty(), "{text}");
        }
        assert!(
            errors("crates/lk-agents-core/src/x.rs", "async fn f() { let _ = a.await; }")
                .is_empty()
        );
    }

    // Catches: a runtime fallback that logs and carries on, a decision the core never sees.
    #[test]
    fn an_error_logged_in_a_runtime_fallback_fails() {
        let path = "crates/lk-agents-runtime/src/voice/x.rs";
        let fallbacks = [
            "fn f() { r.unwrap_or_else(|e| { log::error!(\"failed: {e}\"); None }); }",
            "fn f() { match r { Ok(v) => v, Err(e) => { log::error!(\"{e}\"); d() } } }",
            "fn f() { match r { Ok(v) => v, Err(e) => log::error!(\"{e}\"), } }",
        ];
        for text in fallbacks {
            assert_eq!(errors(path, text).len(), 1, "{text}");
        }
        assert!(errors(path, "fn f() { match r { Err(e) => return Err(e), _ => {} } }").is_empty());
        let inference = "crates/lk-agents-inference/src/x.rs";
        assert!(errors(inference, fallbacks[0]).is_empty(), "only the runtime");
    }

    #[test]
    fn long_files_fail() {
        assert_eq!(errors("crates/lk-agents-core/src/x.rs", &"\n".repeat(MAX_LINES + 1)).len(), 1);
    }
}
