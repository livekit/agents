//! Source rules over the git-tracked Rust files (AGENTS.md "Rust architecture rules"):
//!
//! 1. Handwritten files stay within [`MAX_LINES`] lines ("Layout and tests").
//! 2. In `lk-agents-py`, `#[pyclass]`, `#[pymethods]`, `#[pyfunction]` and `#[pymodule]` appear
//!    only in generated files and the `_core` module shell ("Bindings and the host").
//! 3. In `lk-agents-py`, Python is called only from [`HOST_ENTRY`] files and tests
//!    ("Bindings and the host").
//! 4. The runtime's DST executor modules ([`EXECUTORS`]) use no Tokio runtime (`tokio::spawn`,
//!    `tokio::time`, `tokio::net`, `tokio::runtime`, `Handle::current`) and every `select!` is
//!    `biased;` ("Runtime executors and the DST").
//! 5. In the async crates ([`SELECT_ORDERED`]), no `select!` branch that runs the producer
//!    (`run()`) precedes a branch reading its data (`next()`, `recv()`), so queued data is read
//!    before the end is acted on ("Runtime executors and the DST").
//!
//! Rules 4 and 5 skip test files and `#[cfg(test)]` items. pyo3 and `#[host]` placement need no
//! rule here: the crate graph ([`crate::graph`]) leaves the compiler to reject them.

use std::path::Path;
use std::process::Command;

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use quote::ToTokens;

use crate::tokens::{
    attribute, group, ident, last_call, levels, line, path_segment, punct, select_futures,
};

const MAX_LINES: usize = 1250;
const GENERATED: &str = "//! GENERATED FILE -- DO NOT EDIT.";

/// Where Rust calls Python: HostInvoker's entry and log forwarding (rule 3).
const HOST_ENTRY: &[&str] = &["py_calls.rs", "py_streams.rs", "py_logging.rs"];
const HOST_CALLS: &[&str] =
    &["call", "call0", "call1", "call_method", "call_method0", "call_method1"];
const PYO3_MACROS: &[&str] = &["pyclass", "pymethods", "pyfunction", "pymodule"];

/// The runtime modules the DST runs on its own scheduler (rule 4).
const EXECUTORS: &[&str] = &["voice", "room_io", "avatar", "rtc"];
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
    if SELECT_ORDERED.contains(&crate_name) && !source.test() {
        let product = without_tests(&source.text);
        if executor(source.path) {
            executors(source, product.clone(), &mut errors);
        }
        select_order(source, product, &mut errors);
    }
    errors
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

/// No Tokio runtime and only biased selects in the modules the DST schedules.
fn executors(source: &Source, tokens: TokenStream, errors: &mut Vec<String>) {
    levels(tokens, &mut |level| {
        for (at, token) in level.iter().enumerate() {
            match ident(Some(token)).as_deref() {
                Some("tokio") => {
                    if let Some(item) = tokio_runtime(level, at) {
                        let message = format!(
                            "tokio::{item} in a DST executor: use the injected Spawner and Clock"
                        );
                        errors.push(source.error(line(token), &message));
                    }
                }
                Some("Handle")
                    if matches!(
                        path_segment(level, at + 1).as_deref(),
                        Some("current" | "try_current")
                    ) =>
                {
                    errors.push(source.error(
                        line(token),
                        "Handle::current in a DST executor: use the injected Spawner",
                    ));
                }
                Some("select") if punct(level.get(at + 1), '!') => {
                    let body = group(level.get(at + 2), Delimiter::Brace)
                        .map(|g| g.stream().into_iter().collect::<Vec<_>>());
                    if body.is_some_and(|b| ident(b.first()).as_deref() != Some("biased")) {
                        errors.push(
                            source.error(line(token), "select! in a DST executor must be biased;"),
                        );
                    }
                }
                _ => {}
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
    fn tokio_runtime_in_an_executor_fails() {
        assert_eq!(errors(EXECUTOR, "fn f() { tokio::spawn(x); }").len(), 1);
        assert_eq!(errors(EXECUTOR, "use tokio::{sync, time};").len(), 1);
        assert_eq!(errors(EXECUTOR, "fn f() { tokio::task::spawn_local(x); }").len(), 1);
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
    fn long_files_fail() {
        assert_eq!(errors("crates/lk-agents-core/src/x.rs", &"\n".repeat(MAX_LINES + 1)).len(), 1);
    }
}
