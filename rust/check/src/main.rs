//! The `[check]` rules of AGENTS.md "Rust architecture rules" that Clippy and codegen do not
//! cover. Run by `make -C rust check`. Each module lists the rules it checks:
//!
//! - [`graph`]: the crate graph, sans-I/O dependency bans, pyo3 dependencies.
//! - [`sources`]: file length, binding placement, host calls, DST executors, `select!` order.
//! - [`tokens`]: token helpers for `sources`; it checks nothing itself.

mod graph;
mod sources;
mod tokens;

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let rust = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("rust/check sits in rust/");
    let mut errors = graph::check(rust);
    errors.extend(sources::check(rust));
    for error in &errors {
        eprintln!("{error}");
    }
    if errors.is_empty() {
        return ExitCode::SUCCESS;
    }
    eprintln!(
        "{} architecture violation(s); see AGENTS.md \"Rust architecture rules\"",
        errors.len()
    );
    ExitCode::FAILURE
}
