//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `invoke` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/invoke.rs.
type LoggingImpl = <crate::Selected as super::Implementations>::Logging;

/// Waits until every line the core logged so far has reached the host's logging. Tests call
/// it at their boundaries; the interpreter's exit calls it.
#[pyfunction]
pub fn flush_logs(py: Python<'_>) -> PyResult<()> {
    py.detach(|| <LoggingImpl as ::lk_agents_abi::invoke::Logging>::flush_logs());
    Ok(())
}
