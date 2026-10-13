//! `livekit.agents._core` -- the PyO3 binding.
//!
//! Pure glue: every class, function and exception `_core` holds is declared beside its Rust item
//! in the core crate that owns it (`#[host(..)]`, expanded in place behind that crate's `python`
//! feature, which only this crate turns on) and registers itself; `#[pymodule]` performs the
//! registrations.
//!
//! Importing this module must start nothing (§1.13): everything that starts lives behind
//! `init(role)`, and every entry point calls `check_pid()` first, so a fork-inherited core raises
//! instead of deadlocking.

use pyo3::prelude::*;

// Linked for their registrations alone: nothing here names their items, and a crate the binary
// never mentions is not linked (its `inventory` entries with it).
extern crate lk_agents_inference as _;
extern crate lk_agents_media as _;
extern crate lk_agents_rtc as _;
extern crate lk_agents_voice as _;
extern crate lk_agents_worker as _;

/// `init(role)` and the registration table.
pub mod runtime;
/// `Storage`, `ThreadDb`, `Lease`, `Streams`, `StreamReader`, `Registry` and the storage errors.
pub mod storage;
/// `Thread`, `Turn` and `Boundary`: the run and the turn of `lk-agents-storage` over a Python host.
pub mod turn;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Nothing here may spawn, connect, or allocate a runtime. The core's `log` lines reach the
    // Python logger of the same name from import on (installing a logger starts nothing).
    let _ = pyo3_log::try_init();
    runtime::register_all(m)
}
