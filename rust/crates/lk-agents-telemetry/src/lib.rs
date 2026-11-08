//! Records the spans, events and measurements declared in `lk-agents-abi`: [`span::ManualSpan`]
//! for sans-I/O machines, [`ScopedSpan`] for async work, and [`process`] instruments.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

// `#[span]` functions name the span API by its crate path, here as in every caller.
extern crate self as lk_agents_telemetry;

pub mod process;
mod scoped;
pub mod span;

pub use scoped::*;
