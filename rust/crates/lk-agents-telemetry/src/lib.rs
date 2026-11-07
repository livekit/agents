//! `lk-agents-telemetry`: recording the framework's spans, events and measurements
//! (docs/internals/telemetry.md "Span API"). Spans are declared in `lk-agents-abi`; what they
//! record leaves as `Record`s, which `lk_agents_net::export` exports.
//!
//! The two span handles:
//! - [`span::ManualSpan`]: owned by a sans-I/O state machine, opened and ended through its
//!   [`span::Trace`].
//! - [`ScopedSpan`]: a span of async work under the process [`Tracer`], ended when its scope
//!   exits.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

// `#[span]` functions name the span API by its crate path, here as in every caller.
extern crate self as lk_agents_telemetry;

mod scoped;
pub mod span;

pub use scoped::*;
