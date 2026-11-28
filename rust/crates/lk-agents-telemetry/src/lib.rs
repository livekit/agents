//! Records the spans, events and measurements declared in `lk-agents-abi`: [`span::ManualSpan`]
//! for sans-I/O machines, [`ScopedSpan`] for async work, and [`process`] instruments.
//!
//! # What keeps the two audiences apart
//!
//! A declaration is the customer's observability or LiveKit's product statistics, and its name,
//! its module and its audience must agree, or it does not compile. `#[span]`, `#[event]` and
//! `#[metric]` expand to `declared::<Audience>(name, module_path!())`:
//!
//! ```
//! use lk_agents_abi::telemetry::decl::{Observability, ProductStats, declared};
//! const _: () = declared::<ProductStats>("lk.product.agents.sessions", "lk_agents_abi::productstats::sessions");
//! const _: () = declared::<Observability>("lk.agents.turn.e2e_latency", "lk_agents_abi::telemetry::instruments");
//! ```
//!
//! A product statistic without the prefix:
//!
//! ```compile_fail,E0080
//! use lk_agents_abi::telemetry::decl::{ProductStats, declared};
//! const _: () = declared::<ProductStats>("lk.agents.sessions", "lk_agents_abi::productstats::sessions");
//! ```
//!
//! The prefix on a customer declaration:
//!
//! ```compile_fail,E0080
//! use lk_agents_abi::telemetry::decl::{Observability, declared};
//! const _: () = declared::<Observability>("lk.product.agents.x", "lk_agents_abi::telemetry::instruments");
//! ```
//!
//! A product statistic declared outside `abi::productstats`:
//!
//! ```compile_fail,E0080
//! use lk_agents_abi::telemetry::decl::{ProductStats, declared};
//! const _: () = declared::<ProductStats>("lk.product.agents.x", "lk_agents_abi::telemetry::instruments");
//! ```
//!
//! A customer declaration inside it:
//!
//! ```compile_fail,E0080
//! use lk_agents_abi::telemetry::decl::{Observability, declared};
//! const _: () = declared::<Observability>("lk.agents.x", "lk_agents_abi::productstats");
//! ```
//!
//! `#[derive(Attrs)]` expands to `key_declared(key, module_path!(), owned)` for each key. A
//! product statistics key outside the module, and another key of ours inside it:
//!
//! ```compile_fail,E0080
//! use lk_agents_abi::telemetry::decl::key_declared;
//! const _: () = key_declared("lk.product.agents.model.name", "lk_agents_abi::telemetry::spans", true);
//! ```
//!
//! ```compile_fail,E0080
//! use lk_agents_abi::telemetry::decl::key_declared;
//! const _: () = key_declared("lk.pii.transcript", "lk_agents_abi::productstats::models", true);
//! ```
//!
//! A product statistic's fields are `Bounded` (`#[metric(audience = ProductStats)]` and
//! `#[derive(Bounded)]` expand to `bounded::<FieldType>()`): free text and content are not.
//!
//! ```compile_fail,E0277
//! lk_agents_abi::telemetry::decl::bounded::<String>();
//! ```
//!
//! ```compile_fail,E0277
//! use lk_agents_abi::telemetry::decl::{Content, bounded};
//! bounded::<Content<String>>();
//! ```
//!
//! And nothing outside ABI can declare a type `Bounded`:
//!
//! ```compile_fail,E0603
//! struct Transcript(String);
//! impl lk_agents_abi::telemetry::decl::bounded::Sealed for Transcript {}
//! ```

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![cfg_attr(
    test,
    allow(
        clippy::disallowed_methods,
        reason = "tests drive Tokio's paused clock and spawn directly; the library is linted without"
    )
)]

// `#[span]` functions name the span API by its crate path, here as in every caller.
extern crate self as lk_agents_telemetry;

mod observability;
pub mod process;
pub mod span;

pub use observability::*;

#[cfg(any(test, feature = "testkit"))]
pub mod testkit;
