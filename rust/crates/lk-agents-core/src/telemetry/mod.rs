//! Telemetry decisions on supplied data: GenAI provider names ([`gen_ai`]), destination content
//! policy and retention charges ([`policy`]), and the session report's privacy projection, records
//! and tags ([`report`]). Spans are recorded through `lk-agents-telemetry`.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub use lk_agents_abi::telemetry::spans::common::log_keys;

pub mod gen_ai;
pub mod policy;
pub mod report;
