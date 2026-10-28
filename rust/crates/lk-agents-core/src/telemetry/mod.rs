//! Telemetry policy on supplied data: destination redaction and SDK retention charges
//! ([`policy`]), metric series admission ([`metrics`]) and event-loop stall classification
//! ([`loop_monitor`]). Recording and delivery go through the host's OpenTelemetry SDK.

pub mod loop_monitor;
pub mod metrics;
pub mod policy;
