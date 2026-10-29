//! Telemetry on supplied data: GenAI provider names ([`gen_ai`]), destination redaction
//! and SDK retention charges ([`policy`]), metric series admission ([`metrics`]), event-loop stall
//! classification ([`loop_monitor`]) and the W3C parents transports carry ([`transport`]).
//! Recording and delivery go through the host's OpenTelemetry SDK.

pub mod gen_ai;
pub mod loop_monitor;
pub mod metrics;
pub mod policy;
pub mod transport;
