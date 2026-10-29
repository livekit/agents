//! Telemetry on supplied data: GenAI provider names ([`gen_ai`]), destination redaction and SDK
//! retention charges ([`policy`]), metric series admission ([`metrics`]), provider metric folds
//! ([`provider`]), the records a metrics event becomes ([`records`]), the W3C parents
//! transports carry ([`transport`]) and the per-model usage rollup ([`usage`]).
//! Recording and delivery go through the host's OpenTelemetry SDK.

pub mod gen_ai;
pub mod metrics;
pub mod policy;
pub mod provider;
pub mod records;
pub mod transport;
pub mod usage;
