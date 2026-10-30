//! Telemetry on supplied data: GenAI provider names ([`gen_ai`]), destination redaction and SDK
//! retention charges ([`policy`]), metric series admission ([`metrics`]), provider metric folds
//! ([`provider`]), the records a metrics event becomes ([`records`]), the session report's
//! privacy projection and record shape ([`report`]) and the W3C parents transports carry
//! ([`transport`]).
//! Recording and delivery go through the host's OpenTelemetry SDK.

pub mod gen_ai;
pub mod metrics;
pub mod policy;
pub mod provider;
pub mod records;
pub mod report;
pub mod transport;
