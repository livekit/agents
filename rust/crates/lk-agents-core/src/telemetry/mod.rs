//! Telemetry on supplied data: GenAI provider names ([`gen_ai`]), destination content policy and
//! retention charges ([`policy`]), provider metric folds ([`provider`]), the log fields a metrics
//! event becomes ([`records`]), the session report's privacy projection, records and tags
//! ([`report`]), the spans sans-I/O machines own ([`span`]) and the W3C parents transports carry
//! ([`transport`]).

pub mod gen_ai;
pub mod policy;
pub mod provider;
pub mod records;
pub mod report;
pub mod span;
pub mod transport;
