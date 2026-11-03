//! Telemetry contracts: the span, metric and report record declarations, the wire attribute names,
//! the policy and provider metric declarations, the session report's records. Rust emits completed
//! records; the host materialises them in its own OTel SDK.

pub mod policy;
pub mod provider;

/// Every string is a dashboard wire contract. None carries user content.
pub mod attrs {
    /// Provider correlation ids, for joining against provider logs.
    pub const PROVIDER_REQUEST_IDS: &str = "lk.provider_request_ids";

    /// The adaptive interruption verdict on `user_speaking`.
    pub const IS_INTERRUPTION: &str = "lk.is_interruption";
    pub const INTERRUPTION_PROBABILITY: &str = "lk.interruption.probability";
    pub const INTERRUPTION_TOTAL_DURATION: &str = "lk.interruption.total_duration";
    pub const INTERRUPTION_PREDICTION_DURATION: &str = "lk.interruption.prediction_duration";
    pub const INTERRUPTION_DETECTION_DELAY: &str = "lk.interruption.detection_delay";
}

pub mod attributes;
pub mod decl;
pub mod host;
pub mod instruments;
pub mod record;
pub mod report;
pub mod report_records;
pub mod semconv;
pub mod spans;

pub mod transport;
pub mod upload;
