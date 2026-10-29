//! Telemetry contracts: the wire attribute names, the policy and provider metric declarations, the
//! session report's records. Rust emits completed records; the host materialises them in its own
//! OTel SDK.

pub mod policy;
pub mod provider;

/// Every string is a dashboard wire contract. None carries user content.
pub mod attrs {
    /// Provider correlation ids, for joining against provider logs.
    pub const PROVIDER_REQUEST_IDS: &str = "lk.provider_request_ids";
}

pub mod attributes;
pub mod report;

pub mod observation;
pub mod transport;
pub mod upload;
