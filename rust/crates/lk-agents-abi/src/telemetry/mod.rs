//! Telemetry contracts: the span, metric and report record declarations, the wire attribute names,
//! the policy and provider metric declarations, the session report's records.

pub mod policy;
pub mod provider;

pub mod attributes;
pub mod decl;
pub mod host;
pub mod instruments;
pub mod record;
pub mod report;
pub mod report_records;
pub mod semconv;
pub mod spans;
