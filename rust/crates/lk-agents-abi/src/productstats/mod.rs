use crate::telemetry::decl::{Attrs, Bounded, Label, SpanValue};
use crate::telemetry::record::Measurement;
use crate::telemetry::semconv::otel;
use runner::Runner;

// LiveKit's own fleet statistics of the framework and its plugins, never customer data: every
// label is a closed enum or a bounded identifier, never content, an id or a name of a room,
// participant, job, thread or agent. Every key starts with `lk.product.agents.`, which no other
// declaration uses. `lk-agents-codegen --check` refuses a declaration of this module that breaks
// either rule.

pub mod features;
pub mod models;
pub mod runner;
pub mod sessions;
#[cfg(any(test, feature = "testkit"))]
pub mod testkit;

/// One measurement of a declared product statistic, as it travels among a recorder's records.
/// Only the `ProductStats` audience makes one, from a measurement of a declaration of its own.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductStat(Measurement);

impl ProductStat {
    pub(crate) fn new(measurement: Measurement) -> Self {
        ProductStat(measurement)
    }

    pub fn into_measurement(self) -> Measurement {
        self.0
    }
}

/// The runtime an agent runs on, which its host binding names.
#[crate::host(native_enum(name = "Runtime"))]
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeName {
    Python,
    Node,
    Bun,
}

/// Where the process stores its threads.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadStorage {
    AgentDb,
    Sqlite,
    None,
}

/// What every product statistics export says of its process.
#[derive(Attrs, Bounded, Debug, Clone, PartialEq)]
pub struct ProductStatsResource {
    /// The framework's service name, `livekit-agents`.
    #[attr(otel::SERVICE_NAME)]
    pub service: Label,
    /// The framework's version.
    #[attr("lk.product.agents.framework.version")]
    pub framework_version: Label,
    /// The runtime the agent runs on.
    #[attr("lk.product.agents.runtime.name")]
    pub runtime_name: RuntimeName,
    /// That runtime's version.
    #[attr("lk.product.agents.runtime.version")]
    pub runtime_version: Label,
    /// The operating system (`std::env::consts::OS`).
    #[attr("lk.product.agents.os")]
    pub os: Label,
    /// The CPU architecture (`std::env::consts::ARCH`).
    #[attr("lk.product.agents.arch")]
    pub arch: Label,
    #[attrs]
    pub runner: Runner,
    /// Where threads are stored.
    #[attr("lk.product.agents.storage")]
    pub storage: ThreadStorage,
}
