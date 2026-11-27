use crate::telemetry::decl::{Attrs, Label};
use crate::telemetry::runner::Runner;
use crate::telemetry::semconv::otel;

// LiveKit's own fleet statistics of the framework and its plugins, never customer data: every
// label is a closed enum or a bounded identifier, never content, an id or a name of a room,
// participant, job, thread or agent. `lk-agents-codegen --check` refuses a declaration of this
// module that breaks it. No statistic is declared: nothing is collected.

#[cfg(any(test, feature = "testkit"))]
pub mod testkit;

/// What every product statistics export says of its process.
#[derive(Attrs, Debug, Clone, PartialEq)]
pub struct ProductStatsResource {
    /// The framework's service name, `livekit-agents`.
    #[attr(otel::SERVICE_NAME)]
    pub service: Label,
    /// The framework's version.
    #[attr("lk.productstats.framework.version")]
    pub framework_version: Label,
    /// The Python version the framework runs on.
    #[attr("lk.productstats.python.version")]
    pub python_version: Option<Label>,
    /// The operating system (`std::env::consts::OS`).
    #[attr("lk.productstats.os")]
    pub os: Label,
    /// The CPU architecture (`std::env::consts::ARCH`).
    #[attr("lk.productstats.arch")]
    pub arch: Label,
    #[attrs]
    pub runner: Runner,
}
