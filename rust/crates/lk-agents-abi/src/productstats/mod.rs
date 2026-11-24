use crate::errors::{ApiError, ApiErrorKind, ProviderError};
use crate::telemetry::decl::{Attrs, Label, SpanValue, metric};
use crate::telemetry::instruments::DURATION_BOUNDS;
use crate::telemetry::runner::Runner;
use crate::telemetry::semconv::otel;

// LiveKit's own fleet statistics of the framework and its plugins, never customer data: every
// label is a closed enum or a bounded identifier (provider, model, plugin, version, frame
// names), never content, an id or a name of a room, participant, job, thread or agent.
// `lk-agents-codegen --check` refuses a declaration of this module that breaks it.

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
    /// Each loaded LiveKit plugin, as `package==version`.
    #[attr("lk.productstats.plugins")]
    pub plugins: Vec<Label>,
}

/// The kind of model a provider statistic is about.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Stt,
    Tts,
    Llm,
    Realtime,
    Eot,
    Vad,
}

/// The provider, model and operation a statistic is about.
#[derive(Attrs, Debug, Clone, PartialEq)]
pub struct ProviderLabels {
    /// STT, TTS, LLM, realtime, end of turn or VAD.
    #[attr("lk.productstats.operation")]
    pub operation: Operation,
    /// The provider, as the GenAI registry spells it.
    #[attr("lk.productstats.provider")]
    pub provider: Label,
    /// The model's name.
    #[attr("lk.productstats.model")]
    pub model: Label,
}

/// The class of a provider failure: the closed part of `error.type`.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureType {
    Timeout,
    Cancelled,
    ApiError,
    ApiStatusError,
    ApiConnectionError,
    /// A plugin's exception of any other class.
    Exception,
}

impl FailureType {
    pub fn of_api(error: &ApiError) -> Self {
        FailureType::of_kind(error.kind)
    }

    pub fn of(error: &ProviderError) -> Self {
        match error {
            ProviderError::Local(error) => FailureType::of_api(error),
            ProviderError::Host { api_kind: Some(kind), .. } => FailureType::of_kind(*kind),
            ProviderError::Host { api_kind: None, .. } => FailureType::Exception,
        }
    }

    fn of_kind(kind: ApiErrorKind) -> Self {
        match kind {
            ApiErrorKind::Api => FailureType::ApiError,
            ApiErrorKind::Status => FailureType::ApiStatusError,
            ApiErrorKind::Connection => FailureType::ApiConnectionError,
            ApiErrorKind::Timeout => FailureType::Timeout,
            ApiErrorKind::Cancelled => FailureType::Cancelled,
        }
    }
}

/// Provider requests, retries excluded.
#[metric(name = "lk.productstats.provider.requests", kind = Counter, unit = "{request}")]
#[derive(Attrs)]
pub struct ProviderRequests {
    #[attrs]
    pub provider: ProviderLabels,
}

/// Provider requests that failed, by the class of their last failure.
#[metric(name = "lk.productstats.provider.failures", kind = Counter, unit = "{request}")]
#[derive(Attrs)]
pub struct ProviderFailures {
    #[attrs]
    pub provider: ProviderLabels,
    /// The class of the failure.
    #[attr("lk.productstats.error.type")]
    pub error_type: FailureType,
}

/// Provider attempts retried after a failure.
#[metric(name = "lk.productstats.provider.retries", kind = Counter, unit = "{attempt}")]
#[derive(Attrs)]
pub struct ProviderRetries {
    #[attrs]
    pub provider: ProviderLabels,
}

/// Fallback adapters moving off this provider to the next.
#[metric(name = "lk.productstats.provider.fallbacks", kind = Counter, unit = "{fallback}")]
#[derive(Attrs)]
pub struct ProviderFallbacks {
    #[attrs]
    pub provider: ProviderLabels,
}

/// Provider sessions reconnected.
#[metric(name = "lk.productstats.provider.reconnects", kind = Counter, unit = "{reconnect}")]
#[derive(Attrs)]
pub struct ProviderReconnects {
    #[attrs]
    pub provider: ProviderLabels,
}

/// Time to a TTS request's first audio.
#[metric(name = "lk.productstats.provider.ttfb", kind = Histogram, unit = "s", bounds = DURATION_BOUNDS)]
#[derive(Attrs)]
pub struct ProviderTtfb {
    #[attrs]
    pub provider: ProviderLabels,
}

/// Time to an LLM or realtime response's first token.
#[metric(name = "lk.productstats.provider.ttft", kind = Histogram, unit = "s", bounds = DURATION_BOUNDS)]
#[derive(Attrs)]
pub struct ProviderTtft {
    #[attrs]
    pub provider: ProviderLabels,
}

/// Duration of a provider request.
#[metric(
    name = "lk.productstats.provider.duration",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct ProviderDuration {
    #[attrs]
    pub provider: ProviderLabels,
}

/// How a job ended.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobOutcome {
    Ok,
    Failed,
    Cancelled,
}

/// Jobs started.
#[metric(name = "lk.productstats.job.starts", kind = Counter, unit = "{job}")]
#[derive(Attrs)]
pub struct JobStarts;

/// Jobs ended, by outcome.
#[metric(name = "lk.productstats.job.ends", kind = Counter, unit = "{job}")]
#[derive(Attrs)]
pub struct JobEnds {
    /// How the job ended.
    #[attr("lk.productstats.job.outcome")]
    pub outcome: JobOutcome,
}

/// How a job process died unexpectedly.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashSignature {
    /// It exited with a failure code of its own.
    Exited,
    /// A signal killed it.
    Signaled,
    /// Its exit could not be read.
    Lost,
}

/// Job processes that died unexpectedly.
#[metric(name = "lk.productstats.process.crashes", kind = Counter, unit = "{process}")]
#[derive(Attrs)]
pub struct ProcessCrashes {
    /// How it died.
    #[attr("lk.productstats.crash.signature")]
    pub signature: CrashSignature,
}

/// The frames of a stall's signature, innermost first.
pub const STALL_SIGNATURE_FRAMES: usize = 3;

/// Event-loop stalls, by the function names of their innermost frames.
#[metric(name = "lk.productstats.event_loop.stalls", kind = Counter, unit = "{stall}")]
#[derive(Attrs)]
pub struct EventLoopStalls {
    /// Up to [`STALL_SIGNATURE_FRAMES`] function names, innermost first, joined by ` < `;
    /// `unknown` when the stack could not be read. No file, line or argument.
    #[attr("lk.productstats.stall.signature")]
    pub signature: Label,
}

/// Why product statistics were dropped.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropReason {
    /// A window waited for a credential and was replaced by the next.
    NoCredential,
    /// The collector refused the export, or it failed in transport.
    Refused,
    /// The aggregation's series limit was reached.
    Queue,
}

/// Data points this process dropped, exported on the next export.
#[metric(name = "lk.productstats.export.dropped", kind = Counter, unit = "{point}")]
#[derive(Attrs)]
pub struct ExportDropped {
    /// Why they were dropped.
    #[attr("lk.productstats.drop.reason")]
    pub reason: DropReason,
}

/// A plugin a host loaded, as the resource lists it.
pub fn plugin_label(package: &str, version: &str) -> Label {
    Label(format!("{package}=={version}"))
}

/// What the host tells product statistics.
#[crate::host(functions(pid = false))]
pub trait ProductStatsHost {
    /// A plugin registered on import: a `livekit.plugins.*` package is listed on the resource of
    /// every later export; another package is not.
    #[host(function)]
    fn _core_product_stats_plugin(package: String, version: String);
}
