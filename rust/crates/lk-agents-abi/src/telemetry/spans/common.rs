//! Attributes no single span declares: the job identity the exporter stamps, the keys every span
//! carries, the participant group and the failure event.

use crate::telemetry::decl::{Attrs, Content, SpanValue, event};
use crate::telemetry::semconv::{otel, protocol};

/// The job's identity, stamped by the exporter on every span, log record and Cloud measurement of
/// the job. Never set by a declaration.
#[derive(Attrs)]
pub struct JobIdentity {
    #[attr(protocol::LK_ROOM_ID)]
    pub room_id: String,
    /// The room session id under the spelling Cloud's ingest and session view read today.
    #[attr("room_id")]
    pub legacy_room_id: String,
    #[attr(protocol::LK_JOB_ID)]
    pub job_id: String,
    /// The job id under the spelling Cloud's consumers read today.
    #[attr("job_id")]
    pub legacy_job_id: String,
    #[attr(protocol::LK_PII_ROOM_NAME)]
    pub room_name: Content<String>,
    #[attr(protocol::LK_AGENT_NAME)]
    pub agent_name: String,
    /// LiveKit Cloud agent id.
    #[attr("lk.cloud_agent_id")]
    pub cloud_agent_id: Option<String>,
    /// LiveKit Cloud deployment id.
    #[attr("lk.deployment_id")]
    pub deployment_id: Option<String>,
    /// PII redaction is on for the session; a native boolean on every record.
    #[attr("lk.redaction.enabled")]
    pub redaction: bool,
    /// The session is a simulation; a native boolean on every record.
    #[attr("lk.simulation.enabled")]
    pub simulation: bool,
    /// Simulation run the session belongs to.
    #[attr("lk.simulation.run_id")]
    pub simulation_run_id: Option<String>,
    /// Simulation run job (one scenario) behind the session.
    #[attr("lk.simulation.job_id")]
    pub simulation_job_id: Option<String>,
}

/// The SDK that produced the telemetry, on the resource.
#[derive(Attrs)]
pub struct SdkResource {
    #[attr(protocol::LK_SDK_NAME)]
    pub name: String,
    #[attr(protocol::LK_SDK_VERSION)]
    pub version: String,
}

/// Keys every span carries, written by the span handles rather than a declaration.
#[derive(Attrs)]
pub struct EverySpan {
    /// The room session id in a job, the thread id (`AT_...`) in a text run.
    #[attr(otel::GEN_AI_CONVERSATION_ID)]
    pub conversation_id: Option<String>,
    /// On a recorded failure.
    #[attr(otel::ERROR_TYPE)]
    pub error_type: Option<String>,
    /// The span was dropped without `end`.
    #[attr("lk.cancelled")]
    pub cancelled: Option<bool>,
}

/// A recorded failure, as the OpenTelemetry API's `record_exception` writes it, on any span.
#[event(name = "exception")]
#[derive(Attrs)]
pub struct Exception {
    #[attr(otel::EXCEPTION_TYPE)]
    pub kind: String,
    #[attr(otel::EXCEPTION_MESSAGE)]
    pub message: String,
    #[attr(otel::EXCEPTION_STACKTRACE)]
    pub stacktrace: Option<String>,
}

/// The participant a span is about.
#[derive(Attrs)]
pub struct ParticipantAttrs {
    #[attr(protocol::LK_PARTICIPANT_ID)]
    pub id: String,
    #[attr(protocol::LK_PII_PARTICIPANT_IDENTITY)]
    pub identity: Content<String>,
    /// Kind of the participant, as the protocol enum name (`PARTICIPANT_KIND_*`).
    #[attr("lk.participant_kind")]
    pub kind: String,
}

/// The GenAI operation a span performs.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenAiOperation {
    Chat,
    GenerateContent,
    ExecuteTool,
    InvokeAgent,
    InvokeWorkflow,
    CreateAgent,
}
