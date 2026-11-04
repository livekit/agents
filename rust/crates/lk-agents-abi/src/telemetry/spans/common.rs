//! Attributes no single span declares: the job identity the exporter stamps, the keys every span
//! carries, the participant group and the failure event.

use crate::telemetry::decl::{Attrs, Content, SpanValue, event};
use crate::telemetry::errors::ErrorType;
use crate::telemetry::semconv::{otel, protocol};
use crate::types::attributes;
use crate::voice::rtc;

/// The job's identity, stamped by the exporter on every span, log record and Cloud measurement of
/// the job (and, without the room name, on the Cloud resource). Never set by a declaration; an
/// empty value is omitted.
#[derive(Attrs)]
pub struct JobIdentity {
    #[attr(protocol::LK_ROOM_ID)]
    pub lk_room_id: String,
    /// The room session id: Cloud's ingest, session view and Agent Insights find a job by it.
    #[attr("room_id")]
    pub room_id: String,
    #[attr(protocol::LK_JOB_ID)]
    pub lk_job_id: String,
    /// The job id: Cloud's consumers find a job's records by it.
    #[attr("job_id")]
    pub job_id: String,
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
}

/// The session's recording flags, stamped by the Cloud destination next to the job's identity on
/// every span and log record: Cloud's ingest decides redaction per record from them.
#[derive(Attrs)]
pub struct SessionFlags {
    /// PII redaction is on for the session; a native boolean on every record.
    #[attr(attributes::ATTRIBUTE_REDACTION_ENABLED)]
    pub redaction: bool,
    /// The session is a simulation; a native boolean on every record.
    #[attr(attributes::ATTRIBUTE_SIMULATION_ENABLED)]
    pub simulation: bool,
    /// Simulation run the session belongs to.
    #[attr(attributes::ATTRIBUTE_SIMULATION_RUN_ID)]
    pub simulation_run_id: Option<String>,
    /// Simulation run job (one scenario) behind the session.
    #[attr(attributes::ATTRIBUTE_SIMULATION_JOB_ID)]
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
    /// The conversation's thread id (`AT_...`); for a call without a thread, the room session id.
    #[attr(otel::GEN_AI_CONVERSATION_ID)]
    pub conversation_id: Option<String>,
    /// On a recorded failure.
    #[attr(otel::ERROR_TYPE)]
    pub error_type: Option<ErrorType>,
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
    /// Omitted when the room has not described the participant.
    #[attr(protocol::LK_PARTICIPANT_ID)]
    pub id: Option<String>,
    #[attr(protocol::LK_PII_PARTICIPANT_IDENTITY)]
    pub identity: Content<String>,
    /// Kind of the participant; omitted when the room has not described the participant.
    #[attr("lk.participant_kind")]
    pub kind: Option<ParticipantKind>,
}

/// A participant's kind, as the protocol enum names it.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticipantKind {
    #[wire = "PARTICIPANT_KIND_STANDARD"]
    Standard,
    #[wire = "PARTICIPANT_KIND_INGRESS"]
    Ingress,
    #[wire = "PARTICIPANT_KIND_EGRESS"]
    Egress,
    #[wire = "PARTICIPANT_KIND_SIP"]
    Sip,
    #[wire = "PARTICIPANT_KIND_AGENT"]
    Agent,
    #[wire = "PARTICIPANT_KIND_CONNECTOR"]
    Connector,
    #[wire = "PARTICIPANT_KIND_BRIDGE"]
    Bridge,
}

impl From<rtc::ParticipantKind> for ParticipantKind {
    fn from(kind: rtc::ParticipantKind) -> Self {
        match kind {
            rtc::ParticipantKind::Standard => Self::Standard,
            rtc::ParticipantKind::Ingress => Self::Ingress,
            rtc::ParticipantKind::Egress => Self::Egress,
            rtc::ParticipantKind::Sip => Self::Sip,
            rtc::ParticipantKind::Agent => Self::Agent,
            rtc::ParticipantKind::Connector => Self::Connector,
            rtc::ParticipantKind::Bridge => Self::Bridge,
        }
    }
}

/// `gen_ai.output.type`: what a model request produces.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    Text,
    Speech,
}

/// `gen_ai.tool.type`.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolType {
    Function,
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

/// Every key declared `GenAiContent`, at any depth of a record (`lk-agents-codegen --check`
/// rejects a list that differs). Cloud's ingest does not recognize them, so the Cloud destination
/// removes them itself for a redacted session.
pub const GENAI_CONTENT: &[&str] = &[
    otel::GEN_AI_INPUT_MESSAGES,
    otel::GEN_AI_OUTPUT_MESSAGES,
    otel::GEN_AI_SYSTEM_INSTRUCTIONS,
    otel::GEN_AI_TOOL_DEFINITIONS,
    otel::GEN_AI_TOOL_CALL_ARGUMENTS,
    otel::GEN_AI_TOOL_CALL_RESULT,
    otel::GEN_AI_TOOL_DESCRIPTION,
    otel::GEN_AI_EVALUATION_EXPLANATION,
];

/// Content keys of process log lines and session diagnostics. Logs never reach OTLP; the hosted
/// log sidecar redacts them by their `lk.pii.` prefix.
pub mod log_keys {
    pub const USER_INPUT: &str = "lk.pii.user_input";
    pub const RAW_ARGUMENTS: &str = "lk.pii.raw_arguments";
    pub const REPAIRED: &str = "lk.pii.repaired";
    pub const ERROR: &str = "lk.pii.error";
}
