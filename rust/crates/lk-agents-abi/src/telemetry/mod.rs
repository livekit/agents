//! Telemetry contracts: the span, metric and report record declarations, the policy limits and the
//! session report.

/// Telemetry limits; the decisions are `lk_agents_core::telemetry::policy`'s.
pub mod policy {
    /// Nested host values deeper than this are refused.
    #[crate::host(export(name = "TELEMETRY_MAX_DEPTH"))]
    pub const MAX_DEPTH: usize = 4;

    /// The bound on a job's final telemetry flush.
    pub const SHUTDOWN_TIMEOUT_SECONDS: f64 = 10.0;

    pub const UPLOAD_TIMEOUT_SECONDS: u64 = 900;

    /// What a redacted destination sees in place of an exception message or error status.
    pub const REDACTED_EXCEPTION: &str = "exception details redacted";
}

pub mod decl;
pub mod errors;
pub mod host;
pub mod instruments;
pub mod record;
pub mod report;

/// Keys and names owned by the registries the agents registry depends on. A declaration names
/// these constants; it never redeclares their keys as literals (`lk-agents-codegen --check`).
///
/// Hand-maintained mirror of the pinned registries (the pins are in `lk-agents-codegen`'s
/// telemetry module). Once livekit/protocol publishes `otel/model/shared` with generated Rust
/// constants, `protocol` becomes that generated file, and `otel` a weaver template over the
/// pinned OpenTelemetry and GenAI registries; the declarations do not change.
pub mod semconv {
    /// livekit/protocol `otel/model/shared`: correlation keys shared with the client SDKs.
    pub mod protocol {
        pub const LK_ROOM_ID: &str = "lk.room_id";
        pub const LK_JOB_ID: &str = "lk.job_id";
        pub const LK_PARTICIPANT_ID: &str = "lk.participant_id";
        pub const LK_TRACK_ID: &str = "lk.track_id";
        pub const LK_PII_ROOM_NAME: &str = "lk.pii.room_name";
        pub const LK_PII_PARTICIPANT_IDENTITY: &str = "lk.pii.participant_identity";
        pub const LK_PII_REMOTE_PARTICIPANT_IDENTITY: &str = "lk.pii.remote_participant_identity";
        pub const LK_SDK_NAME: &str = "lk.sdk.name";
        pub const LK_SDK_VERSION: &str = "lk.sdk.version";
        pub const LK_TRACK_SOURCE: &str = "lk.track_source";
        pub const LK_DISCONNECT_REASON: &str = "lk.disconnect_reason";
        pub const LK_AGENT_NAME: &str = "lk.agent_name";
    }

    /// OpenTelemetry semantic conventions v1.44.0 and the GenAI registry.
    pub mod otel {
        pub const ERROR_TYPE: &str = "error.type";
        pub const SERVICE_NAME: &str = "service.name";
        pub const EXCEPTION_TYPE: &str = "exception.type";
        pub const EXCEPTION_MESSAGE: &str = "exception.message";
        pub const EXCEPTION_STACKTRACE: &str = "exception.stacktrace";
        pub const RPC_METHOD: &str = "rpc.method";

        pub const GEN_AI_AGENT_NAME: &str = "gen_ai.agent.name";
        pub const GEN_AI_CONVERSATION_ID: &str = "gen_ai.conversation.id";
        pub const GEN_AI_EVALUATION_EXPLANATION: &str = "gen_ai.evaluation.explanation";
        pub const GEN_AI_EVALUATION_NAME: &str = "gen_ai.evaluation.name";
        pub const GEN_AI_EVALUATION_SCORE_LABEL: &str = "gen_ai.evaluation.score.label";
        pub const GEN_AI_INPUT_MESSAGES: &str = "gen_ai.input.messages";
        pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
        pub const GEN_AI_OUTPUT_MESSAGES: &str = "gen_ai.output.messages";
        pub const GEN_AI_OUTPUT_TYPE: &str = "gen_ai.output.type";
        pub const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";
        pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
        pub const GEN_AI_REQUEST_STREAM: &str = "gen_ai.request.stream";
        pub const GEN_AI_RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
        pub const GEN_AI_RESPONSE_ID: &str = "gen_ai.response.id";
        pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";
        pub const GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK: &str = "gen_ai.response.time_to_first_chunk";
        pub const GEN_AI_SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";
        pub const GEN_AI_TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
        pub const GEN_AI_TOOL_CALL_ID: &str = "gen_ai.tool.call.id";
        pub const GEN_AI_TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";
        pub const GEN_AI_TOOL_DEFINITIONS: &str = "gen_ai.tool.definitions";
        pub const GEN_AI_TOOL_DESCRIPTION: &str = "gen_ai.tool.description";
        pub const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";
        pub const GEN_AI_TOOL_TYPE: &str = "gen_ai.tool.type";
        pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
        pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
        pub const GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: &str =
            "gen_ai.usage.cache_read.input_tokens";
        pub const GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: &str =
            "gen_ai.usage.cache_write.input_tokens";
        pub const GEN_AI_USAGE_REASONING_OUTPUT_TOKENS: &str =
            "gen_ai.usage.reasoning.output_tokens";
        pub const GEN_AI_USAGE_TEXT_INPUT_TOKENS: &str = "gen_ai.usage.text.input_tokens";
        pub const GEN_AI_USAGE_AUDIO_INPUT_TOKENS: &str = "gen_ai.usage.audio.input_tokens";
        pub const GEN_AI_USAGE_IMAGE_INPUT_TOKENS: &str = "gen_ai.usage.image.input_tokens";
        pub const GEN_AI_USAGE_TEXT_OUTPUT_TOKENS: &str = "gen_ai.usage.text.output_tokens";
        pub const GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS: &str = "gen_ai.usage.audio.output_tokens";
        pub const GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS: &str =
            "gen_ai.usage.text.cache_read.input_tokens";
        pub const GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS: &str =
            "gen_ai.usage.audio.cache_read.input_tokens";
        pub const GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS: &str =
            "gen_ai.usage.image.cache_read.input_tokens";
        pub const GEN_AI_WORKFLOW_NAME: &str = "gen_ai.workflow.name";
        /// `input` or `output` on `gen_ai.client.token.usage`. Deprecated upstream; as emitted.
        pub const GEN_AI_TOKEN_TYPE: &str = "gen_ai.token.type";

        /// Metrics imported from the GenAI registry.
        pub const GEN_AI_EXECUTE_TOOL_DURATION: &str = "gen_ai.execute_tool.duration";
        pub const GEN_AI_INVOKE_AGENT_DURATION: &str = "gen_ai.invoke_agent.duration";
    }
}

pub mod spans;

use std::sync::Arc;

use decl::SpanDecl;
use record::{JobKey, Record, SpanContext};

/// What records a context's spans: the process tracer (`lk_agents_net::telemetry::Tracer`), or
/// the DST's.
pub trait Recorder: Send + Sync {
    fn record(&self, record: Record);
    /// Wall-clock unix nanoseconds from the injected clock.
    fn now_nanos(&self) -> u64;
    /// A new span or trace id half, never zero.
    fn next_id(&self) -> u64;
    /// GenAI message capture: whether request and response messages are recorded on spans.
    fn capture_content(&self) -> bool;
}

/// The span work runs under, explicitly passed (`telemetry.md`, "Context across boundaries"):
/// nothing is entered into a thread-local, so the context stays correct across `.await` and
/// concurrent tasks. Every host call takes one; [`Cx::NONE`] records nothing and makes a call run
/// under no span.
#[derive(Clone)]
pub struct Cx {
    recorder: Option<Arc<dyn Recorder>>,
    current: Option<Current>,
    job: Option<JobKey>,
    conversation: Option<Arc<str>>,
}

#[derive(Clone, Copy)]
struct Current {
    span: SpanContext,
    /// The declared span's name; `None` for a span from outside the framework (an application
    /// span, a `traceparent`).
    name: Option<&'static str>,
}

impl Cx {
    /// No span: a host call made under it runs with no current span, and spans opened under it
    /// record nothing.
    pub const NONE: Cx = Cx { recorder: None, current: None, job: None, conversation: None };

    /// A context with no current span, outside any job, recording to `recorder`.
    pub fn root(recorder: Arc<dyn Recorder>) -> Cx {
        Cx { recorder: Some(recorder), ..Cx::NONE }
    }

    /// What records this context's spans; `None` for [`Cx::NONE`].
    pub fn recorder(&self) -> Option<&Arc<dyn Recorder>> {
        self.recorder.as_ref()
    }

    /// Wall-clock unix nanoseconds; 0 for a context that records nothing.
    pub fn now_nanos(&self) -> u64 {
        self.recorder.as_ref().map_or(0, |r| r.now_nanos())
    }

    /// Whether GenAI message content is recorded.
    pub fn capture_content(&self) -> bool {
        self.recorder.as_ref().is_some_and(|r| r.capture_content())
    }

    /// The current span, what a host call activates and a `traceparent` carries.
    pub fn span(&self) -> Option<SpanContext> {
        self.current.map(|c| c.span)
    }

    /// The current span's declared name, for the parent check; `None` for no span or one from
    /// outside the framework.
    pub fn current_name(&self) -> Option<&'static str> {
        self.current.and_then(|c| c.name)
    }

    pub fn job(&self) -> Option<&JobKey> {
        self.job.as_ref()
    }

    pub fn conversation(&self) -> Option<&str> {
        self.conversation.as_deref()
    }

    /// The same context, recording for `job`.
    pub fn with_job(mut self, job: Option<JobKey>) -> Cx {
        self.job = job;
        self
    }

    /// The same context, its spans carrying `gen_ai.conversation.id`.
    pub fn with_conversation(mut self, conversation: Option<&str>) -> Cx {
        self.conversation = conversation.map(Arc::from);
        self
    }

    /// The same context under `span`, a framework span of type `P` recorded elsewhere (a voice
    /// effect's `SpanContext`).
    pub fn under<P: SpanDecl>(mut self, span: SpanContext) -> Cx {
        self.current = Some(Current { span, name: Some(P::NAME) });
        self
    }

    /// The same context under a span from outside the framework: an application span captured
    /// at a host entry, or a `traceparent`. `None` starts new traces.
    pub fn continuing(mut self, span: Option<SpanContext>) -> Cx {
        self.current = span.map(|span| Current { span, name: None });
        self
    }

    /// A new span's context under the current span: its trace, or a new one.
    pub fn child_context(&self) -> SpanContext {
        let next = || self.recorder.as_ref().map_or(1, |r| r.next_id());
        let trace_id = match self.current {
            Some(c) => c.span.trace_id,
            None => (u128::from(next()) << 64) | u128::from(next()),
        };
        SpanContext {
            trace_id,
            span_id: next(),
            sampled: self.current.is_none_or(|c| c.span.sampled),
        }
    }
}
