//! Telemetry contracts: the span, metric and report record declarations, the wire attribute names,
//! the policy and provider metric declarations, the session report's records.

/// Telemetry limits and the host value a session report's event retention charges; the decisions
/// are `lk_agents_core::telemetry::policy`'s.
pub mod policy {
    /// Nested host values deeper than this are refused.
    #[crate::host(export(name = "TELEMETRY_MAX_DEPTH"))]
    pub const MAX_DEPTH: usize = 4;

    /// Host mappings and sequences longer than this are refused.
    pub const MAX_COLLECTION: usize = 128;

    /// The bound on a job's final telemetry flush.
    pub const SHUTDOWN_TIMEOUT_SECONDS: f64 = 10.0;

    pub const UPLOAD_TIMEOUT_SECONDS: u64 = 900;

    /// What a redacted destination sees in place of an exception message or error status.
    pub const REDACTED_EXCEPTION: &str = "exception details redacted";

    pub const REPORT_EVENT_ITEMS: usize = 1024;
    pub const REPORT_EVENT_BYTES: usize = 1024 * 1024;
    /// Retained bytes one event may charge.
    pub const REPORT_EVENT_RECORD_BYTES: usize = 64 * 1024;

    /// A host value as OpenTelemetry's `AnyValue`, read once from the host object: a mapping's
    /// non-string keys are skipped, a host record (a pydantic model, a dataclass) reads as the
    /// mapping of its fields.
    #[crate::host(codec(extract, emit, host = "object", py = "py_values::telemetry_value"))]
    #[derive(Debug, Clone, PartialEq)]
    pub enum TelemetryValue {
        None,
        Bool(bool),
        Int(i64),
        Float(f64),
        String(String),
        Bytes(Vec<u8>),
        Array(Vec<TelemetryValue>),
        Map(Vec<(String, TelemetryValue)>),
        /// What OTel cannot carry: another host object, an integer beyond 64 bits, or a collection
        /// nested deeper than [`MAX_DEPTH`] or longer than [`MAX_COLLECTION`], which is not walked.
        /// Retention refuses it.
        Unsupported,
    }
}

/// The provider metric folds the host's streams drive; the implementation is `lk-agents-core`'s.
pub mod provider {
    use crate::metrics::base::{LlmMetrics, SttMetrics, TtsMetrics};
    use crate::{LlmChunk, SpeechEvent, SynthesizedAudio};

    /// The host's context tuple, `(timestamp, label, cancelled, model, provider)`, as the folds'.
    pub type HostContext = (f64, String, bool, String, String);

    /// A `RECOGNITION_USAGE` event without its usage.
    #[crate::host(error(kind = "AssertionError"))]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
    #[error("recognition_usage must be provided for RECOGNITION_USAGE event")]
    pub struct RecognitionUsageMissing;

    #[crate::host(functions(pid = false))]
    pub trait ProviderMetrics {
        /// `STT.recognize`'s metrics for one batch recognition.
        #[host(function)]
        fn stt_batch_metrics(
            event: SpeechEvent,
            start: f64,
            now: f64,
            audio_durations: Vec<f64>,
            host: HostContext,
        ) -> SttMetrics;

        /// `RecognizeStream`'s metrics for one event, and whether it resets the retry count.
        #[host(function)]
        fn stt_stream_metrics(
            event: SpeechEvent,
            host: HostContext,
        ) -> Result<(Option<SttMetrics>, bool), RecognitionUsageMissing>;
    }

    /// `LLMStream`'s metric fold as the host drives it.
    #[crate::host(class(name = "LlmMetricState", pid = false))]
    pub trait LlmMetricState: Send + Sized + 'static {
        #[host(new)]
        fn new(start: f64) -> Self;

        /// Whether `chunk` is the first with a response.
        #[host(call)]
        fn observe(&mut self, now: f64, chunk: LlmChunk) -> bool;

        /// `None` for a failed attempt or one that yielded no chunk.
        #[host(call)]
        fn finish(&self, now: f64, failed: bool, host: HostContext) -> Option<LlmMetrics>;
    }

    /// The TTS streams' metric fold as the host drives it.
    #[crate::host(class(name = "TtsMetricState", pid = false))]
    pub trait TtsMetricState: Send + Sized + 'static {
        #[host(new)]
        fn new(streamed: bool) -> Self;

        /// A streamed segment's text, in the order its audio is finished.
        #[host(call)]
        fn queue_text(&mut self, text: String);

        /// Whether `audio` ends its segment.
        #[host(call)]
        fn observe(&mut self, now: f64, start: f64, audio: SynthesizedAudio) -> bool;

        /// The record of a finished request or segment, then the figures reset.
        #[host(call)]
        fn finish(
            &mut self,
            now: f64,
            start: f64,
            failed: bool,
            batch_text: String,
            host: HostContext,
        ) -> Option<TtsMetrics>;
    }
}

pub mod attributes;
pub mod decl;
pub mod host;
pub mod instruments;
pub mod record;
pub mod report;

/// The session report's records, exported to Cloud as OTLP logs.
pub mod report_records {
    use super::decl::{Attrs, JsonObject, KvList, SpanRef, record};
    use crate::usage::ModelUsage;

    /// One chat history item, linked to the span of the turn it belongs to.
    #[record(scope = "chat_history", body = "chat item")]
    #[derive(Attrs)]
    pub struct ChatItemRecord {
        /// One of `message`, `function_call`, `function_call_output`, `agent_handoff`,
        /// `agent_config_update`, in the `livekit.agent.ChatContext.ChatItem` protobuf JSON form.
        #[attr("chat.item")]
        pub item: KvList<JsonObject>,
        #[span_id]
        pub turn: Option<SpanRef>,
    }

    /// The session summary, timestamped at the session start.
    #[record(scope = "chat_history", body = "session report")]
    #[derive(Attrs)]
    pub struct SessionReportRecord {
        /// The resolved session options, recursively serialized; `instructions` omitted, `keyterms`
        /// as `lk.pii.keyterms`.
        #[attr("session.options")]
        pub options: KvList<JsonObject>,
        /// When the report was made, in unix seconds.
        #[attr("session.report_timestamp")]
        pub report_timestamp: f64,
        /// The session's tags, sorted; absent when there are none.
        #[attr("session.tags")]
        pub tags: Option<Vec<String>>,
        /// The agent's name.
        #[attr("agent_name")]
        pub agent_name: String,
        /// The framework's version.
        #[attr("sdk_version")]
        pub sdk_version: String,
        /// The `Usage` model's entries: one per model and kind; absent when nothing was used.
        #[attr("usage")]
        pub usage: Option<Vec<KvList<ModelUsage>>>,
    }

    /// One judge's evaluation.
    #[record(scope = "evaluations", body = "evaluation")]
    #[derive(Attrs)]
    pub struct EvaluationRecord {
        /// `name`, `tag` (`lk.judge.<name>:<verdict>`), `verdict`, `reasoning`, `instructions`.
        #[attr("evaluation")]
        pub evaluation: KvList<JsonObject>,
    }

    /// One tag the application added.
    #[record(scope = "evaluations", body = "tag")]
    #[derive(Attrs)]
    pub struct TagRecord {
        /// `name`, `metadata`.
        #[attr("tag")]
        pub tag: KvList<JsonObject>,
    }

    /// The session's outcome.
    #[record(scope = "evaluations", body = "outcome")]
    #[derive(Attrs)]
    pub struct OutcomeRecord {
        /// `outcome` (`success`, `fail`), `reason`.
        #[attr("outcome")]
        pub outcome: KvList<JsonObject>,
    }
}

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

        /// Metrics imported from the GenAI registry.
        pub const GEN_AI_EXECUTE_TOOL_DURATION: &str = "gen_ai.execute_tool.duration";
        pub const GEN_AI_INVOKE_AGENT_DURATION: &str = "gen_ai.invoke_agent.duration";
    }
}

pub mod spans;

use std::sync::Arc;

use decl::SpanDecl;
use record::{JobKey, Record, SpanId, SpanRef, TraceContext, TraceId};

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
    span: SpanRef,
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

    /// The caller's span at a host-to-framework entry (the span current in the caller's
    /// `contextvars`), recording to `recorder`.
    pub fn entry(recorder: Arc<dyn Recorder>, caller: Option<SpanRef>) -> Cx {
        Cx::root(recorder).continuing(caller)
    }

    /// The context a [`TraceContext`] carries across an API that cannot carry a `Cx`.
    pub fn resume(recorder: Arc<dyn Recorder>, trace: &TraceContext) -> Cx {
        Cx::root(recorder)
            .continuing(trace.span)
            .with_job(trace.job.clone())
            .with_conversation(trace.conversation.as_deref())
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
    pub fn span(&self) -> Option<SpanRef> {
        self.current.map(|c| c.span)
    }

    /// The current span's declared name, for the parent check; `Some(None)` for a span from
    /// outside the framework.
    pub fn current_name(&self) -> Option<Option<&'static str>> {
        self.current.map(|c| c.name)
    }

    pub fn job(&self) -> Option<&JobKey> {
        self.job.as_ref()
    }

    pub fn conversation(&self) -> Option<&str> {
        self.conversation.as_deref()
    }

    /// This context for an API that cannot carry a `Cx`; [`Cx::resume`] reopens it.
    pub fn trace_context(&self) -> TraceContext {
        TraceContext {
            span: self.span(),
            job: self.job.clone(),
            conversation: self.conversation.as_deref().map(str::to_owned),
        }
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
    /// effect's `SpanRef`).
    pub fn under<P: SpanDecl>(mut self, span: SpanRef) -> Cx {
        self.current = Some(Current { span, name: Some(P::NAME) });
        self
    }

    /// The same context under a span from outside the framework: an application span captured
    /// at a host entry, or a `traceparent`. `None` starts new traces.
    pub fn continuing(mut self, span: Option<SpanRef>) -> Cx {
        self.current = span.map(|span| Current { span, name: None });
        self
    }

    /// A new span's context under the current span: its trace, or a new one.
    pub fn child_context(&self) -> SpanRef {
        let next = || self.recorder.as_ref().map_or(1, |r| r.next_id());
        let trace_id = match self.current {
            Some(c) => c.span.trace_id,
            None => TraceId((u128::from(next()) << 64) | u128::from(next())),
        };
        SpanRef {
            trace_id,
            span_id: SpanId(next()),
            sampled: self.current.is_none_or(|c| c.span.sampled),
        }
    }
}
