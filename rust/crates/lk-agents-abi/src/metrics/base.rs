//! `metrics/base.py`. A `///` on a declared type or field is API (it lands in the schema digest the
//! gate compares); Rust-only notes go in `//`. Request metrics embed `usage`'s `TokenUsage`.

use crate::usage::TokenUsage;

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "metrics", path = "livekit.agents.metrics"))]
const _: () = ();

// `metrics/base.py:8`. `Some("")` and `None` differ under `model_dump(exclude_none=True)`, so the
// fields stay `Option`.
#[crate::host(record(module = "metrics", pydantic, emit, extract))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Metadata {
    #[host(default = None)]
    pub model_name: Option<String>,
    #[host(default = None)]
    pub model_provider: Option<String>,
}

// `metrics/base.py:13`. Generated because the gate records `bases`.
#[crate::host(record(module = "metrics", name = "_BaseMetrics", pydantic, repr = "metrics_repr"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BaseMetrics {}

// `metrics/base.py:188`. The counters are per event, not cumulative: totals are the collector's.
#[crate::host(record(
    module = "metrics",
    pydantic,
    base = "_BaseMetrics",
    constant(type = "interruption_metrics"),
    emit,
    extract,
))]
#[derive(Debug, Clone, PartialEq)]
pub struct InterruptionMetrics {
    // A `float` on the model, so not a `SystemTime`.
    pub timestamp: f64,
    /// Latest RTT (Round Trip Time) time taken to perform the inference, in seconds.
    pub total_duration: f64,
    /// Latest time taken to perform the inference from the model side, in seconds.
    pub prediction_duration: f64,
    /// Latest total time from the onset of the speech to the final prediction, in seconds.
    pub detection_delay: f64,
    /// Number of interruptions detected, incrementally counted.
    pub num_interruptions: u32,
    /// Number of backchannels detected, incrementally counted.
    pub num_backchannels: u32,
    /// Number of requests sent to the interruption detection model, incrementally counted.
    pub num_requests: u32,
    #[host(default = None)]
    pub metadata: Option<Metadata>,
}

impl InterruptionMetrics {
    pub const TYPE: &'static str = "interruption_metrics";
}

// `metrics/base.py:121`, emitted per `eot_prediction` (`inference/eot/transports.py:234`).
// `total_duration` (server's `client_e2e_latency`) and `detection_delay` (local) are not synonyms.
/// Per-inference metrics emitted by the EOT model on each prediction.
#[crate::host(record(
    module = "metrics",
    name = "EOTInferenceMetrics",
    pydantic,
    base = "_BaseMetrics",
    constant(type = "eot_inference_metrics"),
    emit,
    extract,
))]
#[derive(Debug, Clone, PartialEq)]
pub struct EotInferenceMetrics {
    pub timestamp: f64,
    /// Earliest audio creation time in an inference to response receive time.
    pub total_duration: f64,
    /// Latest audio creation time in an inference to response receive time.
    pub detection_delay: f64,
    /// Server side model inference time.
    pub prediction_duration: f64,
    /// Number of inference requests made during one inference.
    #[host(default = 1)]
    pub num_requests: u32,
    #[host(default = None)]
    pub metadata: Option<Metadata>,
}

impl EotInferenceMetrics {
    pub const TYPE: &'static str = "eot_inference_metrics";
    /// The model's only defaulted field (so no `Default` impl).
    pub const DEFAULT_NUM_REQUESTS: u32 = 1;
}

// `metrics/base.py:19`.
// Both directions on every record below: read by the collectors, emitted as the dict the pydantic
// `AgentMetrics` union validates. `type` comes from `tag_const`.
#[crate::host(record(
    module = "metrics",
    name = "LLMMetrics",
    pydantic,
    base = "_BaseMetrics",
    constant(type = "llm_metrics"),
    emit,
    extract,
))]
#[derive(Debug, Clone, PartialEq)]
pub struct LlmMetrics {
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    pub duration: f64,
    /// Time to first generated token in seconds; ``None`` if the response generated none.
    #[host(default = None)]
    pub ttft: Option<f64>,
    pub cancelled: bool,
    #[host(default_factory = "TokenUsage()")]
    pub tokens: TokenUsage,
    pub tokens_per_second: f64,
    #[host(default = None)]
    pub speech_id: Option<String>,
    #[host(default = None)]
    pub metadata: Option<Metadata>,
}

impl LlmMetrics {
    pub const TYPE: &'static str = "llm_metrics";
}

// `metrics/base.py:44`.
#[crate::host(record(
    module = "metrics",
    name = "STTMetrics",
    pydantic,
    base = "_BaseMetrics",
    constant(type = "stt_metrics"),
    emit,
    extract,
))]
#[derive(Debug, Clone, PartialEq)]
pub struct SttMetrics {
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    /// The request duration in seconds, 0.0 if the STT is streaming.
    pub duration: f64,
    /// The duration of the pushed audio in seconds; ``None`` when unknown.
    #[host(default = None)]
    pub audio_duration: Option<f64>,
    /// Whether the STT is streaming (e.g using websocket).
    pub streamed: bool,
    #[host(default_factory = "TokenUsage()")]
    pub tokens: TokenUsage,
    /// Time in seconds to acquire the connection. (WebSocket only)
    #[host(default = 0.0)]
    pub acquire_time: f64,
    /// Whether the connection was reused from a pool. (WebSocket only)
    #[host(default = false)]
    pub connection_reused: bool,
    #[host(default = None)]
    pub metadata: Option<Metadata>,
}

impl SttMetrics {
    pub const TYPE: &'static str = "stt_metrics";
}

// `metrics/base.py:67`.
#[crate::host(record(
    module = "metrics",
    name = "TTSMetrics",
    pydantic,
    base = "_BaseMetrics",
    constant(type = "tts_metrics"),
    emit,
    extract,
))]
#[derive(Debug, Clone, PartialEq)]
pub struct TtsMetrics {
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    /// Time to the first audio frame in seconds; ``None`` if no audio arrived.
    #[host(default = None)]
    pub ttfb: Option<f64>,
    pub duration: f64,
    pub audio_duration: f64,
    pub cancelled: bool,
    /// Number of characters synthesized (for character-based billing).
    pub characters_count: u32,
    /// Token-based billing (e.g., OpenAI TTS): text in, audio out.
    #[host(default_factory = "TokenUsage()")]
    pub tokens: TokenUsage,
    pub streamed: bool,
    /// Time in seconds to acquire the connection. (WebSocket only)
    #[host(default = 0.0)]
    pub acquire_time: f64,
    /// Whether the connection was reused from a pool. (WebSocket only)
    #[host(default = false)]
    pub connection_reused: bool,
    #[host(default = None)]
    pub segment_id: Option<String>,
    #[host(default = None)]
    pub speech_id: Option<String>,
    #[host(default = None)]
    pub metadata: Option<Metadata>,
}

impl TtsMetrics {
    pub const TYPE: &'static str = "tts_metrics";
}

// `metrics/base.py:94`. Never reaches the usage rollups.
#[crate::host(record(
    module = "metrics",
    name = "VADMetrics",
    pydantic,
    base = "_BaseMetrics",
    constant(type = "vad_metrics"),
    emit,
    extract,
))]
#[derive(Debug, Clone, PartialEq)]
pub struct VadMetrics {
    pub label: String,
    pub timestamp: f64,
    pub idle_time: f64,
    pub inference_duration_total: f64,
    pub inference_count: u32,
    #[host(default = None)]
    pub metadata: Option<Metadata>,
}

impl VadMetrics {
    pub const TYPE: &'static str = "vad_metrics";
}

// `metrics/base.py:103`. A real zero and an absent measurement are the same value in the delay
// fields, as in the reference.
#[crate::host(record(
    module = "metrics",
    name = "EOUMetrics",
    pydantic,
    base = "_BaseMetrics",
    constant(type = "eou_metrics"),
    emit,
    extract,
))]
#[derive(Debug, Clone, PartialEq)]
pub struct EouMetrics {
    pub timestamp: f64,
    /// Amount of time between the end of speech from VAD and the decision to end the user's turn.
    /// Set to 0.0 if the end of speech was not detected.
    pub end_of_utterance_delay: f64,
    /// Time taken to obtain the transcript after the end of the user's speech.
    /// Set to 0.0 if the end of speech was not detected.
    pub transcription_delay: f64,
    /// Time taken to invoke the user's `Agent.on_user_turn_completed` callback.
    pub on_user_turn_completed_delay: f64,
    #[host(default = None)]
    pub speech_id: Option<String>,
    #[host(default = None)]
    pub metadata: Option<Metadata>,
}

impl EouMetrics {
    pub const TYPE: &'static str = "eou_metrics";
}

// `metrics/base.py:136`. Its usage shares the `LLMUsage` entry of a text LLM of the same
// provider/model.
#[crate::host(record(
    module = "metrics",
    pydantic,
    base = "_BaseMetrics",
    constant(type = "realtime_model_metrics"),
    emit,
    extract,
))]
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeModelMetrics {
    // The only metric whose label has a default.
    #[host(default = "")]
    pub label: String,
    pub request_id: String,
    /// The timestamp of the response creation.
    pub timestamp: f64,
    /// The duration of the response from created to done in seconds.
    #[host(default = 0.0)]
    pub duration: f64,
    /// Metered session-connection seconds (for session-based billing like xAI).
    #[host(default = None)]
    pub connection_duration: Option<f64>,
    /// Time to first audio token in seconds; ``None`` if no audio token was sent.
    #[host(default = None)]
    pub ttft: Option<f64>,
    /// Whether the request was cancelled.
    #[host(default = false)]
    pub cancelled: bool,
    #[host(default_factory = "TokenUsage()")]
    pub tokens: TokenUsage,
    /// The number of tokens per second.
    #[host(default = 0.0)]
    pub tokens_per_second: f64,
    /// Time in seconds to acquire the connection. (WebSocket only)
    #[host(default = 0.0)]
    pub acquire_time: f64,
    /// Whether the connection was reused from a pool. (WebSocket only)
    #[host(default = false)]
    pub connection_reused: bool,
    #[host(default = None)]
    pub metadata: Option<Metadata>,
}

impl RealtimeModelMetrics {
    pub const TYPE: &'static str = "realtime_model_metrics";
}

// `metrics/base.py:200`. `exclude_none` tells `Some(0.0)` from `None`.
#[crate::host(record(
    module = "metrics",
    pydantic,
    base = "_BaseMetrics",
    constant(type = "avatar_metrics"),
    emit,
    extract,
))]
#[derive(Debug, Clone, PartialEq)]
pub struct AvatarMetrics {
    pub timestamp: f64,
    /// Delay between forwarding the first audio frame to the avatar and the playback started.
    #[host(default = 0)]
    pub playback_latency: f64,
    /// Time when the avatar session was started.
    #[host(default = None)]
    pub session_started_time: Option<f64>,
    /// Time when the avatar participant joined and started video track.
    #[host(default = None)]
    pub avatar_joined_time: Option<f64>,
    #[host(default = None)]
    pub metadata: Option<Metadata>,
}

impl AvatarMetrics {
    pub const TYPE: &'static str = "avatar_metrics";
}

// `metrics/base.py:224`'s `AgentMetrics` union, in the reference's order.
#[crate::host(record(module = "metrics", transparent, emit, extract))]
#[derive(Debug, Clone, PartialEq)]
pub enum AgentMetrics {
    Stt(SttMetrics),
    Llm(LlmMetrics),
    Tts(TtsMetrics),
    Vad(VadMetrics),
    Eou(EouMetrics),
    EotInference(EotInferenceMetrics),
    RealtimeModel(RealtimeModelMetrics),
    Interruption(InterruptionMetrics),
    Avatar(AvatarMetrics),
}

impl AgentMetrics {
    /// API: user code branches on it.
    pub fn type_literal(&self) -> &'static str {
        match self {
            AgentMetrics::Stt(_) => SttMetrics::TYPE,
            AgentMetrics::Llm(_) => LlmMetrics::TYPE,
            AgentMetrics::Tts(_) => TtsMetrics::TYPE,
            AgentMetrics::Vad(_) => VadMetrics::TYPE,
            AgentMetrics::Eou(_) => EouMetrics::TYPE,
            AgentMetrics::EotInference(_) => EotInferenceMetrics::TYPE,
            AgentMetrics::RealtimeModel(_) => RealtimeModelMetrics::TYPE,
            AgentMetrics::Interruption(_) => InterruptionMetrics::TYPE,
            AgentMetrics::Avatar(_) => AvatarMetrics::TYPE,
        }
    }

    pub fn metadata(&self) -> Option<&Metadata> {
        match self {
            AgentMetrics::Stt(m) => m.metadata.as_ref(),
            AgentMetrics::Llm(m) => m.metadata.as_ref(),
            AgentMetrics::Tts(m) => m.metadata.as_ref(),
            AgentMetrics::Vad(m) => m.metadata.as_ref(),
            AgentMetrics::Eou(m) => m.metadata.as_ref(),
            AgentMetrics::EotInference(m) => m.metadata.as_ref(),
            AgentMetrics::RealtimeModel(m) => m.metadata.as_ref(),
            AgentMetrics::Interruption(m) => m.metadata.as_ref(),
            AgentMetrics::Avatar(m) => m.metadata.as_ref(),
        }
    }
}
