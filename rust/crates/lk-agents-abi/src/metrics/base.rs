//! `metrics/base.py`. A `///` on a declared type or field is API (it lands in the schema digest the
//! gate compares); Rust-only notes go in `//`. Counts are `u32` here, `u64` in `usage`.

use lk_agents_host_derive::Host;

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "metrics", path = "livekit.agents.metrics.base"))]
const _: () = ();

// `metrics/base.py:8`. `Some("")` and `None` differ under `model_dump(exclude_none=True)`, so the
// fields stay `Option`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(pydantic, zero_defaults, name = "Metadata")]
#[binding(emit, extract)]
pub struct Metadata {
    pub model_name: Option<String>,
    pub model_provider: Option<String>,
}

// `metrics/base.py:13`. Generated because the gate records `bases`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "_BaseMetrics",
    repr = "metrics_repr"
)]
pub struct BaseMetrics {}

// `metrics/base.py:188`. The counters are per event, not cumulative: totals are the collector's.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "InterruptionMetrics",
    base = "_BaseMetrics",
    tag = "type",
    tag_const = TYPE
)]
// Both directions: the emit half is the detector's `"metrics"` event, the read half rebuilds this
// model (`agent_session.py:2370`). The channel tag is `"metrics"`, not the union's `TYPE`.
#[binding(emit, extract, tag = "type", tag_value = "metrics")]
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
    // Emit side only: the host builds `Metadata` itself.
    #[binding(skip)]
    #[python(default = "None")]
    pub metadata: Option<Metadata>,
}

impl InterruptionMetrics {
    pub const TYPE: &'static str = "interruption_metrics";
}

// A second projection `from` the same record, so fields cannot drift.
#[derive(Host)]
#[binding(
    emit,
    from = lk_agents_abi::metrics::base::InterruptionMetrics,
    tag = "type",
    tag_value = "interruption_metrics"
)]
pub struct InterruptionMetricsModel {
    pub timestamp: f64,
    pub total_duration: f64,
    pub prediction_duration: f64,
    pub detection_delay: f64,
    pub num_interruptions: u32,
    pub num_backchannels: u32,
    pub num_requests: u32,
    pub metadata: Option<Metadata>,
}

// `metrics/base.py:121`, emitted per `eot_prediction` (`inference/eot/transports.py:234`).
// `total_duration` (server's `client_e2e_latency`) and `detection_delay` (local) are not synonyms.
/// Per-inference metrics emitted by the EOT model on each prediction.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "EOTInferenceMetrics",
    base = "_BaseMetrics",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract)]
pub struct EotInferenceMetrics {
    pub timestamp: f64,
    /// Earliest audio creation time in an inference to response receive time.
    pub total_duration: f64,
    /// Latest audio creation time in an inference to response receive time.
    pub detection_delay: f64,
    /// Server side model inference time.
    pub prediction_duration: f64,
    /// Number of inference requests made during one inference.
    #[python(default = "1")]
    pub num_requests: u32,
    #[python(default = "None")]
    pub metadata: Option<Metadata>,
}

impl EotInferenceMetrics {
    pub const TYPE: &'static str = "eot_inference_metrics";
    /// The model's only defaulted field (so no `Default` impl).
    pub const DEFAULT_NUM_REQUESTS: u32 = 1;
}

// `metrics/base.py:19`.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "LLMMetrics",
    base = "_BaseMetrics",
    tag = "type",
    tag_const = TYPE
)]
// Both directions on every record below: read by the collectors, emitted as the dict the pydantic
// `AgentMetrics` union validates. `type` comes from `tag_const`.
#[binding(emit, extract)]
pub struct LlmMetrics {
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    pub duration: f64,
    /// Time to first generated token in seconds. -1 if the response generated none.
    pub ttft: f64,
    pub cancelled: bool,
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    // Cache *reads*. Already included in `prompt_tokens` by every provider that reports it.
    pub prompt_cached_tokens: u32,
    /// The number of tokens used to write to the prompt cache (e.g. Anthropic cache writes).
    ///
    /// Not all providers report this. ``prompt_cached_tokens`` covers cache reads.
    #[python(default = "0")]
    pub cache_creation_tokens: u32,
    pub total_tokens: u32,
    pub tokens_per_second: f64,
    #[python(default = "None")]
    pub speech_id: Option<String>,
    #[python(default = "None")]
    pub metadata: Option<Metadata>,
}

impl LlmMetrics {
    pub const TYPE: &'static str = "llm_metrics";
}

// `metrics/base.py:44`.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "STTMetrics",
    base = "_BaseMetrics",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract)]
pub struct SttMetrics {
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    /// The request duration in seconds, 0.0 if the STT is streaming.
    pub duration: f64,
    /// The duration of the pushed audio in seconds.
    pub audio_duration: f64,
    /// Input audio tokens (for token-based billing).
    #[python(default = "0")]
    pub input_tokens: u32,
    /// Output text tokens (for token-based billing).
    #[python(default = "0")]
    pub output_tokens: u32,
    /// Whether the STT is streaming (e.g using websocket).
    pub streamed: bool,
    /// Time in seconds to acquire the connection. (WebSocket only)
    #[python(default = "0.0")]
    pub acquire_time: f64,
    /// Whether the connection was reused from a pool. (WebSocket only)
    #[python(default = "False")]
    pub connection_reused: bool,
    #[python(default = "None")]
    pub metadata: Option<Metadata>,
}

impl SttMetrics {
    pub const TYPE: &'static str = "stt_metrics";
}

// `metrics/base.py:67`.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "TTSMetrics",
    base = "_BaseMetrics",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract)]
pub struct TtsMetrics {
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    pub ttfb: f64,
    pub duration: f64,
    pub audio_duration: f64,
    pub cancelled: bool,
    /// Number of characters synthesized (for character-based billing).
    pub characters_count: u32,
    /// Input text tokens (for token-based billing, e.g., OpenAI TTS).
    #[python(default = "0")]
    pub input_tokens: u32,
    /// Output audio tokens (for token-based billing, e.g., OpenAI TTS).
    #[python(default = "0")]
    pub output_tokens: u32,
    pub streamed: bool,
    /// Time in seconds to acquire the connection. (WebSocket only)
    #[python(default = "0.0")]
    pub acquire_time: f64,
    /// Whether the connection was reused from a pool. (WebSocket only)
    #[python(default = "False")]
    pub connection_reused: bool,
    #[python(default = "None")]
    pub segment_id: Option<String>,
    #[python(default = "None")]
    pub speech_id: Option<String>,
    #[python(default = "None")]
    pub metadata: Option<Metadata>,
}

impl TtsMetrics {
    pub const TYPE: &'static str = "tts_metrics";
}

// `metrics/base.py:94`. Never reaches the usage rollups.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "VADMetrics",
    base = "_BaseMetrics",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract)]
pub struct VadMetrics {
    pub label: String,
    pub timestamp: f64,
    pub idle_time: f64,
    pub inference_duration_total: f64,
    pub inference_count: u32,
    #[python(default = "None")]
    pub metadata: Option<Metadata>,
}

impl VadMetrics {
    pub const TYPE: &'static str = "vad_metrics";
}

// `metrics/base.py:103`. A real zero and an absent measurement are the same value in the delay
// fields, as in the reference.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    name = "EOUMetrics",
    base = "_BaseMetrics",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract)]
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
    #[python(default = "None")]
    pub speech_id: Option<String>,
    #[python(default = "None")]
    pub metadata: Option<Metadata>,
}

impl EouMetrics {
    pub const TYPE: &'static str = "eou_metrics";
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[python(
    pydantic,
    zero_defaults,
    name = "CachedTokenDetails",
    nested_in = "RealtimeModelMetrics"
)]
#[binding(emit, extract)]
pub struct CachedTokenDetails {
    pub audio_tokens: u32,
    pub text_tokens: u32,
    pub image_tokens: u32,
}

// A cached total without a breakdown adds nothing to the per-modality cached counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[python(
    pydantic,
    zero_defaults,
    name = "InputTokenDetails",
    nested_in = "RealtimeModelMetrics"
)]
#[binding(emit, extract)]
pub struct InputTokenDetails {
    pub audio_tokens: u32,
    pub text_tokens: u32,
    pub image_tokens: u32,
    pub cached_tokens: u32,
    #[python(ty = "RealtimeModelMetrics.CachedTokenDetails | None", default = "None")]
    pub cached_tokens_details: Option<CachedTokenDetails>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[python(
    pydantic,
    zero_defaults,
    name = "OutputTokenDetails",
    nested_in = "RealtimeModelMetrics"
)]
#[binding(emit, extract)]
pub struct OutputTokenDetails {
    pub text_tokens: u32,
    pub audio_tokens: u32,
    // Deprecated upstream; the deprecated `UsageSummary` still sums it, `ModelUsageCollector` does not.
    pub image_tokens: u32,
}

// `metrics/base.py:136`. Rolls up into an `LlmModelUsage` merged with a text LLM of the same
// provider/model.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    zero_defaults,
    name = "RealtimeModelMetrics",
    base = "_BaseMetrics",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract)]
pub struct RealtimeModelMetrics {
    // The only metric whose label has a default.
    #[python(default = "\"\"")]
    pub label: String,
    pub request_id: String,
    /// The timestamp of the response creation.
    #[python(required)]
    pub timestamp: f64,
    /// The duration of the response from created to done in seconds.
    pub duration: f64,
    /// The duration of the session connection in seconds (for session-based billing like xAI).
    pub session_duration: f64,
    /// Time to first audio token in seconds. -1 if no audio token was sent.
    #[python(default = "-1")]
    pub ttft: f64,
    /// Whether the request was cancelled.
    #[python(default = "False")]
    pub cancelled: bool,
    /// The number of input tokens used in the Response, including text and audio tokens.
    pub input_tokens: u32,
    /// The number of output tokens sent in the Response, including text and audio tokens.
    pub output_tokens: u32,
    /// The total number of tokens in the Response.
    pub total_tokens: u32,
    /// The number of tokens per second.
    pub tokens_per_second: f64,
    /// Details about the input tokens used in the Response.
    pub input_token_details: InputTokenDetails,
    /// Details about the output tokens used in the Response.
    pub output_token_details: OutputTokenDetails,
    /// Time in seconds to acquire the connection. (WebSocket only)
    pub acquire_time: f64,
    /// Whether the connection was reused from a pool. (WebSocket only)
    #[python(default = "False")]
    pub connection_reused: bool,
    pub metadata: Option<Metadata>,
}

impl RealtimeModelMetrics {
    pub const TYPE: &'static str = "realtime_model_metrics";
    /// The "no audio token was sent" sentinel.
    pub const NO_TTFT: f64 = -1.0;
}

// `metrics/base.py:200`. `exclude_none` tells `Some(0.0)` from `None`.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    zero_defaults,
    name = "AvatarMetrics",
    base = "_BaseMetrics",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract)]
pub struct AvatarMetrics {
    #[python(required)]
    pub timestamp: f64,
    /// Delay between forwarding the first audio frame to the avatar and the playback started.
    #[python(default = "0")]
    pub playback_latency: f64,
    /// Time when the avatar session was started.
    pub session_started_time: Option<f64>,
    /// Time when the avatar participant joined and started video track.
    pub avatar_joined_time: Option<f64>,
    pub metadata: Option<Metadata>,
}

impl AvatarMetrics {
    pub const TYPE: &'static str = "avatar_metrics";
}

/// `metrics/base.py:224`'s `AgentMetrics` union, in the reference's order.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(union, name = "AgentMetrics")]
#[binding(emit, extract)]
pub enum AgentMetrics {
    Stt(SttMetrics),
    Llm(LlmMetrics),
    Tts(TtsMetrics),
    Vad(VadMetrics),
    Eou(EouMetrics),
    EotInference(EotInferenceMetrics),
    RealtimeModel(RealtimeModelMetrics),
    #[binding(twin = InterruptionMetricsModel)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::testkit as tk;


    // Catches: a copy-pasted arm in `type_literal`.
    #[test]
    fn every_variant_reports_its_own_literal() {
        let all = [
            (AgentMetrics::Stt(tk::stt()), SttMetrics::TYPE),
            (AgentMetrics::Llm(tk::llm()), LlmMetrics::TYPE),
            (AgentMetrics::Tts(tk::tts()), TtsMetrics::TYPE),
            (AgentMetrics::Vad(tk::vad()), VadMetrics::TYPE),
            (AgentMetrics::Eou(tk::eou()), EouMetrics::TYPE),
            (AgentMetrics::EotInference(tk::eot()), EotInferenceMetrics::TYPE),
            (AgentMetrics::RealtimeModel(tk::realtime()), RealtimeModelMetrics::TYPE),
            (AgentMetrics::Interruption(tk::interruption()), InterruptionMetrics::TYPE),
            (AgentMetrics::Avatar(tk::avatar()), AvatarMetrics::TYPE),
        ];
        for (m, want) in &all {
            assert_eq!(m.type_literal(), *want);
        }
    }

    // Catches: a `metadata()` arm reading the wrong variant's field.
    #[test]
    fn metadata_reads_the_variant_it_was_given() {
        let mut vad = tk::vad();
        vad.metadata = tk::meta("livekit", "silero");
        assert_eq!(
            AgentMetrics::Vad(vad).metadata().and_then(|m| m.model_name.clone()).as_deref(),
            Some("silero")
        );
        let mut avatar = tk::avatar();
        avatar.metadata = tk::meta("tavus", "phoenix");
        assert_eq!(
            AgentMetrics::Avatar(avatar).metadata().and_then(|m| m.model_name.clone()).as_deref(),
            Some("phoenix")
        );
    }
}
