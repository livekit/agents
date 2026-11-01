//! `inference.STT`'s native client: the LiveKit Inference gateway's WebSocket, implemented by
//! `lk-agents-inference`.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::errors::{ApiError, ProviderError};
use crate::inference::fallback::SttFallbackModel;
use crate::utils::stream::{StreamInput, StreamOutput};
use crate::vad::types::Vad;
use crate::{ApiConnectOptions, AudioFrame, SpeechEvent, Stt};

/// The gateway STT behind `inference.STT`. Its options, keyterms and open streams are shared by
/// every view (`identified`); each view's connections carry its own job's identification. As the
/// session's STT it implements the keyterm and conversation hooks.
#[crate::host(class(name = "InferenceStt", native = Stt))]
pub trait InferenceStt: Stt + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(sink(item = AudioFrame, class = "InferenceSttStream"))]
    type Audio: InferenceSttAudio + 'static;
    #[host(source(item = SpeechEvent, class = "InferenceSttStream"))]
    type Events: StreamOutput<SpeechEvent> + 'static;

    /// A client of `model` on the gateway at `base_url`. Starts nothing. `vad`'s end of speech
    /// finalizes the transcript (Speechmatics only).
    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &Self::Env,
        #[host(kw)] model: Option<String>,
        language: Option<String>,
        encoding: String,
        sample_rate: u32,
        base_url: String,
        api_key: String,
        api_secret: String,
        #[host(shape_error(
            kind = "ValueError",
            message = "expected a dict of extra kwargs, got {value}"
        ))]
        extra_kwargs: Map<String, Value>,
        #[host(default = None, shape_error(items, kind = "ValueError", message = "each fallback entry must be a dict"))]
        fallback: Option<Vec<SttFallbackModel>>,
        conn_options: ApiConnectOptions,
        #[host(default = None)] vad: Option<Arc<dyn Vad>>,
    ) -> Result<Self, ApiError>;

    /// The model's VAD as the host resolved it on `update_options(model=)`; `None` drops it.
    /// Later streams use it.
    #[host(call)]
    fn set_vad(&self, #[host(default = None)] vad: Option<Arc<dyn Vad>>);

    /// This client as a session of the job `headers` (`get_inference_headers()`) identifies holds
    /// it: the same options, keyterms and open streams, its connections carrying that job.
    /// `reporter` is the host `stt.STT` whose `error` events report this view's failures.
    #[host(call)]
    fn identified(
        &self,
        headers: Vec<(String, String)>,
        #[host(default = None)] reporter: Option<Arc<dyn Stt>>,
    ) -> Self;

    /// Read by later streams, and sent to the open ones as a mid-stream `session.update`.
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw, default = None)] model: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra: Option<Map<String, Value>>,
    );

    /// The session's keyterms: a warning once for a model without keyterm prompting, else every
    /// open stream applies them (now if the user is silent, at end of speech if not).
    #[host(call)]
    fn update_keyterms(&self, keyterms: Vec<String>) {
        Stt::update_keyterms(self, &keyterms);
    }

    /// An assistant turn's text, its tail, biases recognition as `agent_context` on a model that
    /// carries it.
    #[host(call)]
    fn push_conversation_item(&self, role: &str, text: &str) {
        Stt::push_conversation_item(self, role, text);
    }

    /// Opens a stream; its pump opens the socket. `start_time_offset` shifts its timestamps onto
    /// the caller's input timeline.
    #[host(call)]
    fn stream(
        &self,
        #[host(kw, default = None)] language: Option<String>,
        #[host(default = None)] start_time_offset: Option<f64>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<crate::DuplexStream<Self::Audio, Self::Events>, ProviderError>;
}

/// One live recognition's input: audio, and the options it is re-sent mid-stream.
#[crate::host(sink)]
pub trait InferenceSttAudio: StreamInput<AudioFrame> {
    /// A mid-stream `session.update`, for the providers that take one (AssemblyAI, Deepgram Flux).
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw, default = None)] model: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra: Option<Map<String, Value>>,
    );
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(
    module = "stt_provider_options",
    path = "livekit.agents.inference.stt",
    typed_dict
))]
pub struct CartesiaOptions {
    #[host(annotation = "float")]
    pub min_volume: crate::Maybe<f64>,
    #[host(annotation = "float")]
    pub max_silence_duration_secs: crate::Maybe<f64>,
    #[host(annotation = "float")]
    pub turn_start_threshold: crate::Maybe<f64>,
    #[host(annotation = "float")]
    pub turn_eager_end_threshold: crate::Maybe<f64>,
    #[host(annotation = "float")]
    pub turn_end_threshold: crate::Maybe<f64>,
    #[host(annotation = "int")]
    pub turn_end_timeout_ms: crate::Maybe<i64>,
    #[host(annotation = "str | list[str]")]
    pub keyterm: crate::Maybe<serde_json::Value>,
}

#[crate::host(record(
    module = "stt_provider_options",
    path = "livekit.agents.inference.stt",
    typed_dict
))]
pub struct DeepgramOptions {
    #[host(annotation = "bool")]
    pub filler_words: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub interim_results: crate::Maybe<bool>,
    #[host(annotation = "int")]
    pub endpointing: crate::Maybe<i64>,
    #[host(annotation = "bool")]
    pub punctuate: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub smart_format: crate::Maybe<bool>,
    #[host(annotation = "list[tuple[str, float]]")]
    pub keywords: crate::Maybe<Vec<(String, f64)>>,
    #[host(annotation = "str | list[str]")]
    pub keyterm: crate::Maybe<serde_json::Value>,
    #[host(annotation = "bool")]
    pub profanity_filter: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub numerals: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub mip_opt_out: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub vad_events: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub diarize: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub dictation: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub detect_language: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub no_delay: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub utterance_end: crate::Maybe<bool>,
    #[host(annotation = "str | list[str]")]
    pub redact: crate::Maybe<serde_json::Value>,
    #[host(annotation = "str | list[str]")]
    pub replace: crate::Maybe<serde_json::Value>,
    #[host(annotation = "str | list[str]")]
    pub search: crate::Maybe<serde_json::Value>,
    #[host(annotation = "str | list[str]")]
    pub tag: crate::Maybe<serde_json::Value>,
    #[host(annotation = "int")]
    pub channels: crate::Maybe<i64>,
    #[host(annotation = "str")]
    pub version: crate::Maybe<String>,
    #[host(annotation = "str")]
    pub callback: crate::Maybe<String>,
    #[host(annotation = "str")]
    pub callback_method: crate::Maybe<String>,
    #[host(annotation = "str")]
    pub extra: crate::Maybe<String>,
}

#[crate::host(record(
    module = "stt_provider_options",
    path = "livekit.agents.inference.stt",
    typed_dict
))]
pub struct DeepgramFluxOptions {
    #[host(annotation = "float")]
    pub eager_eot_threshold: crate::Maybe<f64>,
    #[host(annotation = "float")]
    pub eot_threshold: crate::Maybe<f64>,
    #[host(annotation = "int")]
    pub eot_timeout_ms: crate::Maybe<i64>,
    #[host(annotation = "str | list[str]")]
    pub keyterm: crate::Maybe<serde_json::Value>,
    #[host(annotation = "bool")]
    pub mip_opt_out: crate::Maybe<bool>,
    #[host(annotation = "str | list[str]")]
    pub tag: crate::Maybe<serde_json::Value>,
    #[host(annotation = "bool")]
    pub detect_language: crate::Maybe<bool>,
}

#[crate::host(record(
    module = "stt_provider_options",
    path = "livekit.agents.inference.stt",
    typed_dict
))]
pub struct AssemblyaiOptions {
    #[host(annotation = "bool")]
    pub format_turns: crate::Maybe<bool>,
    #[host(annotation = "float")]
    pub end_of_turn_confidence_threshold: crate::Maybe<f64>,
    #[host(annotation = "int")]
    pub min_end_of_turn_silence_when_confident: crate::Maybe<i64>,
    #[host(annotation = "int")]
    pub max_turn_silence: crate::Maybe<i64>,
    #[host(annotation = "list[str]")]
    pub keyterms_prompt: crate::Maybe<Vec<String>>,
    #[host(annotation = "bool")]
    pub language_detection: crate::Maybe<bool>,
    #[host(annotation = "float")]
    pub inactivity_timeout: crate::Maybe<f64>,
    #[host(annotation = "str")]
    pub prompt: crate::Maybe<String>,
    #[host(annotation = "bool")]
    pub speaker_labels: crate::Maybe<bool>,
    #[host(annotation = "str")]
    pub agent_context: crate::Maybe<String>,
    #[host(annotation = "int")]
    pub previous_context_n_turns: crate::Maybe<i64>,
    #[host(annotation = "typing.Literal['near-field', 'far-field']")]
    pub voice_focus: crate::Maybe<String>,
    #[host(annotation = "float")]
    pub voice_focus_threshold: crate::Maybe<f64>,
    #[host(annotation = "typing.Literal['min_latency', 'balanced', 'max_accuracy']")]
    pub mode: crate::Maybe<String>,
}

#[crate::host(record(
    module = "stt_provider_options",
    path = "livekit.agents.inference.stt",
    typed_dict
))]
pub struct SpeechmaticsOptions {
    #[host(annotation = "str")]
    pub domain: crate::Maybe<String>,
    #[host(annotation = "str")]
    pub output_locale: crate::Maybe<String>,
    #[host(annotation = "float")]
    pub max_delay: crate::Maybe<f64>,
    #[host(annotation = "str")]
    pub max_delay_mode: crate::Maybe<String>,
    #[host(annotation = "str")]
    pub diarization: crate::Maybe<String>,
    #[host(annotation = "float")]
    pub speaker_sensitivity: crate::Maybe<f64>,
    #[host(annotation = "int")]
    pub max_speakers: crate::Maybe<i64>,
    #[host(annotation = "bool")]
    pub prefer_current_speaker: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub enable_partials: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub enable_entities: crate::Maybe<bool>,
    #[host(annotation = "dict[str, typing.Any]")]
    pub punctuation_overrides: crate::Maybe<std::collections::BTreeMap<String, serde_json::Value>>,
    #[host(annotation = "list[dict[str, typing.Any]]")]
    pub additional_vocab: crate::Maybe<Vec<std::collections::BTreeMap<String, serde_json::Value>>>,
    #[host(annotation = "float")]
    pub end_of_utterance_silence_trigger: crate::Maybe<f64>,
    #[host(annotation = "dict[str, typing.Any]")]
    pub audio_filtering_config: crate::Maybe<std::collections::BTreeMap<String, serde_json::Value>>,
    #[host(annotation = "dict[str, typing.Any]")]
    pub transcript_filtering_config:
        crate::Maybe<std::collections::BTreeMap<String, serde_json::Value>>,
}

#[crate::host(record(
    module = "stt_provider_options",
    path = "livekit.agents.inference.stt",
    typed_dict
))]
pub struct XaiOptions {
    #[host(annotation = "bool")]
    pub diarize: crate::Maybe<bool>,
    #[host(annotation = "int")]
    pub endpointing: crate::Maybe<i64>,
    #[host(annotation = "bool")]
    pub format: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub interim_results: crate::Maybe<bool>,
}

#[crate::host(record(
    module = "stt_provider_options",
    path = "livekit.agents.inference.stt",
    typed_dict
))]
pub struct InworldOptions {
    #[host(annotation = "bool")]
    pub enable_voice_profile: crate::Maybe<bool>,
    #[host(annotation = "int")]
    pub voice_profile_top_n: crate::Maybe<i64>,
    #[host(annotation = "bool")]
    pub include_word_timestamps: crate::Maybe<bool>,
    #[host(annotation = "typing.Literal['LINEAR16', 'AUTO_DETECT']")]
    pub audio_encoding: crate::Maybe<String>,
    #[host(annotation = "int")]
    pub inactivity_timeout_seconds: crate::Maybe<i64>,
    #[host(annotation = "float")]
    pub end_of_turn_confidence_threshold: crate::Maybe<f64>,
    #[host(annotation = "int")]
    pub min_end_of_turn_silence_when_confident: crate::Maybe<i64>,
    #[host(annotation = "list[str]")]
    pub prompts: crate::Maybe<Vec<String>>,
    #[host(annotation = "float")]
    pub vad_threshold: crate::Maybe<f64>,
}

#[crate::host(record(
    module = "stt_provider_options",
    path = "livekit.agents.inference.stt",
    typed_dict
))]
pub struct GoogleOptions {
    #[host(annotation = "list[str]")]
    pub language_codes: crate::Maybe<Vec<String>>,
    #[host(annotation = "list[str]")]
    pub custom_vocabulary: crate::Maybe<Vec<String>>,
}
