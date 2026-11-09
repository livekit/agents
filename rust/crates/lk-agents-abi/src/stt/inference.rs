use std::sync::Arc;

use serde_json::{Map, Value};

use crate::errors::{Invalid, ProviderError};
use crate::inference::fallback::SttFallbacks;
use crate::utils::stream::{StreamInput, StreamOutput};
use crate::vad::inference::VadModels;
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
    type Audio: StreamInput<AudioFrame> + 'static;
    #[host(source(item = SpeechEvent, class = "InferenceSttStream"))]
    type Events: StreamOutput<SpeechEvent> + 'static;

    /// A client of `model` (``"provider/model[:language]"``) on the gateway. Starts nothing. The
    /// gateway's URL and key pair resolve from the given values, the environment and the
    /// server's settings. `vad`'s end of speech finalizes the transcript of a model that
    /// endpoints client-side (Speechmatics, OpenAI), which gets a default one from
    /// `vad_models` when none is given; a model that endpoints server-side ignores it.
    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &Self::Env,
        vad_models: Arc<dyn VadModels>,
        #[host(default = None)] model: Option<String>,
        #[host(kw, default = None)] language: Option<String>,
        #[host(default = None)] base_url: Option<String>,
        #[host(default = None)] encoding: Option<String>,
        #[host(default = None)] sample_rate: Option<u32>,
        #[host(default = None)] api_key: Option<String>,
        #[host(default = None)] api_secret: Option<String>,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra_kwargs: Option<Map<String, Value>>,
        #[host(default = None)] fallback: Option<SttFallbacks>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
        #[host(default = None)] vad: Option<Arc<dyn Vad>>,
    ) -> Result<Self, Invalid>;

    /// `"unknown"` without a model.
    #[host(getter)]
    fn model(&self) -> String;

    /// Recomputed from the model, its fallbacks and the extra kwargs.
    #[host(getter)]
    fn capabilities(&self) -> crate::SttCapabilities;

    /// The policy this client was given, which a session or run that supplies none keeps.
    #[host(getter)]
    fn conn_options(&self) -> ApiConnectOptions;

    /// This client as a session of the job `headers` (`get_inference_headers()`) identifies holds
    /// it: the same options, keyterms and open streams, its connections carrying that job.
    /// `reporter` is the host `stt.STT` whose `error` events report this view's failures.
    #[host(call)]
    fn identified(
        &self,
        headers: crate::Headers,
        #[host(default = None)] reporter: Option<Arc<dyn Stt>>,
    ) -> Self;

    /// Read by later streams, and sent to the open ones as a mid-stream `session.update`. A new
    /// model resolves its VAD again.
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

/// `inference.STT`: the LiveKit Inference gateway's STT.
#[crate::host(facade(
    path = "livekit.agents.inference.stt",
    extended,
    name = "STT",
    base = "stt.STT",
    native = InferenceStt,
    legs(stream),
    leg = "_core_native",
    base_args(capabilities),
    host_args(vad_models = ".._bindings.silero_model")
))]
pub trait InferenceSttFacade {
    /// LiveKit Inference STT.
    ///
    /// Args:
    ///     model: ``"provider/model[:language]"``.
    ///     language: The language to transcribe; overrides the model string's.
    ///     base_url: The gateway URL; ``LIVEKIT_INFERENCE_URL`` or the cloud gateway by default.
    ///     encoding: The audio encoding sent to the gateway.
    ///     sample_rate: The sample rate sent to the gateway. Defaults to 24000 for OpenAI,
    ///         16000 for other providers.
    ///     api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
    ///     api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
    ///     extra_kwargs: Provider options passed through to the gateway.
    ///     fallback: Fallback models, names or ``FallbackModel`` entries, run by the gateway.
    ///     conn_options: The connection timeout, and the retries the gateway runs server-side.
    ///     vad: A VAD whose end of speech finalizes the transcript (Speechmatics and OpenAI).
    #[host(
        new,
        not_given,
        annotation(
            model = ".inference_stt.STTModels | str",
            encoding = ".inference_stt.STTEncoding"
        ),
        resolve(vad = ".._bindings.core_leg")
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[host(default = None)] model: Option<String>,
        #[host(kw, default = None)] language: Option<String>,
        #[host(default = None)] base_url: Option<String>,
        #[host(default = None)] encoding: Option<String>,
        #[host(default = None)] sample_rate: Option<u32>,
        #[host(default = None)] api_key: Option<String>,
        #[host(default = None)] api_secret: Option<String>,
        #[host(default = None)] extra_kwargs: Option<Map<String, Value>>,
        #[host(default = None)] fallback: Option<SttFallbacks>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
        #[host(default = None)] vad: Option<Arc<dyn Vad>>,
    ) -> Self;

    #[host(getter)]
    fn model(&self) -> String;

    #[host(getter)]
    fn capabilities(&self) -> crate::SttCapabilities;

    /// Update the options; open streams get them as a mid-stream ``session.update``.
    #[host(
        not_given,
        annotation(
            model = ".inference_stt.STTModels | str",
            language = ".inference_stt.STTLanguages | str"
        )
    )]
    fn update_options(
        &self,
        #[host(kw, default = None)] model: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None)] extra: Option<Map<String, Value>>,
    );
}

/// `inference.STT.stream()`: one gateway recognition session; the client reconnects it under its
/// ``conn_options``.
#[crate::host(facade(
    path = "livekit.agents.inference.stt",
    base = "stt.RecognizeStream",
    native = InferenceStt,
    stream(
        op = stream,
        owner = "_stt",
        native_retries,
        method = "stream",
        base_args(conn_options),
        state(start_time_offset),
        provider_default(conn_options),
        annotation(language = ".inference_stt.STTLanguages | str"),
        not_given
    )
))]
pub trait SpeechStream {}

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
    #[host(annotation = "str | list[str]")]
    pub language_hint: crate::Maybe<serde_json::Value>,
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
    #[host(annotation = "str | list[str]")]
    pub language_hint: crate::Maybe<serde_json::Value>,
}

/// `AssemblyaiOptions.voice_focus`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AssemblyaiVoiceFocus {
    #[host(rename = "near-field")]
    NearField,
    #[host(rename = "far-field")]
    FarField,
}

/// `AssemblyaiOptions.mode`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssemblyaiMode {
    MinLatency,
    Balanced,
    MaxAccuracy,
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
    pub voice_focus: crate::Maybe<AssemblyaiVoiceFocus>,
    #[host(annotation = "float")]
    pub voice_focus_threshold: crate::Maybe<f64>,
    pub mode: crate::Maybe<AssemblyaiMode>,
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

/// `InworldOptions.audio_encoding`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InworldAudioEncoding {
    #[host(rename = "LINEAR16")]
    Linear16,
    #[host(rename = "AUTO_DETECT")]
    AutoDetect,
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
    pub audio_encoding: crate::Maybe<InworldAudioEncoding>,
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

/// The latency/accuracy tradeoff of OpenAI live transcription.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenaiDelay {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenaiNoiseReduction {
    NearField,
    FarField,
}

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepgramFluxRedact {
    Numbers,
    AggressiveNumbers,
}

#[crate::host(record(
    module = "stt_provider_options",
    path = "livekit.agents.inference.stt",
    name = "OpenAIOptions",
    typed_dict
))]
pub struct OpenaiOptions {
    #[host(annotation = "list[str]")]
    pub languages: crate::Maybe<Vec<String>>,
    #[host(annotation = "str")]
    pub prompt: crate::Maybe<String>,
    #[host(annotation = "list[str]")]
    pub keywords: crate::Maybe<Vec<String>>,
    pub delay: crate::Maybe<OpenaiDelay>,
    pub noise_reduction: crate::Maybe<OpenaiNoiseReduction>,
}
