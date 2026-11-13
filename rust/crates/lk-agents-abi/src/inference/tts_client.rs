use super::fallback::TtsFallbacks;
use crate::{ApiConnectOptions, Tts};

/// `inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::GatewayTts`.
#[crate::host(class(name = "InferenceTts", native = Tts))]
pub trait InferenceTts: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// A client of `model` (``"provider/model[:voice]"``) on the gateway. Starts nothing. The
    /// gateway's URL and key pair resolve from the given values, the environment and the server's
    /// settings.
    #[host(new)]
    #[allow(clippy::too_many_arguments, reason = "Preserve the public Python signature")]
    fn new(
        env: &Self::Env,
        model: String,
        #[host(kw, default = None)] voice: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None)] encoding: Option<String>,
        #[host(default = None)] sample_rate: Option<u32>,
        #[host(default = None)] base_url: Option<String>,
        #[host(default = None)] api_key: Option<String>,
        #[host(default = None)] api_secret: Option<String>,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
        #[host(default = None)] fallback: Option<TtsFallbacks>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<Self, crate::errors::Invalid>;
    #[host(getter)]
    fn model(&self) -> String;
    /// Streaming, its alignment read from the live model and extra kwargs.
    #[host(getter)]
    fn capabilities(&self) -> crate::TtsCapabilities;
    #[host(getter)]
    fn num_channels(&self) -> u16;
    /// The policy this client was given, which a session or run that supplies none keeps.
    #[host(getter)]
    fn conn_options(&self) -> ApiConnectOptions;
    #[host(getter)]
    fn sample_rate(&self) -> u32;
    #[host(call)]
    fn prewarm(&self, #[host(kw, default = None)] headers: Option<crate::Headers>);
    #[host(call)]
    fn identified(
        &self,
        headers: crate::Headers,
        #[host(default = None)] reporter: Option<std::sync::Arc<dyn Tts>>,
    ) -> Self;
    /// Read by the next synthesis; ``extra_kwargs`` merges into the persistent kwargs.
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw, default = None)] voice: Option<String>,
        #[host(default = None)] model: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    );
    /// Closes the pooled sockets every session's view shares; a synthesis still running
    /// finishes, its socket closed instead of pooled.
    #[host(await)]
    async fn aclose(&self);
}

/// `inference.TTS`: the LiveKit Inference gateway's TTS.
#[crate::host(facade(
    path = "livekit.agents.inference.tts",
    extended,
    name = "TTS",
    base = "tts.TTS",
    native = InferenceTts,
    legs(_synthesize, _stream),
    leg = "_core_native",
    base_args(capabilities, sample_rate, num_channels)
))]
pub trait InferenceTtsFacade {
    /// LiveKit Inference TTS.
    ///
    /// Args:
    ///     model: ``"provider/model[:voice]"``.
    ///     voice: The voice; overrides the model string's.
    ///     language: The language to synthesize.
    ///     encoding: The audio encoding asked of the gateway.
    ///     sample_rate: The sample rate asked of the gateway.
    ///     base_url: The gateway URL; ``LIVEKIT_INFERENCE_URL`` or the cloud gateway by default.
    ///     api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
    ///     api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
    ///     extra_kwargs: Provider options passed through to the gateway.
    ///     fallback: Fallback models, names or ``FallbackModel`` entries, run by the gateway.
    ///     conn_options: The connection timeout, and the retries the gateway runs server-side.
    #[host(
        new,
        not_given,
        annotation(
            model = ".inference_tts.TTSModels | str",
            encoding = ".inference_tts.TTSEncoding"
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the Python constructor's keyword arguments"
    )]
    fn new(
        model: String,
        #[host(kw, default = None)] voice: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None)] encoding: Option<String>,
        #[host(default = None)] sample_rate: Option<u32>,
        #[host(default = None)] base_url: Option<String>,
        #[host(default = None)] api_key: Option<String>,
        #[host(default = None)] api_secret: Option<String>,
        #[host(default = None)] extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
        #[host(default = None)] fallback: Option<TtsFallbacks>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Self;

    #[host(getter)]
    fn model(&self) -> String;

    #[host(getter)]
    fn capabilities(&self) -> crate::TtsCapabilities;

    /// Update the options the next synthesis uses: the gateway's, and how the framework prepares
    /// the text (``tts.TTS.update_options``).
    #[host(
        not_given,
        annotation(
            model = ".inference_tts.TTSModels | str",
            sentence_tokenizer = "..tokenize.SentenceTokenizer"
        ),
        base(sentence_tokenizer, pacing)
    )]
    fn update_options(
        &self,
        #[host(kw, default = None)] voice: Option<String>,
        #[host(default = None)] model: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None)] extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
        sentence_tokenizer: crate::Maybe<crate::HostRef>,
        pacing: crate::Maybe<Option<crate::tts::stream_pacer::SentenceStreamPacer>>,
    );
}

/// `inference.TTS.stream()`: one gateway synthesis, text in, audio and alignment out; the client
/// splits the sentences and converts the markup.
// Public data contracts. Python behavior remains in its adapters.

#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CartesiaSpeedPreset {
    Slow,
    Normal,
    Fast,
}

#[crate::host(record(
    module = "tts_provider_options",
    path = "livekit.agents.inference.tts",
    typed_dict
))]
pub struct CartesiaOptions {
    #[host(annotation = "str")]
    pub emotion: crate::Maybe<String>,
    #[host(annotation = "typing.Literal['slow', 'normal', 'fast'] | float")]
    pub speed: crate::Maybe<serde_json::Value>,
    #[host(annotation = "float")]
    pub volume: crate::Maybe<f64>,
    #[host(annotation = "float")]
    pub duration: crate::Maybe<f64>,
    #[host(annotation = "int")]
    pub max_buffer_delay_ms: crate::Maybe<i64>,
    #[host(annotation = "bool")]
    pub add_timestamps: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub add_phoneme_timestamps: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub use_normalized_timestamps: crate::Maybe<bool>,
}

#[crate::host(record(
    module = "tts_provider_options",
    path = "livekit.agents.inference.tts",
    typed_dict
))]
pub struct DeepgramOptions {
    #[host(annotation = "bool")]
    pub mip_opt_out: crate::Maybe<bool>,
}

/// Rime-specific WebSocket JSON query parameters.
///     See: https://docs.rime.ai/api-reference/endpoint/websockets-json
///
#[crate::host(record(
    module = "tts_provider_options",
    path = "livekit.agents.inference.tts",
    typed_dict
))]
pub struct RimeOptions {
    #[host(annotation = "float")]
    pub speed_alpha: crate::Maybe<f64>,
    #[host(annotation = "bool")]
    pub pause_between_brackets: crate::Maybe<bool>,
    #[host(annotation = "bool")]
    pub phonemize_between_brackets: crate::Maybe<bool>,
    #[host(annotation = "str")]
    pub inline_speed_alpha: crate::Maybe<String>,
    #[host(annotation = "bool")]
    pub no_text_normalization: crate::Maybe<bool>,
}

/// `InworldOptions.delivery_mode`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InworldDeliveryMode {
    #[host(rename = "DELIVERY_MODE_UNSPECIFIED")]
    #[serde(rename = "DELIVERY_MODE_UNSPECIFIED")]
    Unspecified,
    #[host(rename = "STABLE")]
    Stable,
    #[host(rename = "BALANCED")]
    Balanced,
    #[host(rename = "CREATIVE")]
    Creative,
}

/// `InworldOptions.timestamp_type`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InworldTimestampType {
    #[host(rename = "TIMESTAMP_TYPE_UNSPECIFIED")]
    #[serde(rename = "TIMESTAMP_TYPE_UNSPECIFIED")]
    Unspecified,
    #[host(rename = "WORD")]
    Word,
    #[host(rename = "CHARACTER")]
    Character,
}

/// `InworldOptions.apply_text_normalization`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InworldTextNormalization {
    #[host(rename = "APPLY_TEXT_NORMALIZATION_UNSPECIFIED")]
    #[serde(rename = "APPLY_TEXT_NORMALIZATION_UNSPECIFIED")]
    Unspecified,
    #[host(rename = "ON")]
    On,
    #[host(rename = "OFF")]
    Off,
}

#[crate::host(record(
    module = "tts_provider_options",
    path = "livekit.agents.inference.tts",
    typed_dict
))]
pub struct InworldOptions {
    #[host(annotation = "float")]
    pub speaking_rate: crate::Maybe<f64>,
    #[host(annotation = "float")]
    pub temperature: crate::Maybe<f64>,
    pub delivery_mode: crate::Maybe<InworldDeliveryMode>,
    pub timestamp_type: crate::Maybe<InworldTimestampType>,
    pub apply_text_normalization: crate::Maybe<InworldTextNormalization>,
}

#[crate::host(record(
    module = "tts_provider_options",
    path = "livekit.agents.inference.tts",
    typed_dict
))]
pub struct XaiOptions {
    #[host(annotation = "typing.Literal[32000, 64000, 96000, 128000, 192000]")]
    pub bit_rate: crate::Maybe<serde_json::Value>,
    #[host(annotation = "float")]
    pub speed: crate::Maybe<f64>,
    #[host(annotation = "typing.Literal[0, 1, 2]")]
    pub optimize_streaming_latency: crate::Maybe<serde_json::Value>,
}

/// `FishAudioOptions.latency`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FishAudioLatency {
    Normal,
    Balanced,
    Low,
}

/// See https://docs.fish.audio/api-reference/endpoint/websocket/tts-live
#[crate::host(record(
    module = "tts_provider_options",
    path = "livekit.agents.inference.tts",
    typed_dict
))]
pub struct FishAudioOptions {
    #[host(annotation = "float")]
    pub speed: crate::Maybe<f64>,
    #[host(annotation = "float")]
    pub volume: crate::Maybe<f64>,
    #[host(annotation = "bool")]
    pub normalize_loudness: crate::Maybe<bool>,
    #[host(annotation = "float")]
    pub temperature: crate::Maybe<f64>,
    #[host(annotation = "float")]
    pub top_p: crate::Maybe<f64>,
    #[host(annotation = "bool")]
    pub normalize: crate::Maybe<bool>,
    pub latency: crate::Maybe<FishAudioLatency>,
    #[host(annotation = "int")]
    pub chunk_length: crate::Maybe<i64>,
    #[host(annotation = "int")]
    pub max_new_tokens: crate::Maybe<i64>,
    #[host(annotation = "int")]
    pub min_chunk_length: crate::Maybe<i64>,
    #[host(annotation = "bool")]
    pub condition_on_previous_chunks: crate::Maybe<bool>,
    #[host(annotation = "float")]
    pub early_stop_threshold: crate::Maybe<f64>,
}

#[crate::host(record(
    module = "tts_provider_options",
    path = "livekit.agents.inference.tts",
    typed_dict
))]
pub struct AssemblyaiOptions {
    #[host(annotation = "bool")]
    pub word_timestamps: crate::Maybe<bool>,
}
