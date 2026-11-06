//! The native gateway synthesis resource. Runtime owns its connections and secrets.

use super::fallback::TtsFallbackModel;
use crate::errors::ProviderError;
use crate::{ApiConnectOptions, Tts};

/// `inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::GatewayTts`.
#[crate::host(class(name = "InferenceTts", native = Tts))]
pub trait InferenceTts: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "TtsStream"))]
    type Stream;
    /// `inference.TTS`'s mixin builds it on first use, after a fork, from the TTS's attributes.
    #[host(new)]
    #[host(facade(
        module = "provider_hosts",
        class = "InferenceTtsHost",
        target = "_native",
        cached,
        legs(synthesize, stream),
        leg = "_core_native",
        annotation(
            language = ".language.LanguageCode | None",
            sample_rate = "int",
            extra_kwargs = "dict[str, Any]",
            fallback = "list[.inference_tts.FallbackModel]"
        )
    ))]
    #[allow(clippy::too_many_arguments, reason = "Preserve the public Python signature")]
    fn new(
        env: &Self::Env,
        model: String,
        #[host(kw)] base_url: String,
        api_key: String,
        api_secret: String,
        conn_options: ApiConnectOptions,
        #[host(default = None)] voice: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = "pcm_s16le")] encoding: String,
        #[host(default = 24_000)] sample_rate: u32,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
        #[host(default = None, shape_error(items, kind = "ValueError", message = "each fallback entry must be a dict"))]
        fallback: Option<Vec<TtsFallbackModel>>,
    ) -> Result<Self, crate::llm::inference::InvalidPayload>;
    #[host(getter)]
    fn sample_rate(&self) -> u32;
    #[host(getter)]
    fn aligned_transcript(&self) -> bool;
    #[host(call)]
    fn prewarm(&self, #[host(kw, default = None)] headers: Option<Vec<(String, String)>>);
    #[host(call)]
    fn identified(
        &self,
        headers: Vec<(String, String)>,
        #[host(default = None)] reporter: Option<std::sync::Arc<dyn Tts>>,
    ) -> Self;
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw, default = None)] model: Option<String>,
        #[host(default = None)] voice: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra: Option<serde_json::Map<String, serde_json::Value>>,
    );
    #[host(await)]
    async fn stream(
        &self,
        #[host(kw, default = None)] headers: Option<Vec<(String, String)>>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
        #[host(default = false)] expressive: bool,
    ) -> Result<Self::Stream, ProviderError>;
}

// Public data contracts. Python behavior remains in its adapters.

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InworldDeliveryMode {
    #[host(rename = "DELIVERY_MODE_UNSPECIFIED")]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InworldTimestampType {
    #[host(rename = "TIMESTAMP_TYPE_UNSPECIFIED")]
    Unspecified,
    #[host(rename = "WORD")]
    Word,
    #[host(rename = "CHARACTER")]
    Character,
}

/// `InworldOptions.apply_text_normalization`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InworldTextNormalization {
    #[host(rename = "APPLY_TEXT_NORMALIZATION_UNSPECIFIED")]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
