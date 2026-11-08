//! One entry of the gateway's server-side fallback chain (`FallbackModel`). On the wire STT sends an
//! absent `extra_kwargs` as `null` and TTS as `{}`; that lives in the request builders.

/// Inference Fallback Adapter: configuration for a fallback STT model that runs server-side in LiveKit Inference, providing automatic fallback between providers.
///
/// Extra fields are passed through to the provider.
///
/// Example:
///     >>> FallbackModel(model="deepgram/nova-3", extra_kwargs={"keyterm": ["livekit"]})
//
// The `///` above is the reference's docstring (`inference/stt.py`), carried by the `TypedDict`.
#[crate::host(record(module = "inference_stt", name = "FallbackModel", typed_dict, extract, emit))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SttFallbackModel {
    /// Model name (e.g. "deepgram/nova-3", "assemblyai/universal-streaming", "cartesia/ink-whisper").
    pub model: String,
    /// Extra configuration for the model.
    //
    // Omitted rather than nulled: `_normalize_fallback("x")` must equal `[{"model": "x"}]`. On the
    // read side absent and `None` are one request.
    #[host(default = None)]
    pub extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Inference Fallback Adapter: configuration for a fallback TTS model that runs server-side in LiveKit Inference, providing automatic fallback between providers.
///
/// Extra fields are passed through to the provider.
///
/// Example:
///     >>> FallbackModel(model="cartesia/sonic", voice="")
//
// The `///` above is the reference's docstring (`inference/tts.py`, a `total=True` `TypedDict`).
#[crate::host(record(module = "inference_tts", name = "FallbackModel", typed_dict, extract, emit))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TtsFallbackModel {
    /// Model name (e.g. "cartesia/sonic", "inworld/inworld-tts-1", "rime/coda").
    pub model: String,
    /// Voice to use for the model.
    //
    // `""` when a bare model string named no voice, or the entry gives none.
    #[host(default = "")]
    pub voice: String,
    /// Extra configuration for the model.
    //
    // Omitted, not `None`: a `None` would reach the gateway as `"extra": null`, STT's protocol.
    #[host(default = None)]
    pub extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

/// One server-side fallback of `inference.STT`: a `FallbackModel` or a model name.
#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "FallbackModelType",
    extract,
    transparent
))]
#[derive(Debug, Clone)]
pub enum SttFallbackEntry {
    Model(SttFallbackModel),
    Name(String),
}

/// `inference.STT(fallback=)`: one entry or a list of them.
#[crate::host(record(module = "inference_stt", name = "FallbackModels", extract, transparent))]
#[derive(Debug, Clone)]
pub enum SttFallbacks {
    Many(Vec<SttFallbackEntry>),
    Model(SttFallbackModel),
    Name(String),
}

impl SttFallbacks {
    /// The entries, in order.
    pub fn entries(self) -> Vec<SttFallbackEntry> {
        match self {
            SttFallbacks::Many(entries) => entries,
            SttFallbacks::Model(model) => vec![SttFallbackEntry::Model(model)],
            SttFallbacks::Name(name) => vec![SttFallbackEntry::Name(name)],
        }
    }
}

/// One server-side fallback of `inference.TTS`: a `FallbackModel` or a model name.
#[crate::host(record(
    module = "inference_tts",
    path = "livekit.agents.inference.tts",
    name = "FallbackModelType",
    extract,
    transparent
))]
#[derive(Debug, Clone)]
pub enum TtsFallbackEntry {
    Model(TtsFallbackModel),
    Name(String),
}

/// `inference.TTS(fallback=)`: one entry or a list of them.
#[crate::host(record(module = "inference_tts", name = "FallbackModels", extract, transparent))]
#[derive(Debug, Clone)]
pub enum TtsFallbacks {
    Many(Vec<TtsFallbackEntry>),
    Model(TtsFallbackModel),
    Name(String),
}

impl TtsFallbacks {
    /// The entries, in order.
    pub fn entries(self) -> Vec<TtsFallbackEntry> {
        match self {
            TtsFallbacks::Many(entries) => entries,
            TtsFallbacks::Model(model) => vec![TtsFallbackEntry::Model(model)],
            TtsFallbacks::Name(name) => vec![TtsFallbackEntry::Name(name)],
        }
    }
}
