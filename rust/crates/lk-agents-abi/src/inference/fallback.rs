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
