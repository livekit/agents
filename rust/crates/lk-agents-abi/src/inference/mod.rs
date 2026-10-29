//! The inference gateway's host functions. The server-side fallback entries are in [`fallback`];
//! the gateway avatar's declaration in [`avatar_session`].

pub mod avatar_session;
pub mod fallback;

use serde_json::Value;

/// The inference gateway's credential ladders (`inference/_utils.py`). Each returns `""` when
/// nothing resolves: the caller raises with its own message, and tests the explicit value
/// (truthiness vs `is_given`) itself.
#[crate::host(functions)]
pub trait InferenceCredentials {
    /// `explicit`, else `LIVEKIT_INFERENCE_API_KEY`, else `server` (the running job's server
    /// key), else `LIVEKIT_API_KEY`; a set but empty `LIVEKIT_INFERENCE_API_KEY` shadows the rest.
    #[host(function)]
    fn inference_resolve_api_key(
        #[host(default = None)] explicit: Option<String>,
        #[host(default = None)] server: Option<String>,
    ) -> String;

    /// The same ladder for the secret.
    #[host(function)]
    fn inference_resolve_api_secret(
        #[host(default = None)] explicit: Option<String>,
        #[host(default = None)] server: Option<String>,
    ) -> String;
}
/// The gateway's model tables, request headers and local model registry the host reads: pure
/// lookups, safe in a forked child.
#[crate::host(functions(pid = false))]
pub trait InferenceModels {
    /// The gateway's quota dimensions among a response's header pairs (names case-insensitive);
    /// an empty value is not reported.
    #[host(function)]
    fn inference_quota_usage(headers: Vec<(String, String)>) -> Vec<(String, String)>;

    /// `LIVEKIT_INFERENCE_URL`, then staging if the server URL names the staging cloud, then
    /// production. `server_url` is the running job's (`AgentServer(ws_url=)`), else `LIVEKIT_URL`.
    #[host(function)]
    fn inference_default_url(#[host(default = None)] server_url: Option<String>) -> String;

    /// Loads the native models' weights; `false` when no provider is linked.
    #[host(function)]
    fn local_models_init(vad: bool, eot: bool) -> bool;

    /// `"provider/model:voice"` split into the model and its voice.
    #[host(function)]
    fn tts_parse_model_string(model: String) -> (String, Option<String>);

    /// Whether the model streams word timings for its audio under these options.
    #[host(function)]
    fn tts_has_aligned_transcript(
        model: String,
        extra_kwargs: serde_json::Map<String, Value>,
    ) -> bool;

    /// The speech-markup dialect the model's provider reads.
    #[host(function)]
    fn tts_markup_provider_key(model: String) -> String;

    /// `"provider/model:language"` split into the model and its language.
    #[host(function)]
    fn stt_parse_model_string(model: String) -> (String, Option<String>);

    /// The lowest reasoning effort the model supports, or `None` if it has no such control.
    #[host(function)]
    fn min_reasoning_effort(model: String) -> Option<String>;

    /// Whether AMD may fill in its own LLM and STT from the gateway (`detector.py:184-194`).
    #[host(function)]
    fn amd_auto_select_models(livekit_url: String, api_key: String, api_secret: String) -> bool;

    /// Whether `model` is in AMD's evaluated `"llm"` or `"stt"` table.
    #[host(function)]
    fn amd_is_evaluated_model(model: String, kind: String) -> Result<bool, InvalidOption>;

    /// `AMDPredictionEvent.is_human` for a category's wire value.
    #[host(function)]
    fn amd_category_is_human(category: String) -> Result<bool, InvalidOption>;

    /// `AMDPredictionEvent.is_machine` for a category's wire value.
    #[host(function)]
    fn amd_category_is_machine(category: String) -> Result<bool, InvalidOption>;

    /// The parameters the given model refuses. `has_tools` is `bool(tools)`; `reasoning_effort`
    /// is the request's `reasoning_effort` and `reasoning` the effort of its Responses
    /// `reasoning`.
    #[host(function)]
    fn llm_unsupported_params(
        model: String,
        #[host(kw, default = false)] has_tools: bool,
        #[host(default = None)] reasoning_effort: Option<String>,
        #[host(default = None)] reasoning: Option<String>,
    ) -> Vec<String>;

    /// The turn detector a host constructs: `(model, cloud, warnings)`.
    #[host(function)]
    #[allow(clippy::too_many_arguments)]
    fn eot_select(
        version: Option<String>,
        hosted: bool,
        dev: bool,
        url: String,
        key: String,
        secret: String,
        local_fallback: bool,
    ) -> Result<(crate::tokenize::language::TurnDetectorModel, bool, Vec<String>), InvalidOption>;

    /// `_resolve_vad_for_model`: what the host does with the caller's VAD for this model.
    #[host(function)]
    fn stt_resolve_vad_for_model(
        #[host(default = None)] model: Option<String>,
        #[host(kw, default = false)] has_vad: bool,
    ) -> VadResolution;

    /// `_normalize_fallback`'s string arm for an STT chain.
    #[host(function)]
    fn stt_normalize_fallback_str(model: String) -> fallback::SttFallbackModel;

    /// `_normalize_fallback`'s string arm for a TTS chain.
    #[host(function)]
    fn tts_normalize_fallback_str(model: String) -> fallback::TtsFallbackModel;

    /// `inference.STT`'s capabilities. Word alignment holds only when the model and every
    /// `fallback` model align: any turn may be served by a fallback, and a false claim disables
    /// the fast VAD interruption path.
    #[host(function)]
    fn stt_capabilities(
        #[host(default = None)] model: Option<String>,
        #[host(kw, default = None)] fallback: Option<Vec<String>>,
        #[host(default = None)] extra_kwargs: Option<serde_json::Map<String, Value>>,
    ) -> crate::SttCapabilities;

    /// `stt.FallbackAdapter`'s capabilities, merged from its members' before any chain is built.
    #[host(function)]
    fn stt_fallback_capabilities(entries: Vec<crate::SttCapabilities>) -> crate::SttCapabilities;

    /// `tts.FallbackAdapter`'s capabilities, merged from its members' before any chain is built.
    #[host(function)]
    fn tts_fallback_capabilities(entries: Vec<crate::TtsCapabilities>) -> crate::TtsCapabilities;

    /// The realtime fallback's capabilities: `audio_output` and `turn_detection` must agree (a
    /// `ValueError` naming the capability and both values), the rest are ANDed.
    #[host(function)]
    fn merge_realtime_capabilities(
        caps: Vec<crate::llm::realtime::RealtimeCapabilities>,
    ) -> Result<crate::llm::realtime::RealtimeCapabilities, InvalidOption>;
}

/// What `_resolve_vad_for_model` decides; the host performs it.
#[crate::host(wire_enum)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadResolution {
    /// Keep the caller's instance (Speechmatics, which endpoints client-side).
    Keep,
    /// Ignore the caller's instance and warn: the model endpoints server-side.
    DropAndWarn,
    /// Speechmatics with no VAD supplied: construct the default local one.
    CreateDefault,
    /// Nothing to do.
    None,
}

/// An option value the inference lookups do not know.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct InvalidOption(pub String);

pub mod tts_client;
