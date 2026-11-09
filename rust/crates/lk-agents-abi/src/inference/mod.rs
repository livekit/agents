pub mod avatar_session;
pub mod fallback;
pub mod models;
pub mod stt_wire;
pub mod tts_wire;

use crate::errors::Invalid;

#[crate::host(file(module = "inference_headers"))]
const _: () = ();

/// The gateway's session identification header.
#[crate::host(constant(path = "livekit.agents.inference"))]
pub const HEADER_SESSION_ID: &str = "X-LiveKit-Session-ID";
/// The provider a gateway request asks for.
#[crate::host(constant(path = "livekit.agents.inference"))]
pub const HEADER_INFERENCE_PROVIDER: &str = "X-LiveKit-Inference-Provider";
/// A request's inference class (`"low"` for a background job).
#[crate::host(constant(path = "livekit.agents.inference"))]
pub const HEADER_INFERENCE_PRIORITY: &str = "X-LiveKit-Inference-Priority";

/// The inference gateway's credentials and identification. The core reads
/// `LIVEKIT_INFERENCE_URL`, `LIVEKIT_INFERENCE_API_KEY`, `LIVEKIT_INFERENCE_API_SECRET` and
/// `LIVEKIT_WORKER_TOKEN` from the environment, and the server's `LIVEKIT_URL` and key pair from
/// the running server's settings (in a job, its `AgentServer`'s), else the environment.
/// Where a gateway client connects, and with what.
#[crate::host(record(
    module = "inference_credentials",
    path = "livekit.agents.inference",
    name = "Credentials",
    dataclass,
    frozen,
    emit
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayCredentials {
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
}

#[crate::host(functions)]
pub trait InferenceCredentials {
    /// `explicit`, else `LIVEKIT_INFERENCE_API_KEY`, else the server's `LIVEKIT_API_KEY`; `""`
    /// when nothing resolves. A set but empty `LIVEKIT_INFERENCE_API_KEY` shadows the server's.
    #[host(function)]
    fn inference_api_key(#[host(default = None)] explicit: Option<String>) -> String;

    /// The same ladder for the secret, over `LIVEKIT_INFERENCE_API_SECRET`.
    #[host(function)]
    fn inference_api_secret(#[host(default = None)] explicit: Option<String>) -> String;

    /// A nonempty `LIVEKIT_INFERENCE_URL`, then staging if the server's `LIVEKIT_URL` names the
    /// staging cloud, then production.
    #[host(function)]
    fn inference_url() -> String;

    /// The gateway's URL (a nonempty `base_url`, else [`Self::inference_url`]) and key pair (the
    /// ladders above); a `ValueError` when the key or the secret does not resolve.
    #[host(function)]
    fn inference_credentials(
        #[host(default = None)] base_url: Option<String>,
        #[host(default = None)] api_key: Option<String>,
        #[host(default = None)] api_secret: Option<String>,
    ) -> Result<GatewayCredentials, Invalid>;

    /// An inference request's identification headers, in order: `user_agent`, the priority of a
    /// nonempty `inference_class`, and in a job (`job_headers` given) its nonempty room, job and
    /// agent ids, `LIVEKIT_WORKER_TOKEN` when set, the job's `job_headers` (overriding any of
    /// these) and a fresh session id.
    #[host(function)]
    fn inference_identity_headers(
        user_agent: String,
        #[host(default = None)] inference_class: Option<String>,
        #[host(default = None)] room_sid: Option<String>,
        #[host(default = None)] job_id: Option<String>,
        #[host(default = None)] agent_sid: Option<String>,
        #[host(default = None)] job_headers: Option<crate::Headers>,
    ) -> crate::Headers;
}
/// The gateway's model tables, request headers and local model registry the host reads: pure
/// lookups, safe in a forked child.
#[crate::host(functions(pid = false))]
pub trait InferenceModels {
    /// The gateway's quota dimensions among a response's header pairs (names case-insensitive);
    /// an empty value is not reported.
    #[host(function)]
    fn inference_quota_usage(headers: crate::Headers) -> Vec<(String, String)>;

    /// Loads the native models' weights; `false` when no provider is linked.
    #[host(function)]
    fn local_models_init(vad: bool, eot: bool) -> bool;

    /// The speech-markup dialect the model's provider reads.
    #[host(function)]
    fn tts_markup_provider_key(model: String) -> String;

    /// The lowest reasoning effort the model supports, or `None` if it has no such control.
    #[host(function)]
    fn min_reasoning_effort(model: String) -> Option<String>;

    /// Whether AMD may fill in its own LLM and STT from the gateway (`detector.py:184-194`).
    #[host(function)]
    fn amd_auto_select_models(livekit_url: String, api_key: String, api_secret: String) -> bool;

    /// Whether `model` is in AMD's evaluated `"llm"` or `"stt"` table.
    #[host(function)]
    fn amd_is_evaluated_model(model: String, kind: String) -> Result<bool, Invalid>;

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

    /// The turn detector a host constructs.
    #[host(function)]
    fn eot_select(
        version: Option<String>,
        hosted: bool,
        dev: bool,
        url: String,
        key: String,
        secret: String,
        local_fallback: bool,
    ) -> Result<EotSelection, Invalid>;

    /// `stt.FallbackAdapter`'s capabilities, merged from its members' before any chain is built.
    #[host(function)]
    fn stt_fallback_capabilities(entries: Vec<crate::SttCapabilities>) -> crate::SttCapabilities;

    /// `tts.FallbackAdapter`'s capabilities, merged from its members' before any chain is built.
    #[host(function)]
    fn tts_fallback_capabilities(entries: Vec<crate::TtsCapabilities>) -> crate::TtsCapabilities;
}

pub mod tts_client;

/// The model a turn detector runs, and why it is not the one asked for, to log.
#[crate::host(record(module = "inference_models", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct EotSelection {
    pub model: crate::tokenize::language::TurnDetectorModel,
    pub warnings: Vec<String>,
}
