//! Answering-machine detection's host boundary: `AMD(session, ...)`'s options, the run a session
//! hands back, the run's public events, and the decision-model port.

use crate::errors::{Invalid, ProviderError};
use crate::{ApiConnectOptions, Llm, Stt, ToolSpec};
use std::sync::Arc;

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "amd", path = "livekit.agents.voice.amd"))]
const _: () = ();

/// The classifier AMD selects when `llm` is not given and LiveKit Cloud credentials exist.
#[crate::host(constant)]
pub const DEFAULT_AMD_LLM_MODEL: &str = "google/gemma-4-31b-it";

/// AMD's own STT when `stt` is not given and LiveKit Cloud credentials exist.
#[crate::host(constant)]
pub const DEFAULT_AMD_STT_MODEL: &str = "cartesia/ink-whisper";

/// A per-turn prediction or a stage. `uncertain` and `wait` keep the stage.
#[crate::host(record(module = "amd", name = "AMDCategory", emit, extract, enum))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdCategory {
    #[host(rename = "human")]
    Human,
    #[host(rename = "machine-ivr")]
    MachineIvr,
    #[host(rename = "machine-screening")]
    MachineScreening,
    #[host(rename = "machine-vm")]
    MachineVm,
    #[host(rename = "machine-unavailable")]
    MachineUnavailable,
    /// The stage stays, and the reply is skipped.
    #[host(rename = "wait")]
    Wait,
    /// The stage stays, and a reply is allowed.
    #[host(rename = "uncertain")]
    Uncertain,
}

/// Why a prediction was published or why the run completed.
#[crate::host(record(module = "amd", name = "AMDReason", emit, extract, enum))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdReason {
    /// A model prediction was released normally.
    #[host(rename = "prediction")]
    Prediction,
    /// An empty turn reused the latest prediction without a new request.
    #[host(rename = "reused")]
    Reused,
    /// The turn's inference deadline passed, so the prediction is uncertain in the current
    /// stage. Repeated inference timeouts can also complete the run.
    #[host(rename = "inference_timeout")]
    InferenceTimeout,
    /// Classification failed or returned unusable output; the prediction is uncertain in the
    /// current stage.
    #[host(rename = "inference_error")]
    InferenceError,
    /// An internal task or prediction handler failed, ending the run.
    #[host(rename = "internal_error")]
    InternalError,
    /// AMD reached a human or machine-unavailable category.
    #[host(rename = "finished")]
    Finished,
    /// The limit of consecutive uncertain predictions was reached.
    #[host(rename = "max_uncertain_turns")]
    MaxUncertainTurns,
    /// The time limit since listening started was reached.
    #[host(rename = "timeout")]
    Timeout,
    /// The call stayed silent and inactive for the current stage's idle interval.
    #[host(rename = "idle_timeout")]
    IdleTimeout,
    /// The caller closed AMD or its session before detection completed.
    #[host(rename = "cancelled")]
    Cancelled,
    /// The target participant or audio track was unavailable during listening setup.
    #[host(rename = "participant_missing")]
    ParticipantMissing,
    /// The target participant disconnected before detection completed.
    #[host(rename = "participant_disconnected")]
    ParticipantDisconnected,
    /// The session switched agents during AMD.
    #[host(rename = "agent_changed")]
    AgentChanged,
}

/// Where one AMD run is: `initialized` until entered, `pending` until it listens, `active`
/// while it listens, `finished` once it completed.
#[crate::host(record(module = "amd", name = "AMDLifecycle", emit, extract, enum))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdLifecycle {
    #[host(rename = "initialized")]
    Initialized,
    #[host(rename = "pending")]
    Pending,
    #[host(rename = "active")]
    Active,
    #[host(rename = "finished")]
    Finished,
}

/// `amd_prediction`: at most one per committed turn.
#[crate::host(record(
    module = "amd",
    name = "AMDPredictionEvent",
    pydantic,
    emit,
    extract,
    constant(type = "amd_prediction")
))]
#[python(property(
    name = "is_human",
    ret = "bool",
    core = "_core_amd_prediction_is_human",
    doc = "The prediction is ``human``."
))]
#[python(property(
    name = "is_machine",
    ret = "bool",
    core = "_core_amd_prediction_is_machine",
    doc = "The prediction is one of the four machine categories."
))]
#[derive(Debug, Clone, PartialEq)]
pub struct AmdPredictionEvent {
    /// Participant speech in this turn, in seconds.
    pub speech_duration: f64,
    /// Prediction for this turn. ``uncertain`` and ``wait`` keep the current stage.
    pub category: AmdCategory,
    pub reason: AmdReason,
    /// The session transcript, else AMD's transcript.
    pub transcript: String,
    /// Time from commit to this event, in seconds.
    pub delay: f64,
    /// AMD's turn id: local to the run, from 1. Ids can skip a turn whose request a newer turn
    /// cancelled.
    #[host(default = 0)]
    pub turn_id: u64,
    /// Stage after this prediction. Never ``wait``.
    #[host(default = "uncertain")]
    pub stage: AmdCategory,
    /// Model time for ``prediction`` events, else ``None``.
    #[host(default = None)]
    pub inference_duration: Option<f64>,
}

/// `amd_completed`: once per run, after cleanup.
#[crate::host(record(
    module = "amd",
    name = "AMDCompletedEvent",
    pydantic,
    emit,
    extract,
    constant(type = "amd_completed")
))]
#[python(property(
    name = "is_error",
    ret = "bool",
    core = "_core_amd_completed_is_error",
    doc = "Whether AMD stopped without detecting the call, for example after an internal error\nor a missing participant. Timeouts, idle completions, and a participant\ndisconnect are normal outcomes; read ``category`` for those."
))]
#[derive(Debug, Clone, PartialEq)]
pub struct AmdCompletedEvent {
    /// The call's category. Once the voicemail message has played, a goodbye that
    /// ends the box does not make it ``machine-unavailable``: the category is the stage
    /// before it, ``machine-vm``, or ``machine-ivr`` after a post-message menu.
    pub category: AmdCategory,
    pub reason: AmdReason,
    /// The voicemail message played to its end while AMD was active.
    #[host(default = false)]
    pub voicemail_message_played: bool,
    /// The last classified turn's prediction; None when the run ended before one.
    #[host(default = None)]
    pub last_prediction: Option<AmdPredictionEvent>,
}

/// One option of an observed IVR menu.
#[crate::host(record(module = "amd", name = "IVRMenuOption", pydantic, emit, extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvrMenuOption {
    pub label: String,
    #[host(default = "")]
    pub dtmf: String,
    #[host(default = "")]
    pub spoken_response: String,
}

/// Best-effort observation. This event never authorizes an action.
#[crate::host(record(
    module = "amd",
    name = "AMDMenuObservedEvent",
    pydantic,
    emit,
    extract,
    constant(type = "amd_menu_observed")
))]
#[derive(Debug, Clone, PartialEq)]
pub struct AmdMenuObservedEvent {
    pub session_id: String,
    pub turn_id: u64,
    pub menu: String,
    #[host(default_factory = "list")]
    pub options: Vec<IvrMenuOption>,
    /// Model time of the extraction, in seconds.
    pub extraction_duration: f64,
}

/// What a run tells its host, in order: each prediction and menu, then the completion once.
#[crate::host(record(module = "amd", emit, transparent))]
#[derive(Debug, Clone, PartialEq)]
pub enum AmdEvent {
    Prediction(AmdPredictionEvent),
    MenuObserved(AmdMenuObservedEvent),
    Completed(AmdCompletedEvent),
}

/// `AMD(...)`'s behavior options, with their defaults. Model and participant selection are the
/// constructor's own arguments.
#[crate::host(record(module = "amd", name = "AMDOptions", dataclass, extract, kwargs))]
#[derive(Debug, Clone, PartialEq)]
pub struct AmdOptions {
    /// Discard pre-answer audio from AMD and the session. When False, listen to subscribed SIP
    /// early media.
    #[host(default = true)]
    pub wait_until_answered: bool,
    /// Instructions for each screening turn. None completes AMD when the call enters
    /// screening, without a reply.
    #[host(
        default = "Call state: automated call screening. Answer the screening assistant's latest prompt briefly. Use what you know about who you are and why you are calling. Then wait for its next prompt."
    )]
    pub screening_instructions: Option<String>,
    /// Instructions for one successfully played voicemail message. None completes AMD at the
    /// first voicemail prediction, without a message.
    #[host(
        default = "Call state: voicemail. Deliver one concise, self-contained message. State who you are, why you are calling, and the next step for the recipient."
    )]
    pub voicemail_instructions: Option<String>,
    /// Instructions for each IVR turn. The SDK supplies a DTMF tool only to IVR reply
    /// generations. None completes AMD when the call enters IVR, without a reply or menu
    /// extraction.
    #[host(
        default = "Call state: an automated phone menu. Use the participant's actual prompt and the purpose of this call to choose the next step. Use send_dtmf_events for an explicit keypad choice, or give a short spoken answer when requested. Do not invent a menu option. After the action, wait for the next prompt."
    )]
    pub ivr_instructions: Option<String>,
    /// Temporary instructions for the first human turn after a machine stage. Skipped if no
    /// machine stage preceded the human. None keeps the Agent's own instructions.
    #[host(
        default = "Call state: a human has answered. The latest participant turn is from the human, not the automated system. Earlier automated prompts no longer apply. Respond to the human's latest message and continue the call normally. Do not resume a response to an earlier automated prompt."
    )]
    pub human_instructions: Option<String>,
    /// Fixed text that the agent says first, when AMD starts listening. Set at most one of
    /// opening_text and opening_instructions.
    #[host(default = None)]
    pub opening_text: Option<String>,
    /// Instructions for a reply that the agent generates and says first, when AMD starts
    /// listening. The reply cannot call tools.
    #[host(default = None)]
    pub opening_instructions: Option<String>,
    /// Silent, inactive time before AMD completes outside voicemail, in seconds.
    #[host(default = 10.0)]
    pub idle_timeout: f64,
    /// Silent, inactive time in voicemail, in seconds.
    #[host(default = 10.0)]
    pub voicemail_idle_timeout: f64,
    /// Hard limit from the start of listening, in seconds.
    #[host(default = 60.0)]
    pub timeout: f64,
    /// Prediction deadline per committed turn, in seconds. Late results are ignored.
    #[host(default = 2.0)]
    pub inference_timeout: f64,
    /// Continuous participant silence before a machine reply plays, in seconds. Zero disables
    /// the extra silence wait.
    #[host(default = 1.5)]
    pub machine_silence_threshold: f64,
    /// Consecutive uncertain predictions before completion with the current stage.
    #[host(default = 3)]
    pub max_uncertain_turns: i64,
    /// Prediction timeouts before completion. A valid prediction resets the count.
    #[host(default = 3)]
    pub max_inference_timeouts: i64,
}

/// One classifier or STT model a run uses, as the host resolved it.
#[crate::host(record(module = "amd", extract))]
#[derive(Clone)]
pub struct AmdLlm {
    pub model: Arc<dyn Llm>,
    /// The model reaches only LiveKit Inference: its requests carry AMD's client feature header.
    pub inference: bool,
}

/// A decision model AMD classifies with, as the host resolved it.
#[crate::host(record(module = "amd", extract))]
#[derive(Clone)]
pub struct AmdDecision {
    pub model: Arc<dyn AmdDecisionModel>,
    /// The model reaches only LiveKit Inference: its requests carry AMD's client feature header.
    pub inference: bool,
}

/// AMD's own streaming STT, as the host resolved it.
#[crate::host(record(module = "amd", extract))]
#[derive(Clone)]
pub struct AmdStt {
    pub model: Arc<dyn Stt>,
    /// The model reaches only LiveKit Inference: its connections carry AMD's client feature
    /// header.
    pub inference: bool,
}

/// What AMD classifies with: an LLM, which also reads menus, or a decision model.
#[crate::host(record(module = "amd", extract, transparent))]
#[derive(Clone)]
pub enum AmdClassifier {
    Llm(AmdLlm),
    Decision(AmdDecision),
}

/// What `AMD.__aenter__` hands the session: everything one run needs beyond the session.
/// The host closes the models it created once the run completed.
#[crate::host(record(module = "amd", extract))]
#[derive(Clone)]
pub struct AmdEntry {
    pub options: AmdOptions,
    pub classifier: AmdClassifier,
    #[host(default = None)]
    pub stt: Option<AmdStt>,
    /// The participant to listen to; `None` listens to the one the session's room I/O links.
    #[host(default = None)]
    pub participant: Option<String>,
    /// AMD's built-in `send_dtmf_events`, which the host runs.
    pub dtmf_tool: ToolSpec,
    /// The session's `llm_conn_options`: classifier and menu requests use them.
    pub llm_conn_options: ApiConnectOptions,
    /// The session's `stt_conn_options`: AMD's STT stream uses them.
    pub stt_conn_options: ApiConnectOptions,
}

/// Why a run did not attach: `AMD.__aenter__`'s errors, in its words.
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AmdEnterRefused {
    /// The session cannot run AMD as it is: a handoff, an `AgentTask` or another run.
    #[error("{0}")]
    Refused(String),
    /// AMD's options or models do not fit the session: a realtime model without client-side
    /// turn detection, per-response tools or STT, or a non-streaming STT.
    #[error("{0}")]
    #[host(kind = "ValueError")]
    Invalid(String),
    #[error(transparent)]
    #[host(delegate)]
    Session(#[from] crate::voice::session::SessionRefused),
}

/// One `evaluate` call with a single `Choice` decision.
#[crate::host(record(module = "amd", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct AmdChoiceRequest {
    /// `evaluate(input=...)`: the request state and the transcript.
    pub input: String,
    /// The decision's name.
    pub name: String,
    /// The choice's prompt.
    pub prompt: String,
    /// The choice's options, in order. They carry no descriptions: a short definition next to a
    /// choice outweighed the prompt's conditions.
    pub options: Vec<String>,
    pub conn_options: ApiConnectOptions,
    /// `extra_headers`.
    pub extra_headers: std::collections::BTreeMap<String, String>,
}

/// A host decision model (`llm.DecisionModel.evaluate` with one `Choice`), reached through
/// `voice.amd`'s glue, which builds the decision and reads its result.
#[crate::host(proxy(host = "livekit.agents.llm.DecisionModel", owns))]
pub trait AmdDecisionModel: Send + Sync {
    /// The chosen option, or `None` when the response holds no choice result for the decision.
    #[host(await, function = "livekit.agents.voice.amd._choose")]
    fn choose<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        request: &'a AmdChoiceRequest,
    ) -> crate::BoxFuture<'a, Result<Option<String>, ProviderError>>;

    /// `provider/model`, for logs.
    #[host(call, function = "livekit.agents.voice.amd._decision_label")]
    fn label(&self) -> Result<String, crate::errors::HostError>;
}

/// One AMD run, from its attach to its outcome. Dropping it does not cancel the run.
#[crate::host(class(name = "AmdRun"))]
pub trait AmdRun: Send + Sync + Sized + 'static {
    /// Where the run is: `pending` until it listens, then `active`, then `finished`.
    #[host(getter)]
    fn lifecycle(&self) -> AmdLifecycle;

    /// The run's next event, in order; `None` once the run's events ended. One reader.
    #[host(await)]
    fn next_event(&self) -> crate::BoxFuture<'static, Option<AmdEvent>>;

    /// `execute()`: the completion, once the session detached and the legs closed. Every caller
    /// gets the same one; a caller that stops waiting does not cancel the run.
    #[host(await)]
    fn execute(&self) -> crate::BoxFuture<'static, AmdCompletedEvent>;

    /// `aclose()`: completes the run with `cancelled` if it is still running, and returns its
    /// completion.
    #[host(await)]
    fn aclose(&self) -> crate::BoxFuture<'static, AmdCompletedEvent>;

    /// A prediction or menu listener in the host raised: the run completes with
    /// `internal_error`.
    #[host(call)]
    fn listener_failed(&self);
}

/// AMD's pure rules the host's types read.
#[crate::host(functions)]
pub trait AmdRules {
    /// `AMD(...)`'s option checks, in its order: the `ValueError` it raises.
    #[host(function, name = "_core_amd_check_options")]
    fn check_options(options: AmdOptions) -> Result<(), Invalid>;

    /// `AMDPredictionEvent.is_human`.
    #[host(function, name = "_core_amd_prediction_is_human")]
    fn prediction_is_human(event: AmdPredictionEvent) -> bool;

    /// `AMDPredictionEvent.is_machine`.
    #[host(function, name = "_core_amd_prediction_is_machine")]
    fn prediction_is_machine(event: AmdPredictionEvent) -> bool;

    /// `AMDCompletedEvent.is_error`.
    #[host(function, name = "_core_amd_completed_is_error")]
    fn completed_is_error(event: AmdCompletedEvent) -> bool;

    /// Whether AMD may pick its own Inference models: `LIVEKIT_URL` names LiveKit Cloud and an
    /// Inference key pair resolves (`LIVEKIT_INFERENCE_API_KEY`/`_SECRET`, else the server's
    /// `LIVEKIT_API_KEY`/`_SECRET`).
    #[host(function, name = "_core_amd_auto_models")]
    fn auto_models() -> bool;
}
