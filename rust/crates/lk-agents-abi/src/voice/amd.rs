//! The AMD verdict. [`AmdCategory`] mirrors a `(str, Enum)`, so `ev.category == "machine-vm"` works;
//! a `#[pyclass]` enum would compare `False` silently.

use std::time::Duration;

use lk_agents_host_derive::Host;

// `classifier.py:28`. The wire strings are public API (enum values, an OTel attribute, the job
// tagger key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[crate::host(record(
    module = "amd",
    path = "livekit.agents.voice.amd",
    name = "AMDCategory",
    emit,
    extract,
    enum
))]
#[crate::host(wire_strs(msg = format!("unknown amd category {s:?}")))]
pub enum AmdCategory {
    #[host(rename = "human")]
    Human,
    #[host(rename = "machine-ivr")]
    MachineIvr,
    #[host(rename = "machine-vm")]
    MachineVm,
    #[host(rename = "machine-unavailable")]
    MachineUnavailable,
    #[host(rename = "uncertain")]
    Uncertain,
}

// `from_wire` also parses the LLM's `save_prediction(label=...)`; `None` is the reference's
// swallowed coercion error: no verdict.
impl AmdCategory {
    /// `classifier.py:45`.
    pub fn is_human(self) -> bool {
        matches!(self, AmdCategory::Human)
    }

    /// `classifier.py:49`. `Uncertain` is neither human nor machine; `detector.py` relies on that.
    pub fn is_machine(self) -> bool {
        matches!(
            self,
            AmdCategory::MachineIvr | AmdCategory::MachineVm | AmdCategory::MachineUnavailable
        )
    }
}

/// A closed set of the reference's five call sites. The wire string, not the variant name, crosses:
/// public API (`ATTR_AMD_REASON`, the job tagger).
// `from_wire` is `None` rather than a fallback: `reason` is public API.
#[crate::host(string)]
#[crate::host(wire_strs(msg = format!("unknown amd reason {s:?}")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdReason {
    /// `start_detection_timer`.
    DetectionTimeout,
    /// The listening window opened and no speech and no transcript ever arrived.
    NoSpeechTimeout,
    /// Short utterance, no transcript: the pre-baked human verdict.
    ShortGreeting,
    Llm,
    /// `detector.py:467`.
    ParticipantMissing,
}

// `classifier.py:36`. Also the `amd_result` session event.
#[crate::host(record(
    module = "amd",
    name = "AMDPredictionEvent",
    emit,
    extract,
    constant(type = "amd_prediction")
))]
#[python(property(name = "is_human", ret = "bool"), property(name = "is_machine", ret = "bool"))]
#[derive(Debug, Clone, PartialEq)]
pub struct AmdPrediction {
    pub speech_duration: Duration,
    pub category: AmdCategory,
    // `str` on the host: narrowing the reference's free-form `str` would be an API change.
    pub reason: AmdReason,
    // Empty for every fallback verdict (`classifier.py:373`), so only an `llm` verdict can be
    // `machine-ivr` downstream.
    pub transcript: String,
    // Zero when speech never ended.
    pub delay: Duration,
}

impl AmdPrediction {
    pub fn is_human(&self) -> bool {
        self.category.is_human()
    }

    pub fn is_machine(&self) -> bool {
        self.category.is_machine()
    }
}

/// `AMD_PROMPT` (`classifier.py:58`), verbatim: a tuned prompt. Overridable via `DetectionOptions`.
pub const AMD_PROMPT: &str = r#"Task:
Classify the call greeting transcript into exactly one of these categories:

human: A person answered (e.g., "Hello?", "This is John.").
machine-ivr: A prompt to press a key (e.g., "Press 1 to continue").
machine-vm: A voicemail greeting where leaving a message IS possible.
machine-unavailable: Any greeting indicating it's NOT possible to leave message, eg because mailbox is full, not setup, etc.
uncertain: For partial transcripts that are ambiguous.

Examples:
Input: "The person you called has a voice mailbox that hasn't been set up yet. Goodbye."
Output: machine-unavailable

Input: "Thank you for calling Truly Pizza in Dana Pointe. Our hours of operation are 11AM to 8PM, Sunday through Thursday, 11AM to 9PM, Friday and Saturday, and we're closed on Tuesdays."
Output: uncertain

Input: "You for calling Truly Pizza in Dana Pointe. Our hours of operation are 11AM to 8PM, Sunday through Thursday, 11AM to 9PM, Friday and Saturday, and we're closed on Tuesdays. If you'd like to place an order, please press 1 or head to our website to order online for pickup and local delivery."
Output: machine-ivr

Input: "Please state your name and why you're calling, and I will check if the person is available"
Output: machine-ivr
Note: this should apply for any call screening prompts.

Input: "I'm away from my desk. If you leave a message, I will get back to you."
Output: machine-vm

Input: "Hello, this is Lisa."
Output: human"#;

// `voice/amd/detector.py`'s constants; the classifier's are `lk_agents_voice::amd::classifier`'s.

/// `detector.py:165`.
pub const DEFAULT_LLM_MODEL: &str = "google/gemini-3.1-flash-lite";
pub const DEFAULT_STT_MODEL: &str = "cartesia/ink-whisper";
/// How long `_setup` waits for the callee's audio track before settling `participant_missing`.
pub const TRACK_PUBLICATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
pub const SIP_CALL_STATUS_ATTR: &str = "sip.callStatus";
pub const SIP_CALL_STATUS_ACTIVE: &str = "active";

/// `EVALUATED_LLM_MODELS` (`detector.py:44`).
pub const EVALUATED_LLM_MODELS: &[&str] = &[
    "google/gemini-3.1-flash-lite",
    "google/gemini-3-flash-preview",
    "openai/gpt-4.1",
    "openai/gpt-5.2",
    "openai/gpt-5.4",
    "openai/gpt-5.1",
    "openai/gpt-4o",
    "openai/gpt-5.1-chat-latest",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-5.2-chat-latest",
    "google/gemini-2.5-flash-lite",
];

/// `EVALUATED_STT_MODELS`.
pub const EVALUATED_STT_MODELS: &[&str] =
    &["deepgram/nova-3", "assemblyai/universal-streaming-multilingual", "cartesia/ink-whisper"];

/// The detector's constants, as one dict so a host cannot pick up half of them.
#[crate::host(record(module = "amd", emit, typed_dict))]
pub struct AmdConstants {
    pub default_llm_model: String,
    pub default_stt_model: String,
    pub prompt: String,
    pub evaluated_llm_models: Vec<String>,
    pub evaluated_stt_models: Vec<String>,
    pub human_speech_threshold: f64,
    pub human_silence_threshold: f64,
    pub machine_silence_threshold: f64,
    pub no_speech_threshold: f64,
    pub detection_timeout: f64,
    pub max_endpointing_delay: f64,
    /// The extension budget: `MAX_EXTENSIONS` requests, each clamped to `MAX_EXTENSION_SECS`.
    pub max_extensions: u32,
    pub max_extension_secs: f64,
}

/// The detector's reference constants and its endpointing rule.
#[crate::host(functions(pid = false))]
pub trait AmdFunctions {
    /// These are the reference's, not tuning knobs: the default models are what the benchmark
    /// ran on.
    #[host(function)]
    fn amd_constants() -> AmdConstants;
    /// `_resolve_classifier`'s `max_endpointing_delay`: explicit option, else the activity's,
    /// else the default. `provided` is `None` only when the key was absent: an explicit default
    /// still beats a disagreeing activity.
    #[host(function)]
    fn amd_resolve_max_endpointing_delay(
        provided: Option<Duration>,
        activity: Option<Duration>,
    ) -> Duration;
    /// `AMDPredictionEvent.is_human`.
    #[host(function)]
    fn amd_prediction_is_human(prediction: AmdPrediction) -> bool;
    /// `AMDPredictionEvent.is_machine`.
    #[host(function)]
    fn amd_prediction_is_machine(prediction: AmdPrediction) -> bool;
}

// --- the user's tuning, as the host takes it ---------------------------------------------------

// `voice/amd/detector.py`: per-`AMD` overrides. A `typing` `TypedDict` (not `typing_extensions`).
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    typed_dict_from = "typing",
    module = "amd",
    path = "livekit.agents.voice.amd",
    name = "DetectionOptions"
)]
pub struct AmdDetectionOptions {
    pub human_speech_threshold: Option<f64>,
    pub human_silence_threshold: Option<f64>,
    pub machine_silence_threshold: Option<f64>,
    pub no_speech_threshold: Option<f64>,
    pub timeout: Option<f64>,
    pub max_endpointing_delay: Option<f64>,
    pub prompt: Option<String>,
}

// --- the machine's config, as the host hands it over ------------------------------------------

/// With AMD's own STT, session transcripts are dropped, or the greeting is seen twice.
#[crate::host(wire_strs(msg = format!("unknown AMD transcript source {s:?}: expected stt or amd_stt")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TranscriptSource {
    /// AMD's default when it was not given an `stt=`.
    #[host(rename = "stt")]
    SessionStt,
    AmdStt,
}

#[cfg(feature = "python")]
crate::python::from_host_str!(TranscriptSource);

/// `AMD._run`'s config dict (`voice/amd/__init__.py`). Every key defaults to the classifier's own,
/// resolved and validated by `AmdConfig::resolve`.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[binding(extract, from_items, default)]
pub struct AmdConfigKwargs {
    pub human_speech_threshold: Option<f64>,
    pub human_silence_threshold: Option<f64>,
    pub machine_silence_threshold: Option<f64>,
    pub no_speech_threshold: Option<f64>,
    pub detection_timeout: Option<f64>,
    /// Already resolved by the host (`amd_resolve_max_endpointing_delay`).
    pub max_endpointing_delay: Option<f64>,
    /// `"stt"` for the session's finals, `"amd_stt"` for a recogniser of AMD's own.
    pub source: Option<TranscriptSource>,
    pub wait_until_finished: bool,
    pub interrupt_on_machine: bool,
    pub ivr_detection: bool,
    /// `DetectionOptions.prompt`; `None` is [`AMD_PROMPT`].
    pub prompt: Option<String>,
}

/// Runtime-owned answering-machine detector attached to a live session.
pub trait AmdPort: Send + Sync {
    fn execute(&self);
    fn close(&self);
    fn listening(&self) -> bool;
}
