//! The AMD verdict. [`AmdCategory`] mirrors a `(str, Enum)`, so `ev.category == "machine-vm"` works;
//! a `#[pyclass]` enum would compare `False` silently.

use std::time::Duration;

use lk_agents_host_derive::Host;

// `classifier.py:28`. The wire strings are public API (enum values, an OTel attribute, the job
// tagger key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Host)]
#[python(enum, module = "amd", path = "livekit.agents.voice.amd.classifier", name = "AMDCategory")]
#[host(wire(msg = format!("unknown amd category {s:?}")))]
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

/// A closed set of the reference's five call sites.
// `from_wire` is `None` rather than a fallback: `reason` is public API.
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

// The wire string, not the variant name: public API (`ATTR_AMD_REASON`, the job tagger).
#[cfg(feature = "python")]
crate::python::into_host_str!(AmdReason, |v| v.as_str());

// `classifier.py:36`.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(
    pydantic,
    module = "amd",
    path = "livekit.agents.voice.amd.classifier",
    name = "AMDPredictionEvent",
    tag = "type",
    tag_value = "amd_prediction",
    property(name = "is_human", ret = "bool"),
    property(name = "is_machine", ret = "bool")
)]
// The `amd_result` session event, a wire kind distinct from the model's `amd_prediction` tag.
#[binding(emit, tag = "type", tag_value = "amd_result")]
pub struct AmdPrediction {
    #[python(ty = "float")]
    pub speech_duration: Duration,
    pub category: AmdCategory,
    // `str` on the host: narrowing the reference's free-form `str` would be an API change.
    #[python(ty = "str")]
    pub reason: AmdReason,
    // Empty for every fallback verdict (`classifier.py:373`), so only an `llm` verdict can be
    // `machine-ivr` downstream.
    pub transcript: String,
    // Zero when speech never ended.
    #[python(ty = "float")]
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

/// The detector's constants, as the sixteen keys the binding's `amd_constants` answers with.
#[derive(Host)]
#[binding(emit)]
pub struct AmdConstants {
    pub default_llm_model: String,
    pub default_stt_model: String,
    pub track_publication_timeout: f64,
    pub sip_call_status_attr: String,
    pub sip_call_status_active: String,
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

// --- the user's tuning, as the host takes it ---------------------------------------------------

// `voice/amd/detector.py`: per-`AMD` overrides. A `typing` `TypedDict` (not `typing_extensions`).
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    typed_dict_from = "typing",
    module = "amd",
    path = "livekit.agents.voice.amd.detector",
    name = "DetectionOptions"
)]
#[node(interface)]
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

#[cfg(test)]
mod category_tests {
    use super::{AmdCategory, AmdReason};

    // Catches: `uncertain` folded into one side, or an unknown category parsed as a verdict.
    #[test]
    fn uncertain_is_neither_human_nor_machine() {
        let uncertain = AmdCategory::from_wire("uncertain").unwrap();
        assert!(!uncertain.is_human() && !uncertain.is_machine());
        assert!(AmdCategory::from_wire("machine-vm").unwrap().is_machine());
        assert!(AmdCategory::from_wire("human").unwrap().is_human());
        assert_eq!(AmdCategory::from_wire("robot"), None);
        assert_eq!(AmdReason::from_wire("no_speech_timeout"), Some(AmdReason::NoSpeechTimeout));
        assert_eq!(AmdReason::from_wire("Llm"), None, "variant names are not the wire form");
        assert_eq!(AmdReason::from_wire("because"), None, "a settle needs a reason the wire knows");
    }
}
