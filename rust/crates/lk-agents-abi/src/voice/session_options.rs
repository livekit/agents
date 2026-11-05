//! The `AgentSession` option seam: kwargs as written, legs as stated, the resolved configuration.
//! The policy lives in `lk-agents-voice/src/agent_activity/options.rs` and `turn.rs`.

use crate::Maybe;
use crate::Modality;
use crate::tts::markup::SpeechSteering;
use lk_agents_host_derive::Vocabulary;
use std::fmt;

/// A public literal nobody recognises; `ValueError` carries the message verbatim.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown {what} {value:?}; expected {expected}")]
pub struct UnknownLiteral {
    pub what: &'static str,
    pub value: String,
    pub expected: &'static str,
}

/// A `tts_text_transforms` entry that names no builtin. Refused, not skipped.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "unknown tts_text_transforms entry {0:?}; use \"filter_markdown\" / \"filter_emoji\", or apply \
     your own transform in a tts_node"
)]
pub struct UnknownTextTransform(pub String);

/// `Auto` is the reference's `None`: decide from the legs.
// `Auto` is `"unset"`. An unknown literal is refused, never read as `Auto`.
#[crate::host(wire_strs(err = UnknownLiteral, msg = UnknownLiteral {
    what: "turn_detection mode",
    value: s.to_owned(),
    expected: "\"stt\", \"vad\", \"manual\", \"realtime_llm\" or \"unset\"",
}))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Vocabulary)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub enum TurnDetection {
    Vad,
    Stt,
    Manual,
    RealtimeLlm,
    #[host(rename = "unset")]
    Auto,
}

/// Picks the endpointing base and the turn machine's countdown arm.
#[crate::host(record(module = "session_options", extract))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Vocabulary)]
pub enum DetectorKind {
    Streaming,
}

/// Which builtin transforms run between the segmenter and the TTS.
#[crate::host(codec(
    extract,
    host = "collections.abc.Sequence[livekit.agents.voice.agent_session.TextTransforms]",
    via = Vec<String>
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct TextTransformSet {
    pub filter_markdown: bool,
    pub filter_emoji: bool,
}

impl TextTransformSet {
    /// `DEFAULT_TTS_TEXT_TRANSFORMS = ["filter_markdown", "filter_emoji"]`.
    pub const ALL: TextTransformSet =
        TextTransformSet { filter_markdown: true, filter_emoji: true };

    /// The reference's list form as the set; an unknown name is refused.
    pub fn from_names(names: &[String]) -> Result<Self, UnknownTextTransform> {
        let mut set = TextTransformSet { filter_markdown: false, filter_emoji: false };
        for name in names {
            match name.as_str() {
                "filter_markdown" => set.filter_markdown = true,
                "filter_emoji" => set.filter_emoji = true,
                other => return Err(UnknownTextTransform(other.to_owned())),
            }
        }
        Ok(set)
    }

    pub fn names(self) -> Vec<String> {
        let TextTransformSet { filter_markdown, filter_emoji } = self;
        [("filter_markdown", filter_markdown), ("filter_emoji", filter_emoji)]
            .into_iter()
            .filter(|(_, on)| *on)
            .map(|(name, _)| name.to_owned())
            .collect()
    }
}

impl TryFrom<Vec<String>> for TextTransformSet {
    type Error = UnknownTextTransform;
    fn try_from(names: Vec<String>) -> Result<Self, Self::Error> {
        Self::from_names(&names)
    }
}

/// ``"fixed"`` for fixed delay, ``"dynamic"`` for dynamic delay.
#[crate::host(record(module = "turn", extract))]
#[crate::host(wire_strs(err = UnknownLiteral, msg = UnknownLiteral {
    what: "endpointing mode",
    value: s.to_owned(),
    expected: "\"fixed\" or \"dynamic\"",
}))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub enum EndpointingMode {
    Fixed,
    Dynamic,
}

/// ``"adaptive"`` for ML-based detection, ``"vad"`` for simple voice-activity detection.
#[crate::host(record(module = "turn", extract, emit))]
#[crate::host(wire_strs(err = UnknownLiteral, msg = UnknownLiteral {
    what: "interruption mode",
    value: s.to_owned(),
    expected: "\"adaptive\" or \"vad\"",
}))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub enum InterruptionMode {
    Adaptive,
    Vad,
}

/// `interruption["backchannel_boundary"]`: a scalar applies to both edges, a pair keeps them apart.
#[crate::host(record(module = "session_options", extract, transparent))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackchannelBoundary {
    Both(f64),
    Split((f64, f64)),
}

impl BackchannelBoundary {
    pub fn start(self) -> f64 {
        match self {
            BackchannelBoundary::Both(v) => v,
            BackchannelBoundary::Split((s, _)) => s,
        }
    }

    pub fn end(self) -> f64 {
        match self {
            BackchannelBoundary::Both(v) => v,
            BackchannelBoundary::Split((_, e)) => e,
        }
    }
}

/// `SessionConnectOptions`: each leg's retry budget and timeout; not given (`None`) is the
/// provider's own policy, else the defaults.
#[crate::host(record(
    module = "connect", name = "SessionConnectOptions",
    path = "livekit.agents.voice.agent_session", dataclass,
    host_fields(max_unrecoverable_errors(
        ty = "int", default = ".session_wire.DEFAULT_MAX_UNRECOVERABLE_ERRORS",
        doc = "Maximum number of consecutive unrecoverable errors from STT, LLM, or TTS."
    )),
    boundary, extract, default_on_null,
    validate = crate::voice::session_options::SessionConnOptions::validate
))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SessionConnOptions {
    #[host(rename = "stt_conn_options", default = None)]
    pub stt: Option<crate::ApiConnectOptions>,
    #[host(rename = "llm_conn_options", default = None)]
    pub llm: Option<crate::ApiConnectOptions>,
    #[host(rename = "tts_conn_options", default = None)]
    pub tts: Option<crate::ApiConnectOptions>,
}

/// Of a detector object only whether it streams crosses.
#[crate::host(codec(
    extract,
    host = "livekit.agents.voice.io.TurnDetectionMode | None",
    py = "py_values::turn_detection_kwarg"
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetectionKwarg {
    Mode(TurnDetection),
    Detector { streaming: bool },
    Null,
}

impl TurnDetectionKwarg {
    /// The leaf this value contributes to a patch; an object or `None` leaves it to the legs.
    pub fn detection(self) -> Option<TurnDetection> {
        match self {
            TurnDetectionKwarg::Mode(m) => Some(m),
            TurnDetectionKwarg::Detector { streaming: _ } | TurnDetectionKwarg::Null => None,
        }
    }
}

// In these groups a `///` is the reference's `voice/turn.py` docstring verbatim; Rust notes are
// `//`. Every key is optional: `Option` with `default = None` reads absent, `None` and
// `NOT_GIVEN` alike as unset; `Maybe<Option<_>>` keeps an explicit `None` ("off") apart.

/// Configuration for endpointing.
///
/// All keys are optional. Missing keys inherit from the session default
/// (at the ``Agent`` level) or use the documented defaults
/// (at the ``AgentSession`` level).
#[crate::host(record(
    module = "turn",
    name = "EndpointingOptions",
    path = "livekit.agents.voice.io",
    typed_dict,
    extract
))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EndpointingKwargs {
    /// Endpointing mode. ``"fixed"`` for fixed delay, ``"dynamic"`` for dynamic delay. Defaults to ``"fixed"``.
    #[host(default = None)]
    pub mode: Option<EndpointingMode>,
    /// Minimum time (s) since last detected speech before declaring the
    /// user's turn complete. Defaults to ``0.5``.
    #[host(default = None)]
    pub min_delay: Option<f64>,
    /// Maximum time (s) the agent waits before terminating the turn.
    /// Defaults to ``3.0``.
    #[host(default = None)]
    pub max_delay: Option<f64>,
    /// Exponential moving average coefficient for dynamic endpointing.
    /// The higher the value, the more weight is given to the history.
    /// Defaults to ``0.9``. Only applies when mode is ``dynamic``.
    #[host(default = None)]
    pub alpha: Option<f64>,
}

/// Configuration for interruption handling.
///
/// All keys are optional. Missing keys inherit from the session default
/// (at the ``Agent`` level) or use the documented defaults
/// (at the ``AgentSession`` level).
///
/// ``mode`` absent means the session picks the best available strategy.
#[crate::host(record(
    module = "turn",
    name = "InterruptionOptions",
    path = "livekit.agents.voice.io",
    typed_dict,
    extract
))]
#[derive(Clone, Copy, Default, PartialEq)]
pub struct InterruptionKwargs {
    /// Whether interruptions are enabled. Defaults to ``True``.
    #[host(default = None)]
    pub enabled: Option<bool>,
    /// Interruption handling strategy. ``"adaptive"`` for ML-based
    /// detection, ``"vad"`` for simple voice-activity detection.
    /// Absent means auto-detect.
    //
    // A given `None` is kept apart: `update_options` refuses the construction-only key either way.
    pub mode: Maybe<Option<InterruptionMode>>,
    /// Drop buffered audio while the agent speaks and cannot be
    /// interrupted. Defaults to ``True``.
    #[host(default = None)]
    pub discard_audio_if_uninterruptible: Option<bool>,
    /// Minimum speech length (s) to register as an interruption.
    /// Defaults to ``0.5``.
    #[host(default = None)]
    pub min_duration: Option<f64>,
    /// Minimum word count to consider an interruption (STT only).
    /// Defaults to ``0``.
    #[host(default = None)]
    pub min_words: Option<i64>,
    /// Resume the agent's speech after a false interruption.
    /// Defaults to ``True``.
    #[host(default = None)]
    pub resume_false_interruption: Option<bool>,
    /// Seconds of silence after an interruption before it is
    /// classified as false. ``None`` disables. Defaults to ``2.0``.
    pub false_interruption_timeout: Maybe<Option<f64>>,
    /// Seconds near the start/end of each agent turn during which overlapping
    /// speech classified as a backchannel by the adaptive detector is suppressed
    /// (events flagged as interruptions still pass through). Use a tuple to apply
    /// different values for start and end separately. ``None`` disables. Defaults
    /// to ``(1.0, 1.0)``. End value accounts for STT transcript timestamp
    /// inaccuracy.
    pub backchannel_boundary: Maybe<Option<BackchannelBoundary>>,
}

/// Configuration for preemptive generation.
#[crate::host(record(
    module = "turn",
    name = "PreemptiveGenerationOptions",
    path = "livekit.agents.voice.io",
    typed_dict,
    extract
))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PreemptiveKwargs {
    /// Whether preemptive generation is enabled. Defaults to ``True``.
    #[host(default = None)]
    pub enabled: Option<bool>,
    /// Whether to also run TTS preemptively before the turn is confirmed.
    /// When ``False`` (default), only LLM runs preemptively; TTS starts once the
    /// turn is confirmed and the speech is scheduled.
    #[host(default = None)]
    pub preemptive_tts: Option<bool>,
    /// Maximum user speech duration (s) for which preemptive generation
    /// is attempted. Beyond this threshold, preemptive generation is skipped
    /// since long utterances are more likely to change and users may expect
    /// slower responses. Defaults to ``10.0``.
    #[host(default = None)]
    pub max_speech_duration: Option<f64>,
    /// Maximum number of preemptive generation attempts per user turn.
    /// The counter resets when the turn completes. Defaults to ``3``.
    #[host(default = None)]
    pub max_retries: Option<i64>,
}

/// Configuration for detecting when a user has been speaking too long
/// without the agent successfully responding.
///
/// The framework tracks accumulated word count and wall-clock duration
/// across consecutive user turns. Counters only reset when the agent
/// transitions to ``speaking`` state (i.e., produces audio output).
///
/// Both thresholds default to ``None`` (disabled). Set at least one to
/// enable the feature.
#[crate::host(record(
    module = "turn",
    name = "UserTurnLimitOptions",
    path = "livekit.agents.voice.io",
    typed_dict,
    extract
))]
#[derive(Clone, Copy, Default, PartialEq)]
pub struct UserTurnLimitKwargs {
    /// Maximum accumulated word count before triggering. Uses the
    /// framework's WordTokenizer for counting. ``None`` disables word-based
    /// limiting. Defaults to ``None``.
    pub max_words: Maybe<Option<i64>>,
    /// Maximum wall-clock duration (seconds) since the user first started
    /// speaking in the current accumulation window. ``None`` disables
    /// duration-based limiting. Defaults to ``None``.
    pub max_duration: Maybe<Option<f64>>,
}

/// Configuration for the turn handling system.
///
/// Can be passed as a plain dict::
///
#[cfg_attr(doctest, doc = "```text")]
///     AgentSession(
///         turn_handling={
///             "endpointing": {"min_delay": 0.3},
///             "interruption": {"enabled": False},
///             "preemptive_generation": {"preemptive_tts": True},
///         },
///     )
#[cfg_attr(doctest, doc = "```")]
///
/// All keys are optional and default to sensible values.
//
// The `cfg_attr(doctest)` lines fence the Python sample for `cargo test --doc` only; the codegen
// reads `///` alone, so the generated docstring is unchanged.
#[crate::host(record(
    module = "turn",
    name = "TurnHandlingOptions",
    path = "livekit.agents.voice.io",
    typed_dict,
    extract
))]
#[derive(Clone, Copy, Default, PartialEq)]
pub struct TurnHandlingKwargs {
    /// Strategy for deciding when the user has finished speaking.
    /// Absent means the session auto-selects.
    //
    // An explicit `None` is [`TurnDetectionKwarg::Null`], not an absent key.
    pub turn_detection: Maybe<TurnDetectionKwarg>,
    /// Endpointing configuration. Defaults to ``{"min_delay": 0.5, "max_delay": 3.0}``.
    #[host(default = None)]
    pub endpointing: Option<EndpointingKwargs>,
    /// Interruption handling configuration. Use ``{"enabled": False}`` to disable.
    #[host(default = None)]
    pub interruption: Option<InterruptionKwargs>,
    /// Preemptive generation configuration. Use ``{"enabled": False}`` to disable.
    #[host(default = None)]
    pub preemptive_generation: Option<PreemptiveKwargs>,
    /// User turn limit configuration. Use ``{"max_words": 50}`` to enable.
    #[host(default = None)]
    pub user_turn_limit: Option<UserTurnLimitKwargs>,
}

/// Every option keyword of the reference's `AgentSession.__init__`, once. Host-object kwargs
/// (`stt`, `llm`, `tools`, ...) are legs, not options.
#[crate::host(record(module = "session_options", typed_dict, extract))]
#[derive(Clone, PartialEq)]
pub struct SessionKwargs {
    #[host(default = None)]
    pub turn_handling: Option<TurnHandlingKwargs>,
    #[host(default = 3)]
    pub max_tool_steps: i64,
    pub use_tts_aligned_transcript: Maybe<bool>,
    /// Unsaid, "none", or a set.
    pub tts_text_transforms: Maybe<Option<TextTransformSet>>,
    #[host(default = 0.0)]
    pub min_consecutive_speech_delay: f64,
    pub expressive: Maybe<Expressive>,
    /// Unsaid is the default warmup, `None` none at all (outbound SIP).
    pub aec_warmup_duration: Maybe<Option<f64>>,
    #[host(default = false)]
    pub ivr_detection: bool,
    /// `None` disables the away timer.
    #[host(default = 15.0)]
    pub user_away_timeout: Option<f64>,
    /// `None` disables the timeout, `0` arms it with nothing left to wait.
    #[host(default = None)]
    pub transcription_timeout: Option<f64>,
    #[host(default = 2.0)]
    pub session_close_transcript_timeout: f64,
    pub conn_options: SessionConnOptions,
}

/// `AgentSession.update_options(..)`'s option surface.
#[crate::host(record(module = "session_options", typed_dict, extract))]
#[derive(Clone, Default, PartialEq)]
pub struct UpdateKwargs {
    #[host(default = None)]
    pub endpointing_opts: Option<EndpointingKwargs>,
    /// `None` reverts to automatic selection (`Auto`).
    pub turn_detection: Maybe<TurnDetectionKwarg>,
    pub expressive: Maybe<Expressive>,
    /// The user-defined keyterms, replaced; the detected ones stay.
    #[host(default = None)]
    pub keyterms: Option<Vec<String>>,
    /// Merged over the session's interruption handling; its `mode` is set at construction.
    #[host(default = None)]
    pub interruption: Option<InterruptionKwargs>,
    #[host(default = None)]
    pub min_consecutive_speech_delay: Option<f64>,
    pub tts_text_transforms: Maybe<Option<TextTransformSet>>,
}

// `Maybe<T>` has `Debug` only for `Maybe<Option<T>>`, so these impls are written out.
fn given<T>(m: &Maybe<T>) -> Option<&T> {
    match m {
        Maybe::NotGiven => None,
        Maybe::Given(v) => Some(v),
    }
}

macro_rules! debug_record {
    ($t:ident { $($plain:ident),* ; $($maybe:ident),* $(,)? }) => {
        impl fmt::Debug for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut s = f.debug_struct(stringify!($t));
                $( s.field(stringify!($plain), &self.$plain); )*
                $( s.field(stringify!($maybe), &given(&self.$maybe)); )*
                s.finish()
            }
        }
    };
}

debug_record!(InterruptionKwargs {
    enabled, discard_audio_if_uninterruptible, min_duration, min_words, resume_false_interruption;
    mode, false_interruption_timeout, backchannel_boundary
});

debug_record!(UserTurnLimitKwargs { ; max_words, max_duration });

debug_record!(TurnHandlingKwargs {
    endpointing, interruption, preemptive_generation, user_turn_limit; turn_detection
});

debug_record!(SessionKwargs {
    turn_handling, max_tool_steps, min_consecutive_speech_delay,
    ivr_detection, user_away_timeout, transcription_timeout,
    session_close_transcript_timeout, conn_options;
    use_tts_aligned_transcript, tts_text_transforms, aec_warmup_duration, expressive
});

debug_record!(UpdateKwargs {
    endpointing_opts, keyterms, interruption, min_consecutive_speech_delay;
    turn_detection, expressive, tts_text_transforms
});

/// `SessionLegs` as the host states it before `start()`, so `session.options` can resolve early.
#[crate::host(record(module = "session_options", typed_dict, extract))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionLegs {
    pub stt: bool,
    pub vad: bool,
    // `self._using_default_vad`: a default VAD is present but not trusted for timing.
    pub using_default_vad: bool,
    pub llm: bool,
    pub tts: bool,
    // The LLM is a realtime model (its socket may own turn detection).
    pub realtime: bool,
    /// An end-of-turn detector was built, and which protocol it speaks.
    pub detector: Option<DetectorKind>,
    // The host's eight-condition gate said yes.
    pub adaptive_detector: bool,
    pub audio_enabled: bool,
    pub audio_output: bool,
    // `session.output.audio.can_pause`, cached (reading it is a host call).
    pub sink_can_pause: bool,
}

impl SessionLegs {
    /// No legs at all: the non-streaming base the `f64` defaults are read against.
    pub const fn none() -> Self {
        SessionLegs {
            stt: false,
            vad: false,
            using_default_vad: false,
            llm: false,
            tts: false,
            realtime: false,
            detector: None,
            adaptive_detector: false,
            audio_enabled: false,
            audio_output: false,
            sink_can_pause: false,
        }
    }
}

/// The resolved `endpointing` group, seconds as `f64`.
#[crate::host(record(module = "session_options", typed_dict, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedEndpointing {
    pub mode: String,
    pub min_delay: f64,
    pub max_delay: f64,
    pub alpha: f64,
}

/// The resolved `interruption` group. `mode` is absent when the session picks.
#[crate::host(record(module = "session_options", typed_dict, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedInterruption {
    pub enabled: bool,
    #[host(default = None)]
    pub mode: Option<String>,
    pub discard_audio_if_uninterruptible: bool,
    pub min_duration: f64,
    pub min_words: u32,
    pub resume_false_interruption: bool,
    pub false_interruption_timeout: Option<f64>,
    pub backchannel_boundary: Option<(f64, f64)>,
}

/// The resolved `preemptive_generation` group.
#[crate::host(record(module = "session_options", typed_dict, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPreemptiveGeneration {
    pub enabled: bool,
    pub preemptive_tts: bool,
    pub max_speech_duration: f64,
    pub max_retries: u32,
}

/// The resolved `user_turn_limit` group.
#[crate::host(record(module = "session_options", typed_dict, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedUserTurnLimit {
    pub max_words: Option<u32>,
    pub max_duration: Option<f64>,
}

/// The session layer's sparse endpointing (`AgentSessionOptions.endpointing_overrides`).
#[crate::host(record(module = "session_options", typed_dict, emit))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EndpointingOverrides {
    #[host(default = None)]
    pub mode: Option<String>,
    #[host(default = None)]
    pub min_delay: Option<f64>,
    #[host(default = None)]
    pub max_delay: Option<f64>,
    #[host(default = None)]
    pub alpha: Option<f64>,
}

/// The resolved `AgentSessionOptions` as the host reads it: every public option of
/// `session.options`, seconds as `f64`. Built only by `resolved_session_options`, whose exhaustive
/// destructure keeps it complete.
#[crate::host(record(module = "session_options", typed_dict, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedSessionOptions {
    pub endpointing: ResolvedEndpointing,
    pub interruption: ResolvedInterruption,
    pub preemptive_generation: ResolvedPreemptiveGeneration,
    pub user_turn_limit: ResolvedUserTurnLimit,
    /// The `turn_detection` the session runs on, as the public literal; `None` is `Auto`.
    pub turn_detection_mode: Option<String>,
    pub endpointing_overrides: EndpointingOverrides,
    pub max_tool_steps: u32,
    pub min_consecutive_speech_delay: f64,
    pub use_tts_aligned_transcript: Option<bool>,
    pub tts_text_transforms: Option<Vec<String>>,
    pub ivr_detection: bool,
    pub user_away_timeout: Option<f64>,
    pub transcription_timeout: Option<f64>,
    pub aec_warmup_duration: Option<f64>,
    pub session_close_transcript_timeout: f64,
}

/// `AgentActivity._resolve_interruption_detection`'s verdict: whether to build the
/// `AdaptiveInterruptionDetector`, and if not, why. Every key is always present.
#[crate::host(record(module = "session_options", typed_dict, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterruptionGateVerdict {
    pub build: bool,
    /// `"cannot_gatekeep"`, `"interruptions_disabled"`, `"asked_vad"`, `"production_default"`.
    pub reason: Option<String>,
    /// `"warning"` or `"info"`.
    pub level: Option<String>,
    pub message: Option<String>,
}

/// The host's reads behind the adaptive-interruption detector: whether to build it, and whether a
/// realtime model's server detects turns.
#[crate::host(functions)]
pub trait InterruptionGateFunctions {
    /// `AgentActivity._rt_turn_detection_enabled` from the host's reads. `turn_detection` is the
    /// explicit mode string (agent over session), `None` when unsaid or a detector object;
    /// `streaming_detector` says it was a streaming detector.
    #[host(function)]
    fn rt_server_turn_detection(
        #[host(kw)] caps_turn_detection: bool,
        can_disable_turn_detection: bool,
        vad: bool,
        turn_detection: Option<String>,
        streaming_detector: bool,
        interruption_given: bool,
    ) -> bool;

    /// `AgentActivity._resolve_interruption_detection`: whether to build the detector, and if not,
    /// why. Constructing it stays the host's.
    #[host(function)]
    #[allow(clippy::too_many_arguments)]
    fn resolve_interruption_detector(
        #[host(kw)] stt_aligned_transcript: Option<bool>,
        stt_streaming: Option<bool>,
        realtime: bool,
        rt_server_turn_detection: bool,
        has_vad: bool,
        turn_detection: Option<String>,
        allow_interruptions: bool,
        agent_mode: Option<InterruptionMode>,
        session_mode: Option<InterruptionMode>,
        hosted: bool,
        dev_mode: bool,
    ) -> InterruptionGateVerdict;
}

// Declared in `telemetry::report`; re-exported so the generated `session_options` module resolves.
pub use crate::telemetry::report::RecordingOptions;

/// Instructions with optional per-modality additions; the host gives the common text.
#[crate::host(codec(extract, host = "str", via = String))]
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Instructions {
    pub common: String,
    pub audio: Option<String>,
    pub text: Option<String>,
}

impl Instructions {
    pub fn plain(common: impl Into<String>) -> Self {
        Instructions { common: common.into(), audio: None, text: None }
    }

    /// Empty parts are dropped, not joined: `audio=""` adds no section and no stray `"\n\n"`.
    pub fn render(&self, modality: Option<Modality>) -> String {
        let addition = match modality {
            Some(Modality::Audio) => self.audio.as_deref(),
            Some(Modality::Text) => self.text.as_deref(),
            None => None,
        };
        [Some(self.common.as_str()), addition]
            .into_iter()
            .flatten()
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl From<String> for Instructions {
    fn from(common: String) -> Self {
        Self::plain(common)
    }
}

/// Configuration for the expressive pipeline, passed as ``AgentSession(expressive=...)``
/// or ``Agent(expressive=...)`` (the agent value overrides the session's).
///
/// Controls how TTS markup instructions are injected into the LLM when expressive is
/// enabled. All keys are optional; common shapes:
///
/// - ``{"speech_steering": {...}}`` — steer delivery and non-verbal sounds on top of
///   the provider-agnostic default instructions.
/// - ``{"tts_instructions_template": "..."}`` — a fully custom prompt.
/// - ``{"tts_instructions_append": "..."}`` — your own rules appended to the template.
///
/// Any explicit template overrides the default; unset parts fall back to the
/// provider-agnostic default.
//
// What the user wrote, sparse; `ExpressiveInputs` in `lk-agents-voice` is what resolving it
// produced.
#[crate::host(record(
    module = "expressive",
    name = "ExpressiveOptions",
    path = "livekit.agents.voice.agent_session",
    typed_dict,
    extract
))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpressiveConfig {
    #[host(default = None)]
    pub speech_steering: Option<SpeechSteering>,
    #[host(default = None)]
    pub tts_instructions_template: Option<Instructions>,
    #[host(default = None)]
    pub tts_instructions_append: Option<String>,
}

/// `bool | ExpressiveOptions`, as `VoiceSession(expressive=)`, `Agent(expressive=)` and
/// `update_options(expressive=)` take it: `False`/`None` is off, `True` the default options, a dict
/// (even an empty one) on with those options.
#[crate::host(codec(
    extract,
    host = "bool | livekit.agents._generated.expressive.ExpressiveOptions | None",
    py = "py_values::expressive"
))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Expressive(pub Option<ExpressiveConfig>);

impl SessionConnOptions {
    pub fn validate(&self) -> Result<(), crate::types::connect::ApiConnectOptionsError> {
        for options in [&self.stt, &self.llm, &self.tts].into_iter().flatten() {
            options.validate()?;
        }
        Ok(())
    }
}
