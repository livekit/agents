//! The `AgentSession` option seam: kwargs as written, legs as stated, the resolved configuration.
//! The policy lives in `lk-agents-voice/src/agent_activity/options.rs` and `turn.rs`.

use crate::Maybe;
use crate::Modality;
use crate::tts::markup::SpeechSteering;
use lk_agents_host_derive::{Host, Vocabulary};
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
#[crate::host(wire_strs(err = UnknownLiteral, msg = UnknownLiteral {
    what: "detector kind",
    value: s.to_owned(),
    expected: "\"streaming\", \"batch\" or None",
}))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Vocabulary)]
pub enum DetectorKind {
    Batch,
    Streaming,
}

/// Which builtin transforms run between the segmenter and the TTS.
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

// An unknown literal is refused, never folded into a default.
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

// An unknown literal is refused, never folded into a default.
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackchannelBoundary {
    Both(f64),
    Split(f64, f64),
}

impl BackchannelBoundary {
    pub fn start(self) -> f64 {
        match self {
            BackchannelBoundary::Both(v) => v,
            BackchannelBoundary::Split(s, _) => s,
        }
    }

    pub fn end(self) -> f64 {
        match self {
            BackchannelBoundary::Both(v) => v,
            BackchannelBoundary::Split(_, e) => e,
        }
    }
}

/// A kwarg whose value is a host object, or a dict that holds one: only presence crosses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HostValue {
    #[default]
    NotGiven,
    Given,
}

impl HostValue {
    pub fn is_given(self) -> bool {
        matches!(self, HostValue::Given)
    }
}

/// Of a detector object only whether it streams crosses.
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

/// Facts about the session, set by the builder and by `update_agent`, never by the user.
pub type LegSet = SessionLegs;

// In these groups a `///` is the reference's `voice/turn.py` docstring verbatim; Rust notes are
// `//`. The dicts are `total=False`: an outer `Option` is a key's absence.

/// Configuration for endpointing.
///
/// All keys are optional. Missing keys inherit from the session default
/// (at the ``Agent`` level) or use the documented defaults
/// (at the ``AgentSession`` level).
//
// Written back by `migrate_turn_handling`, which only fills `min_delay`/`max_delay`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "turn",
    path = "livekit.agents.voice.turn",
    name = "EndpointingOptions"
)]
#[binding(extract, emit, from_items)]
pub struct EndpointingKwargs {
    /// Endpointing mode. ``"fixed"`` for fixed delay, ``"dynamic"`` for dynamic delay. Defaults to ``"fixed"``.
    #[python(ty = "Literal[\"fixed\", \"dynamic\"]")]
    #[binding(skip)]
    pub mode: Option<EndpointingMode>,
    /// Minimum time (s) since last detected speech before declaring the
    /// user's turn complete. Defaults to ``0.5``.
    #[binding(skip_if_none)]
    pub min_delay: Option<f64>,
    /// Maximum time (s) the agent waits before terminating the turn.
    /// Defaults to ``3.0``.
    #[binding(skip_if_none)]
    pub max_delay: Option<f64>,
    /// Exponential moving average coefficient for dynamic endpointing.
    /// The higher the value, the more weight is given to the history.
    /// Defaults to ``0.9``. Only applies when mode is ``dynamic``.
    #[binding(skip)]
    pub alpha: Option<f64>,
}

impl EndpointingKwargs {
    pub fn is_empty(&self) -> bool {
        *self == EndpointingKwargs::default()
    }
}

/// Configuration for interruption handling.
///
/// All keys are optional. Missing keys inherit from the session default
/// (at the ``Agent`` level) or use the documented defaults
/// (at the ``AgentSession`` level).
///
/// ``mode`` absent means the session picks the best available strategy.
//
// Nullable keys are `Maybe<Option<_>>`: absent, `None` ("off") and a value are three states. Field
// order is the reference migration's key order (`order` gives the `TypedDict`'s).
#[derive(Clone, Copy, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "turn",
    path = "livekit.agents.voice.turn",
    name = "InterruptionOptions",
    order = "enabled, mode, discard_audio_if_uninterruptible, min_duration, min_words, \
             resume_false_interruption, false_interruption_timeout, backchannel_boundary"
)]
#[binding(extract, emit, from_items)]
pub struct InterruptionKwargs {
    /// Whether interruptions are enabled. Defaults to ``True``.
    #[binding(skip_if_none)]
    pub enabled: Option<bool>,
    /// Interruption handling strategy. ``"adaptive"`` for ML-based
    /// detection, ``"vad"`` for simple voice-activity detection.
    /// Absent means auto-detect.
    #[python(ty = "Literal[\"adaptive\", \"vad\"]")]
    #[binding(skip)]
    pub mode: Option<InterruptionMode>,
    /// Drop buffered audio while the agent speaks and cannot be
    /// interrupted. Defaults to ``True``.
    #[binding(skip_if_none)]
    pub discard_audio_if_uninterruptible: Option<bool>,
    /// Minimum speech length (s) to register as an interruption.
    /// Defaults to ``0.5``.
    #[binding(skip_if_none)]
    pub min_duration: Option<f64>,
    /// Minimum word count to consider an interruption (STT only).
    /// Defaults to ``0``.
    #[binding(skip_if_none)]
    pub min_words: Option<i64>,
    /// Seconds of silence after an interruption before it is
    /// classified as false. ``None`` disables. Defaults to ``2.0``.
    #[python(ty = "float | None")]
    #[binding(skip_if_none)]
    pub false_interruption_timeout: lk_agents_abi::Maybe<Option<f64>>,
    /// Resume the agent's speech after a false interruption.
    /// Defaults to ``True``.
    #[binding(skip_if_none)]
    pub resume_false_interruption: Option<bool>,
    /// Seconds near the start/end of each agent turn during which overlapping
    /// speech classified as a backchannel by the adaptive detector is suppressed
    /// (events flagged as interruptions still pass through). Use a tuple to apply
    /// different values for start and end separately. ``None`` disables. Defaults
    /// to ``(1.0, 1.0)``. End value accounts for STT transcript timestamp
    /// inaccuracy.
    #[python(ty = "float | tuple[float, float] | None")]
    #[binding(skip)]
    pub backchannel_boundary: lk_agents_abi::Maybe<Option<BackchannelBoundary>>,
}

impl InterruptionKwargs {
    pub fn is_empty(&self) -> bool {
        *self == InterruptionKwargs::default()
    }
}

/// Configuration for preemptive generation.
//
// `migrate_turn_handling` writes `enabled` only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "turn",
    path = "livekit.agents.voice.turn",
    name = "PreemptiveGenerationOptions"
)]
#[binding(extract, emit, from_items)]
pub struct PreemptiveKwargs {
    /// Whether preemptive generation is enabled. Defaults to ``True``.
    #[binding(skip_if_none)]
    pub enabled: Option<bool>,
    /// Whether to also run TTS preemptively before the turn is confirmed.
    /// When ``False`` (default), only LLM runs preemptively; TTS starts once the
    /// turn is confirmed and the speech is scheduled.
    #[binding(skip)]
    pub preemptive_tts: Option<bool>,
    /// Maximum user speech duration (s) for which preemptive generation
    /// is attempted. Beyond this threshold, preemptive generation is skipped
    /// since long utterances are more likely to change and users may expect
    /// slower responses. Defaults to ``10.0``.
    #[binding(skip)]
    pub max_speech_duration: Option<f64>,
    /// Maximum number of preemptive generation attempts per user turn.
    /// The counter resets when the turn completes. Defaults to ``3``.
    #[binding(skip)]
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
//
// Both keys nullable: `None` is "no limit".
#[derive(Clone, Copy, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "turn",
    path = "livekit.agents.voice.turn",
    name = "UserTurnLimitOptions"
)]
#[binding(extract, from_items)]
pub struct UserTurnLimitKwargs {
    /// Maximum accumulated word count before triggering. Uses the
    /// framework's WordTokenizer for counting. ``None`` disables word-based
    /// limiting. Defaults to ``None``.
    #[python(ty = "int | None")]
    pub max_words: lk_agents_abi::Maybe<Option<i64>>,
    /// Maximum wall-clock duration (seconds) since the user first started
    /// speaking in the current accumulation window. ``None`` disables
    /// duration-based limiting. Defaults to ``None``.
    #[python(ty = "float | None")]
    pub max_duration: lk_agents_abi::Maybe<Option<f64>>,
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
// Every group and key optional. `order` puts `turn_detection` first, as the reference does.
// The `cfg_attr(doctest)` lines fence the Python sample for `cargo test --doc` only; the codegen
// reads `///` alone, so the generated docstring is unchanged.
#[derive(Debug, Clone, Copy, Default, PartialEq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "turn",
    path = "livekit.agents.voice.turn",
    name = "TurnHandlingOptions",
    order = "turn_detection, endpointing, interruption, preemptive_generation, user_turn_limit"
)]
pub struct TurnHandlingKwargs {
    /// Endpointing configuration. Defaults to ``{"min_delay": 0.5, "max_delay": 3.0}``.
    #[python(ty = "EndpointingOptions")]
    #[binding(default)]
    pub endpointing: Option<EndpointingKwargs>,
    /// Interruption handling configuration. Use ``{"enabled": False}`` to disable.
    #[python(ty = "InterruptionOptions")]
    #[binding(default)]
    pub interruption: Option<InterruptionKwargs>,
    /// Strategy for deciding when the user has finished speaking.
    /// Absent means the session auto-selects.
    //
    // An explicit `None` is [`TurnDetectionKwarg::Null`], not an absent key.
    #[python(ty = "TurnDetectionMode | None", type_import = "..voice.turn:TurnDetectionMode")]
    #[binding(default)]
    pub turn_detection: Option<TurnDetectionKwarg>,
    /// Preemptive generation configuration. Use ``{"enabled": False}`` to disable.
    #[python(ty = "PreemptiveGenerationOptions")]
    #[binding(default)]
    pub preemptive_generation: Option<PreemptiveKwargs>,
    /// User turn limit configuration. Use ``{"max_words": 50}`` to enable.
    #[python(ty = "UserTurnLimitOptions")]
    #[binding(default)]
    pub user_turn_limit: Option<UserTurnLimitKwargs>,
}

/// `_migrate_turn_handling`'s answer as `Agent.__init__` stores it, with the caller's own
/// `turn_detection` object back in it, in the reference's key order.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[binding(emit, skip_if_none)]
pub struct MigratedTurnHandling {
    pub endpointing: Option<EndpointingKwargs>,
    pub interruption: Option<InterruptionKwargs>,
    /// The caller's own value (a mode or a detector object), `None` for `turn_detection=None`,
    /// absent when not given.
    pub turn_detection: Option<Option<lk_agents_abi::HostRef>>,
    pub preemptive_generation: Option<PreemptiveKwargs>,
}

/// The eleven deprecated `AgentSession.__init__` kwargs `_migrate_turn_handling` takes. The
/// nullable ones are `Maybe<Option<f64>>`: explicit `None` means "off".
#[derive(Clone, Copy, Default, PartialEq, Host)]
#[python(typed_dict, module = "session_options", name = "DeprecatedTurnKwargs")]
#[binding(extract, from_items, default)]
pub struct DeprecatedTurnKwargs {
    #[python(ty = "NotGivenOr[float]", import = "..types:NotGivenOr")]
    pub min_endpointing_delay: lk_agents_abi::Maybe<f64>,
    #[python(ty = "NotGivenOr[float]")]
    pub max_endpointing_delay: lk_agents_abi::Maybe<f64>,
    #[python(ty = "NotGivenOr[float | None]")]
    pub false_interruption_timeout: lk_agents_abi::Maybe<Option<f64>>,
    #[python(
        ty = "NotGivenOr[TurnDetectionMode | None]",
        type_import = "..voice.turn:TurnDetectionMode"
    )]
    pub turn_detection: lk_agents_abi::Maybe<TurnDetectionKwarg>,
    #[python(ty = "NotGivenOr[bool]")]
    pub discard_audio_if_uninterruptible: lk_agents_abi::Maybe<bool>,
    #[python(ty = "NotGivenOr[float]")]
    pub min_interruption_duration: lk_agents_abi::Maybe<f64>,
    #[python(ty = "NotGivenOr[int]")]
    pub min_interruption_words: lk_agents_abi::Maybe<i64>,
    /// Tested `== Given(false)`, **not** `.is_given()`: `allow_interruptions=True` writes nothing.
    #[python(ty = "NotGivenOr[bool]")]
    pub allow_interruptions: lk_agents_abi::Maybe<bool>,
    #[python(ty = "NotGivenOr[bool]")]
    pub resume_false_interruption: lk_agents_abi::Maybe<bool>,
    #[python(ty = "NotGivenOr[float | None]")]
    pub agent_false_interruption_timeout: lk_agents_abi::Maybe<Option<f64>>,
    #[python(ty = "NotGivenOr[bool]")]
    pub preemptive_generation: lk_agents_abi::Maybe<bool>,
}

/// Every option keyword of the reference's `AgentSession.__init__`, once. Host-object kwargs
/// (`stt`, `llm`, `tools`, ...) are legs, not options.
#[derive(Clone, Copy, PartialEq, Host)]
#[python(typed_dict, module = "session_options", name = "SessionKwargs")]
pub struct SessionKwargs {
    /// `turn_handling=`, as written. `None` is `NOT_GIVEN`.
    #[python(
        ty = "NotGivenOr[TurnHandlingOptions]",
        import = "..types:NotGivenOr",
        type_import = "..voice.turn:TurnHandlingOptions"
    )]
    #[binding(default)]
    pub turn_handling: Option<TurnHandlingKwargs>,
    #[python(
        ty = "NotGivenOr[STTContextOptions]",
        type_import = "..voice.keyterm_detection:STTContextOptions"
    )]
    pub stt_context_options: HostValue,
    #[python(
        ty = "NotGivenOr[ToolHandlingOptions]",
        type_import = "..voice.tool_executor:ToolHandlingOptions"
    )]
    pub tool_handling: HostValue,
    pub max_tool_steps: i64,
    #[python(ty = "NotGivenOr[bool]")]
    pub use_tts_aligned_transcript: lk_agents_abi::Maybe<bool>,
    /// `NotGivenOr[Sequence[TextTransforms] | None]`: unsaid, "none", or a set.
    #[python(
        ty = "NotGivenOr[Sequence[TextTransforms] | None]",
        type_import = "collections.abc:Sequence",
        type_import = "..voice.transcription.text_transforms:TextTransforms"
    )]
    pub tts_text_transforms:
        lk_agents_abi::Maybe<Option<lk_agents_abi::voice::session_options::TextTransformSet>>,
    pub min_consecutive_speech_delay: f64,
    /// `bool | ExpressiveOptions`: whether expressive delivery is on; the dict stays host-side.
    #[python(ty = "bool | ExpressiveOptions", type_import = ".expressive:ExpressiveOptions")]
    pub expressive: bool,
    /// `NotGivenOr[float | None]`: unsaid is the default warmup, `None` none at all (outbound SIP).
    #[python(ty = "NotGivenOr[float | None]")]
    pub aec_warmup_duration: lk_agents_abi::Maybe<Option<f64>>,
    pub ivr_detection: bool,
    /// `float | None`: `None` disables the away timer.
    pub user_away_timeout: Option<f64>,
    /// `float | None`: `None` disables the timeout, `0` arms it with nothing left to wait.
    pub transcription_timeout: Option<f64>,
    pub session_close_transcript_timeout: f64,
    #[python(
        ty = "NotGivenOr[SessionConnectOptions]",
        type_import = "..voice.agent_session:SessionConnectOptions"
    )]
    pub conn_options: HostValue,
    /// The eleven deprecated turn kwargs, as one sub-dict.
    pub deprecated: DeprecatedTurnKwargs,
    /// Deprecated; `stt_context_options` supersedes it silently when both are given.
    #[python(
        ty = "NotGivenOr[KeytermsOptions]",
        type_import = "..voice.keyterm_detection:KeytermsOptions"
    )]
    pub keyterms_options: HostValue,
}

/// `AgentSession.update_options(..)`'s option surface (`keyterms` and `expressive` stay host-side).
#[derive(Clone, Copy, Default, PartialEq, Host)]
#[python(typed_dict, module = "session_options", name = "UpdateKwargs")]
pub struct UpdateKwargs {
    #[python(
        ty = "NotGivenOr[EndpointingOptions]",
        import = "..types:NotGivenOr",
        type_import = "..voice.turn:EndpointingOptions"
    )]
    #[binding(default)]
    pub endpointing_opts: Option<EndpointingKwargs>,
    /// `NotGivenOr[TurnDetectionMode | None]`: `None` reverts to automatic selection (`Auto`).
    #[python(
        ty = "NotGivenOr[TurnDetectionMode | None]",
        type_import = "..voice.turn:TurnDetectionMode"
    )]
    #[binding(default)]
    pub turn_detection: lk_agents_abi::Maybe<TurnDetectionKwarg>,
    #[python(ty = "NotGivenOr[float]")]
    #[binding(default)]
    pub min_endpointing_delay: lk_agents_abi::Maybe<f64>,
    #[python(ty = "NotGivenOr[float]")]
    #[binding(default)]
    pub max_endpointing_delay: lk_agents_abi::Maybe<f64>,
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
    enabled, mode, discard_audio_if_uninterruptible, min_duration, min_words, resume_false_interruption;
    false_interruption_timeout, backchannel_boundary
});

debug_record!(UserTurnLimitKwargs { ; max_words, max_duration });

debug_record!(DeprecatedTurnKwargs {
    ;
    min_endpointing_delay, max_endpointing_delay, false_interruption_timeout, turn_detection,
    discard_audio_if_uninterruptible, min_interruption_duration, min_interruption_words,
    allow_interruptions, resume_false_interruption, agent_false_interruption_timeout,
    preemptive_generation
});

debug_record!(SessionKwargs {
    turn_handling, stt_context_options, tool_handling, max_tool_steps, min_consecutive_speech_delay,
    expressive, ivr_detection, user_away_timeout, transcription_timeout,
    session_close_transcript_timeout, conn_options, deprecated, keyterms_options;
    use_tts_aligned_transcript, tts_text_transforms, aec_warmup_duration
});

debug_record!(UpdateKwargs {
    endpointing_opts;
    turn_detection, min_endpointing_delay, max_endpointing_delay
});

/// `LegSet` as the host states it before `start()`, so `session.options` can resolve early.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[python(typed_dict, module = "session_options", name = "SessionLegs")]
#[binding(extract, from_items)]
pub struct SessionLegs {
    pub stt: bool,
    pub vad: bool,
    // `self._using_default_vad`: a default VAD is present but not trusted for timing.
    pub using_default_vad: bool,
    pub llm: bool,
    pub tts: bool,
    // The LLM is a realtime model (its socket may own turn detection).
    pub realtime: bool,
    /// `"streaming"`, `"batch"` or `None`.
    // An end-of-turn detector was built, and which protocol it speaks.
    #[python(ty = "Literal[\"streaming\", \"batch\"] | None")]
    pub detector: Option<lk_agents_abi::voice::session_options::DetectorKind>,
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
#[derive(Debug, Clone, PartialEq, Host)]
#[python(typed_dict, module = "session_options", name = "ResolvedEndpointing")]
#[binding(emit)]
pub struct ResolvedEndpointing {
    pub mode: String,
    pub min_delay: f64,
    pub max_delay: f64,
    pub alpha: f64,
}

/// The resolved `interruption` group. `mode` is absent when the session picks.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(typed_dict, module = "session_options", name = "ResolvedInterruption")]
#[binding(emit)]
pub struct ResolvedInterruption {
    pub enabled: bool,
    #[python(ty = "str")]
    #[binding(skip_if_none)]
    pub mode: Option<String>,
    pub discard_audio_if_uninterruptible: bool,
    pub min_duration: f64,
    pub min_words: u32,
    pub resume_false_interruption: bool,
    pub false_interruption_timeout: Option<f64>,
    pub backchannel_boundary: Option<(f64, f64)>,
}

/// The resolved `preemptive_generation` group.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(typed_dict, module = "session_options", name = "ResolvedPreemptiveGeneration")]
#[binding(emit)]
pub struct ResolvedPreemptiveGeneration {
    pub enabled: bool,
    pub preemptive_tts: bool,
    pub max_speech_duration: f64,
    pub max_retries: u32,
}

/// The resolved `user_turn_limit` group.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(typed_dict, module = "session_options", name = "ResolvedUserTurnLimit")]
#[binding(emit)]
pub struct ResolvedUserTurnLimit {
    pub max_words: Option<u32>,
    pub max_duration: Option<f64>,
}

/// The session layer's sparse endpointing (`AgentSessionOptions.endpointing_overrides`).
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(typed_dict, total = "False", module = "session_options", name = "EndpointingOverrides")]
#[binding(emit, skip_if_none)]
pub struct EndpointingOverrides {
    #[python(ty = "str")]
    pub mode: Option<String>,
    #[python(ty = "float")]
    pub min_delay: Option<f64>,
    #[python(ty = "float")]
    pub max_delay: Option<f64>,
    #[python(ty = "float")]
    pub alpha: Option<f64>,
}

/// `SessionConfig` as the host reads it: every public option of `session.options`, seconds as
/// `f64`. Built only by `resolved_session_options`, whose exhaustive destructure keeps it complete.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(typed_dict, module = "session_options", name = "ResolvedSessionOptions")]
#[binding(emit)]
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
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit)]
pub struct InterruptionGateVerdict {
    pub build: bool,
    /// `"cannot_gatekeep"`, `"interruptions_disabled"`, `"asked_vad"`, `"production_default"`.
    pub reason: Option<String>,
    /// `"warning"` or `"info"`.
    pub level: Option<String>,
    pub message: Option<String>,
}

// Declared in `telemetry::report`; re-exported so the generated `session_options` module resolves.
pub use crate::telemetry::report::RecordingOptions;

/// The sparse `RecordingOptions` for `start(record={...})`, merged onto all-on:
/// `record={"traces": False}` is everything except traces.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[binding(extract, from_items, default)]
pub struct RecordingOverrides {
    pub audio: Option<bool>,
    pub traces: Option<bool>,
    pub logs: Option<bool>,
    pub transcript: Option<bool>,
    pub redaction: Option<bool>,
}

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
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "expressive",
    path = "livekit.agents.voice.agent_session",
    name = "ExpressiveOptions"
)]
pub struct ExpressiveConfig {
    #[python(ty = "SpeechSteeringOptions")]
    pub speech_steering: Option<SpeechSteering>,
    #[python(ty = "Instructions | str", import = "..llm.chat_context:Instructions")]
    pub tts_instructions_template: Option<Instructions>,
    #[python(ty = "str")]
    pub tts_instructions_append: Option<String>,
}

#[cfg(feature = "python")]
mod host {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyString};

    use super::*;
    use crate::python::{MaybeNull, item, item_with};

    crate::python::from_host_str!(EndpointingMode);
    crate::python::from_host_str!(InterruptionMode);
    crate::python::from_host_str!(DetectorKind);
    crate::python::from_host_str!(TurnDetection);

    /// The whole `(common, audio, text)` triple (`beta.workflows` tasks leave `common` empty), or
    /// a bare `str`.
    impl<'a, 'py> FromPyObject<'a, 'py> for Instructions {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if let Ok((common, audio, text)) = obj.extract::<(String, Option<String>, Option<String>)>() {
                return Ok(Instructions { common, audio, text });
            }
            Ok(Instructions::plain(obj.extract::<String>()?))
        }
    }

    /// `NOT_GIVEN` and `None` are "not given", anything else is "given".
    impl<'a, 'py> FromPyObject<'a, 'py> for HostValue {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            Ok(match obj.extract::<MaybeNull<Bound<'py, PyAny>>>()? {
                MaybeNull::NotGiven | MaybeNull::Null => HostValue::NotGiven,
                MaybeNull::Given(_) => HostValue::Given,
            })
        }
    }

    /// A scalar applies to both edges.
    impl<'a, 'py> FromPyObject<'a, 'py> for BackchannelBoundary {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            Ok(match obj.extract::<(f64, f64)>() {
                Ok((start, end)) => BackchannelBoundary::Split(start, end),
                Err(_) => BackchannelBoundary::Both(obj.extract()?),
            })
        }
    }

    /// `Sequence[TextTransforms]`: an unknown name raises rather than silently not running.
    impl<'a, 'py> FromPyObject<'a, 'py> for TextTransformSet {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            TextTransformSet::from_names(&obj.extract::<Vec<String>>()?).map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }

    /// Of a detector object only whether it streams crosses; the check mirrors
    /// `runtime_checkable`: `model`, `provider` and `stream`.
    impl<'a, 'py> FromPyObject<'a, 'py> for TurnDetectionKwarg {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if obj.is_none() {
                return Ok(TurnDetectionKwarg::Null);
            }
            if obj.is_instance_of::<PyString>() {
                return Ok(TurnDetectionKwarg::Mode(obj.extract::<String>()?.parse().map_err(|e: UnknownLiteral| PyValueError::new_err(e.to_string()))?));
            }
            let streaming = obj.hasattr("model")? && obj.hasattr("provider")? && obj.hasattr("stream")?;
            Ok(TurnDetectionKwarg::Detector { streaming })
        }
    }

    /// `NOT_GIVEN` and `None` read as absent: a group or kwarg the caller left unsaid.
    fn unsaid(v: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(matches!(v.extract::<MaybeNull<Bound<'_, PyAny>>>()?, MaybeNull::NotGiven | MaybeNull::Null))
    }

    /// An explicit `turn_detection=None` is [`TurnDetectionKwarg::Null`], not an absent key.
    impl<'a, 'py> FromPyObject<'a, 'py> for TurnHandlingKwargs {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let d = obj.cast::<PyDict>()?;
            Ok(TurnHandlingKwargs {
                endpointing: match d.get_item("endpointing")? { Some(v) => v.extract()?, None => None },
                interruption: match d.get_item("interruption")? { Some(v) => v.extract()?, None => None },
                turn_detection: match d.get_item("turn_detection")? { Some(v) => Some(v.extract()?), None => None },
                preemptive_generation: match d.get_item("preemptive_generation")? { Some(v) => v.extract()?, None => None },
                user_turn_limit: match d.get_item("user_turn_limit")? { Some(v) => v.extract()?, None => None },
            })
        }
    }

    /// `update_options(..)`: `endpointing_opts` unsaid (`NOT_GIVEN`, `None`) is no group.
    impl<'a, 'py> FromPyObject<'a, 'py> for UpdateKwargs {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let d = obj.cast::<PyDict>()?;
            Ok(UpdateKwargs {
                endpointing_opts: match d.get_item("endpointing_opts")? {
                    Some(v) if !unsaid(&v)? => Some(v.extract()?),
                    _ => None,
                },
                turn_detection: match d.get_item("turn_detection")? { Some(v) => v.extract()?, None => Maybe::NotGiven },
                min_endpointing_delay: match d.get_item("min_endpointing_delay")? { Some(v) => v.extract()?, None => Maybe::NotGiven },
                max_endpointing_delay: match d.get_item("max_endpointing_delay")? { Some(v) => v.extract()?, None => Maybe::NotGiven },
            })
        }
    }

    /// `expressive` is its truthiness; the dict stays host-side.
    impl<'a, 'py> FromPyObject<'a, 'py> for SessionKwargs {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            const OWNER: &str = "SessionKwargsFromPy";
            let obj = &*obj;
            Ok(SessionKwargs {
                turn_handling: match obj.get_item("turn_handling") {
                    Ok(v) => item_with(obj, OWNER, "turn_handling", |_| if unsaid(&v)? { Ok(None) } else { v.extract().map(Some) })?,
                    Err(_) => None,
                },
                stt_context_options: item(obj, OWNER, "stt_context_options")?,
                tool_handling: item(obj, OWNER, "tool_handling")?,
                max_tool_steps: item(obj, OWNER, "max_tool_steps")?,
                use_tts_aligned_transcript: item(obj, OWNER, "use_tts_aligned_transcript")?,
                tts_text_transforms: item(obj, OWNER, "tts_text_transforms")?,
                min_consecutive_speech_delay: item(obj, OWNER, "min_consecutive_speech_delay")?,
                expressive: item_with(obj, OWNER, "expressive", |v| v.is_truthy())?,
                aec_warmup_duration: item(obj, OWNER, "aec_warmup_duration")?,
                ivr_detection: item(obj, OWNER, "ivr_detection")?,
                user_away_timeout: item(obj, OWNER, "user_away_timeout")?,
                transcription_timeout: item(obj, OWNER, "transcription_timeout")?,
                session_close_transcript_timeout: item(obj, OWNER, "session_close_transcript_timeout")?,
                conn_options: item(obj, OWNER, "conn_options")?,
                deprecated: item(obj, OWNER, "deprecated")?,
                keyterms_options: item(obj, OWNER, "keyterms_options")?,
            })
        }
    }
}
