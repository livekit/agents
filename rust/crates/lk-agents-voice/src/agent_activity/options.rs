//! The session's and the agent's options (the `AgentSession` keywords, `turn_handling` among them
//! in [`crate::turn`]): the layers, their defaults and resolution, and the host's view.

use crate::audio_recognition::TurnDetectionMode;
use crate::events::DiagnosticEvent;
use crate::turn::{
    Endpointing, Interruption, Nullable, PreemptiveGeneration, TurnPatch, TurnPolicy, UserTurnLimit,
    count, endpointing_patch, migrate_turn_handling, nullable, secs, turn_handling_patch,
};
use lk_agents_abi::Maybe;
use lk_agents_abi::Vocabulary;
pub use lk_agents_abi::voice::session_options::{
    DeprecatedTurnKwargs, DetectorKind, EndpointingKwargs, EndpointingMode,
    EndpointingOverrides, HostValue, InterruptionGateVerdict, InterruptionKwargs,
    InterruptionMode, LegSet, PreemptiveKwargs, RecordingOptions, RecordingOverrides,
    ResolvedEndpointing, ResolvedInterruption, ResolvedPreemptiveGeneration,
    ResolvedSessionOptions, ResolvedUserTurnLimit, SessionKwargs, SessionLegs, TextTransformSet,
    TurnDetection, TurnDetectionKwarg, TurnHandlingKwargs, UpdateKwargs, UserTurnLimitKwargs,
};
use std::time::Duration;

/// The one message every deprecated-kwarg diagnostic carries; the kwarg and hint are fields.
pub const DEPRECATED_KWARG: &str = "deprecated AgentSession keyword";

const TURN_HANDLING_HINT: &str = "Use turn_handling=TurnHandlingOptions(...) instead";

const STT_CONTEXT_HINT: &str = "Use stt_context_options=STTContextOptions(...) instead";

/// `_RECORDING_ALL_ON`. Note `redaction: false` -- "all on" is about the four capture keys.
pub const RECORDING_ALL_ON: RecordingOptions =
    RecordingOptions { audio: true, traces: true, logs: true, transcript: true, redaction: false };

/// `_RECORDING_ALL_OFF`.
pub const RECORDING_ALL_OFF: RecordingOptions =
    RecordingOptions { audio: false, traces: false, logs: false, transcript: false, redaction: false };

/// `AgentActivity._resolve_interruption_detection`: said only when the caller asked for `adaptive`.
pub const INTERRUPTION_DETECTION_INCOMPATIBLE: &str = "interruption_detection is provided, but it's not compatible with the current configuration and will be disabled";

/// `agent_activity.AgentActivity._resolve_interruption_detection`.
pub const ADAPTIVE_OFF_IN_PRODUCTION: &str =
    "adaptive interruption is disabled by default in production mode";

/// The layer a `CoreStep::ConfigApplied` reports. `Session` is `__init__` ∪ `update_options`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Vocabulary)]
pub enum ConfigLayer {
    Session,
    Agent,
}

/// Built only by [`ConfigStack::resolve`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SessionConfig {
    pub turn: TurnPolicy,
    pub speech: SpeechPolicy,
    pub timers: Timers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechPolicy {
    pub max_tool_steps: u32,
    pub min_consecutive_speech_delay: Duration,
    /// `use_tts_aligned_transcript`: `None` is the reference's `NOT_GIVEN` -- "if the TTS can".
    pub use_tts_aligned_transcript: Option<bool>,
    /// `tts_text_transforms`: `None` is "run none".
    pub tts_text_transforms: Option<TextTransformSet>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timers {
    pub user_away: Option<Duration>,
    /// `Some(ZERO)` is a region ("fire at end of speech"), not "off".
    pub transcription_timeout: Option<Duration>,
    /// `None` is no warmup at all (an outbound SIP call).
    pub aec_warmup: Option<Duration>,
    pub session_close_transcript_timeout: Duration,
}

/// [`SessionConfig`] with every leaf optional: what one layer said. `Default` is the empty patch.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionPatch {
    pub turn: TurnPatch,
    pub speech: SpeechPatch,
    pub timers: TimersPatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechPatch {
    pub max_tool_steps: Option<u32>,
    pub min_consecutive_speech_delay: Option<Duration>,
    pub use_tts_aligned_transcript: Option<Nullable<bool>>,
    pub tts_text_transforms: Option<Nullable<TextTransformSet>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimersPatch {
    pub user_away: Option<Nullable<Duration>>,
    pub transcription_timeout: Option<Nullable<Duration>>,
    pub aec_warmup: Option<Nullable<Duration>>,
    pub session_close_transcript_timeout: Option<Duration>,
}

/// How a said value lands: `set` (the value), `some` (a layer picks a mode, never "no mode"),
/// `nullable` (`Null` is off). A leaf added to [`SessionPatch`] must be listed here, or no layer
/// can say it.
macro_rules! each_leaf {
    ($m:ident, $into:expr, $from:expr) => {
        $m!($into, $from, set, turn.detection);
        $m!($into, $from, set, turn.endpointing.mode);
        $m!($into, $from, set, turn.endpointing.min_delay);
        $m!($into, $from, set, turn.endpointing.max_delay);
        $m!($into, $from, set, turn.endpointing.alpha);
        $m!($into, $from, set, turn.interruption.enabled);
        $m!($into, $from, some, turn.interruption.mode);
        $m!($into, $from, set, turn.interruption.discard_audio_if_uninterruptible);
        $m!($into, $from, set, turn.interruption.min_duration);
        $m!($into, $from, set, turn.interruption.min_words);
        $m!($into, $from, set, turn.interruption.resume_false_interruption);
        $m!($into, $from, nullable, turn.interruption.false_interruption_timeout);
        $m!($into, $from, nullable, turn.interruption.backchannel_boundary);
        $m!($into, $from, set, turn.preemptive.enabled);
        $m!($into, $from, set, turn.preemptive.preemptive_tts);
        $m!($into, $from, set, turn.preemptive.max_speech_duration);
        $m!($into, $from, set, turn.preemptive.max_retries);
        $m!($into, $from, nullable, turn.user_turn_limit.max_words);
        $m!($into, $from, nullable, turn.user_turn_limit.max_duration);
        $m!($into, $from, set, turn.ivr_detection);
        $m!($into, $from, set, speech.max_tool_steps);
        $m!($into, $from, set, speech.min_consecutive_speech_delay);
        $m!($into, $from, nullable, speech.use_tts_aligned_transcript);
        $m!($into, $from, nullable, speech.tts_text_transforms);
        $m!($into, $from, nullable, timers.user_away);
        $m!($into, $from, nullable, timers.transcription_timeout);
        $m!($into, $from, nullable, timers.aec_warmup);
        $m!($into, $from, set, timers.session_close_transcript_timeout);
    };
}

/// One leaf of a merge: said wins, unsaid keeps.
macro_rules! merge_leaf {
    ($into:expr, $from:expr, $kind:ident, $($f:ident).+) => {
        if let Some(v) = $from.$($f).+ {
            $into.$($f).+ = Some(v);
        }
    };
}

macro_rules! apply_leaf {
    ($cfg:expr, $patch:expr, set, $($f:ident).+) => {
        if let Some(v) = $patch.$($f).+ {
            $cfg.$($f).+ = v;
        }
    };
    ($cfg:expr, $patch:expr, some, $($f:ident).+) => {
        if let Some(v) = $patch.$($f).+ {
            $cfg.$($f).+ = Some(v);
        }
    };
    ($cfg:expr, $patch:expr, nullable, $($f:ident).+) => {
        if let Some(v) = $patch.$($f).+ {
            $cfg.$($f).+ = v.into_option();
        }
    };
}

impl SessionPatch {
    /// Lay `over` on top of this patch (`update_options`): unsaid leaves keep their earlier
    /// answer rather than falling back to the default.
    pub fn merge(&mut self, over: &SessionPatch) {
        each_leaf!(merge_leaf, self, over);
    }
}

/// The reference's defaults, as one `SessionConfig`. The endpointing base tightens to
/// `0.3 / 2.5` with a streaming detector, and the owner is what `Auto` means for these legs.
pub const fn defaults(legs: &LegSet) -> SessionConfig {
    let (min_delay, max_delay) = match legs.detector {
        Some(DetectorKind::Streaming) => (Duration::from_millis(300), Duration::from_millis(2500)),
        Some(DetectorKind::Batch) | None => {
            (Duration::from_millis(500), Duration::from_millis(3000))
        }
    };
    let detection = TurnDetection::Auto;
    SessionConfig {
        turn: TurnPolicy {
            detection,
            owner: resolve_turn_detection_mode(detection, legs),
            endpointing: Endpointing {
                mode: EndpointingMode::Fixed,
                min_delay,
                max_delay,
                alpha: 0.9,
            },
            interruption: Interruption {
                enabled: true,
                mode: None,
                detection_enabled: legs.adaptive_detector,
                by_audio_activity: by_audio_activity(detection),
                discard_audio_if_uninterruptible: true,
                min_duration: Duration::from_millis(500),
                min_words: 0,
                resume_false_interruption: true,
                false_interruption_timeout: Some(Duration::from_millis(2000)),
                backchannel_boundary: Some((
                    Duration::from_millis(1000),
                    Duration::from_millis(1000),
                )),
            },
            preemptive: PreemptiveGeneration {
                enabled: true,
                preemptive_tts: false,
                max_speech_duration: Duration::from_millis(10_000),
                max_retries: 3,
            },
            user_turn_limit: UserTurnLimit { max_words: None, max_duration: None },
            ivr_detection: false,
        },
        speech: SpeechPolicy {
            max_tool_steps: 3,
            min_consecutive_speech_delay: Duration::ZERO,
            use_tts_aligned_transcript: None,
            tts_text_transforms: Some(TextTransformSet::ALL),
        },
        timers: Timers {
            user_away: Some(Duration::from_millis(15_000)),
            transcription_timeout: None,
            aec_warmup: Some(Duration::from_millis(3000)),
            session_close_transcript_timeout: Duration::from_millis(2000),
        },
    }
}

/// `Auto` falls `realtime_llm` → `vad` → `stt` → `manual`. A mode the legs cannot honour falls back
/// (`vad` without VAD → `stt`, `stt` without STT → `vad`, the rest → `Auto`); `Manual` always
/// holds. `RealtimeLlm` only when no local leg can decide; `DriverBuilder::start` refines it.
pub const fn resolve_turn_detection_mode(
    detection: TurnDetection,
    legs: &LegSet,
) -> TurnDetectionMode {
    match detection {
        TurnDetection::Manual => TurnDetectionMode::Manual,
        TurnDetection::Vad if legs.vad => TurnDetectionMode::Vad,
        TurnDetection::Stt if legs.stt && !legs.realtime => TurnDetectionMode::Stt,
        // `realtime_llm` falls exactly as `Auto` does.
        TurnDetection::Vad
        | TurnDetection::Stt
        | TurnDetection::RealtimeLlm
        | TurnDetection::Auto => auto_turn_detection_mode(legs),
    }
}

/// `Auto`'s fall, with `realtime_llm` taken only when no local VAD or STT exists.
const fn auto_turn_detection_mode(legs: &LegSet) -> TurnDetectionMode {
    if legs.realtime && !legs.vad && !legs.stt {
        TurnDetectionMode::RealtimeLlm
    } else if legs.vad {
        TurnDetectionMode::Vad
    } else if legs.stt {
        TurnDetectionMode::Stt
    } else {
        TurnDetectionMode::Manual
    }
}

/// `_interruption_by_audio_activity_enabled`'s initial value: off under `manual` or
/// `realtime_llm`, where the user's audio is not a turn signal.
const fn by_audio_activity(detection: TurnDetection) -> bool {
    match detection {
        TurnDetection::Manual | TurnDetection::RealtimeLlm => false,
        TurnDetection::Vad | TurnDetection::Stt | TurnDetection::Auto => true,
    }
}

/// The layers, in the order they apply: `defaults(legs)` ⊕ `session` ⊕ `agent`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ConfigStack {
    /// `__init__` (after the deprecated-kwarg migration) ∪ `update_options` patches.
    pub session: SessionPatch,
    /// The active `Agent`'s overrides; replaced whole by `update_agent`.
    pub agent: Option<SessionPatch>,
}

impl ConfigStack {
    /// Pure. The only place a default is read and the only constructor of [`SessionConfig`].
    pub fn resolve(&self, legs: &LegSet) -> SessionConfig {
        let mut cfg = defaults(legs);
        apply(&mut cfg, &self.session);
        if let Some(agent) = &self.agent {
            apply(&mut cfg, agent);
        }
        // The two derived leaves follow the layered `detection`, not the default's.
        cfg.turn.owner = resolve_turn_detection_mode(cfg.turn.detection, legs);
        cfg.turn.interruption.by_audio_activity = by_audio_activity(cfg.turn.detection);
        cfg.turn.interruption.detection_enabled = legs.adaptive_detector;
        cfg
    }
}

fn apply(cfg: &mut SessionConfig, patch: &SessionPatch) {
    each_leaf!(apply_leaf, cfg, patch);
}

fn deprecation(kwarg: &'static str, hint: &'static str) -> DiagnosticEvent {
    DiagnosticEvent::warning(DEPRECATED_KWARG).with("kwarg", kwarg).with("hint", hint)
}

/// One diagnostic per kwarg passed, as `@deprecate_params` warns; `allow_interruptions=True`
/// still warns.
pub fn deprecated_diagnostics(kw: &DeprecatedTurnKwargs) -> Vec<DiagnosticEvent> {
    let DeprecatedTurnKwargs {
        min_endpointing_delay,
        max_endpointing_delay,
        false_interruption_timeout,
        turn_detection,
        discard_audio_if_uninterruptible,
        min_interruption_duration,
        min_interruption_words,
        allow_interruptions,
        resume_false_interruption,
        agent_false_interruption_timeout,
        preemptive_generation,
    } = kw;
    let given: [(&'static str, bool); 11] = [
        ("min_endpointing_delay", min_endpointing_delay.is_given()),
        ("max_endpointing_delay", max_endpointing_delay.is_given()),
        ("false_interruption_timeout", false_interruption_timeout.is_given()),
        ("turn_detection", turn_detection.is_given()),
        ("discard_audio_if_uninterruptible", discard_audio_if_uninterruptible.is_given()),
        ("min_interruption_duration", min_interruption_duration.is_given()),
        ("min_interruption_words", min_interruption_words.is_given()),
        ("allow_interruptions", allow_interruptions.is_given()),
        ("resume_false_interruption", resume_false_interruption.is_given()),
        ("agent_false_interruption_timeout", agent_false_interruption_timeout.is_given()),
        ("preemptive_generation", preemptive_generation.is_given()),
    ];
    given
        .into_iter()
        .filter(|(_, g)| *g)
        .map(|(name, _)| deprecation(name, TURN_HANDLING_HINT))
        .collect()
}

/// The `turn_handling` this session runs on: an explicit `turn_handling=` wins **whole** (the
/// deprecated kwargs are then dropped, though they still warn), else the migration.
pub fn session_turn_handling(kw: &SessionKwargs) -> TurnHandlingKwargs {
    kw.turn_handling.unwrap_or_else(|| migrate_turn_handling(&kw.deprecated))
}

/// `"turn_detection" in turn_handling`: whether the caller named a detector, mode or `None`.
pub fn turn_detection_explicit(kw: &SessionKwargs) -> bool {
    session_turn_handling(kw).turn_detection.is_some()
}

/// Where the session's STT context came from: `stt_context_options` **silently** supersedes the
/// deprecated `keyterms_options` (mapped through `_stt_context_from_keyterms_options`); neither
/// given resolves `None`.
pub fn stt_context_source(kw: &SessionKwargs) -> &'static str {
    match (kw.stt_context_options, kw.keyterms_options) {
        (HostValue::Given, _) => "stt_context_options",
        (HostValue::NotGiven, HostValue::Given) => "keyterms_options",
        (HostValue::NotGiven, HostValue::NotGiven) => "none",
    }
}

/// **The session layer** and its diagnostics: said leaves are `Some`, the rest fall to
/// `defaults(legs)`. Past this line nothing is `f64`.
pub fn session_layer(kw: &SessionKwargs) -> (SessionPatch, Vec<DiagnosticEvent>) {
    let SessionKwargs {
        turn_handling: _,
        stt_context_options: _,
        tool_handling: _,
        max_tool_steps,
        use_tts_aligned_transcript,
        tts_text_transforms,
        min_consecutive_speech_delay,
        expressive: _,
        aec_warmup_duration,
        ivr_detection,
        user_away_timeout,
        transcription_timeout,
        session_close_transcript_timeout,
        conn_options: _,
        deprecated,
        keyterms_options,
    } = *kw;

    let mut diagnostics = deprecated_diagnostics(&deprecated);
    if keyterms_options.is_given() {
        diagnostics.push(deprecation("keyterms_options", STT_CONTEXT_HINT));
    }

    let mut p = turn_handling_patch(&session_turn_handling(kw));
    p.turn.ivr_detection = Some(ivr_detection);
    p.speech.max_tool_steps = Some(count(max_tool_steps));
    p.speech.min_consecutive_speech_delay = Some(secs(min_consecutive_speech_delay));
    p.speech.use_tts_aligned_transcript = nullable_of(use_tts_aligned_transcript);
    p.speech.tts_text_transforms = nullable(tts_text_transforms, |t| t);
    p.timers.user_away = Some(Nullable::from_option(user_away_timeout.map(secs)));
    p.timers.transcription_timeout = Some(Nullable::from_option(transcription_timeout.map(secs)));
    p.timers.aec_warmup = nullable(aec_warmup_duration, secs);
    p.timers.session_close_transcript_timeout = Some(secs(session_close_transcript_timeout));
    (p, diagnostics)
}

/// The deprecated pair builds a **fresh** endpointing patch (`EndpointingOptions()`), so the old
/// form wins when both are passed. One diagnostic for the pair.
pub fn update_patch(kw: &UpdateKwargs) -> (SessionPatch, Vec<DiagnosticEvent>) {
    let UpdateKwargs {
        endpointing_opts,
        turn_detection,
        min_endpointing_delay,
        max_endpointing_delay,
    } = *kw;
    let mut diagnostics = Vec::new();
    let endpointing = if min_endpointing_delay.is_given() || max_endpointing_delay.is_given() {
        diagnostics.push(deprecation(
            "min_endpointing_delay/max_endpointing_delay",
            "use endpointing_opts instead",
        ));
        Some(EndpointingKwargs {
            mode: None,
            min_delay: min_endpointing_delay.given(),
            max_delay: max_endpointing_delay.given(),
            alpha: None,
        })
    } else {
        endpointing_opts
    };
    let mut p = SessionPatch::default();
    if let Some(e) = endpointing {
        p.turn.endpointing = endpointing_patch(&e);
    }
    p.turn.detection = match turn_detection {
        Maybe::NotGiven => None,
        Maybe::Given(TurnDetectionKwarg::Null) => Some(TurnDetection::Auto),
        Maybe::Given(TurnDetectionKwarg::Mode(m)) => Some(m),
        // A detector *object* is the host's to swap; it says nothing about the owner.
        Maybe::Given(TurnDetectionKwarg::Detector { streaming: _ }) => None,
    };
    (p, diagnostics)
}

/// `session.options.turn_detection_mode`: the literal, or `None` for `Auto` -- "from the legs".
fn turn_detection_literal(d: TurnDetection) -> Option<&'static str> {
    match d {
        TurnDetection::Auto => None,
        mode => Some(mode.as_str()),
    }
}

/// [`SessionConfig`] as the host's `ResolvedSessionOptions`. `session` is read for
/// `endpointing_overrides` only. No `..`, so a new leaf must be named.
pub fn resolved_session_options(
    cfg: &SessionConfig,
    session: &SessionPatch,
) -> ResolvedSessionOptions {
    use crate::agent_activity::options as c;
    let SessionConfig {
        turn:
            crate::turn::TurnPolicy {
                detection,
                owner: _,
                endpointing: crate::turn::Endpointing { mode, min_delay, max_delay, alpha },
                interruption:
                    crate::turn::Interruption {
                        enabled,
                        mode: interruption_mode,
                        detection_enabled: _,
                        by_audio_activity: _,
                        discard_audio_if_uninterruptible,
                        min_duration,
                        min_words,
                        resume_false_interruption,
                        false_interruption_timeout,
                        backchannel_boundary,
                    },
                preemptive:
                    crate::turn::PreemptiveGeneration {
                        enabled: preemptive_enabled,
                        preemptive_tts,
                        max_speech_duration,
                        max_retries,
                    },
                user_turn_limit: crate::turn::UserTurnLimit { max_words, max_duration },
                ivr_detection,
            },
        speech:
            c::SpeechPolicy {
                max_tool_steps,
                min_consecutive_speech_delay,
                use_tts_aligned_transcript,
                tts_text_transforms,
            },
        timers:
            c::Timers {
                user_away,
                transcription_timeout,
                aec_warmup,
                session_close_transcript_timeout,
            },
    } = *cfg;
    let crate::turn::EndpointingPatch { mode: o_mode, min_delay: o_min, max_delay: o_max, alpha: o_alpha } =
        session.turn.endpointing;
    ResolvedSessionOptions {
        endpointing: ResolvedEndpointing {
            mode: mode.as_str().to_owned(),
            min_delay: min_delay.as_secs_f64(),
            max_delay: max_delay.as_secs_f64(),
            alpha,
        },
        interruption: ResolvedInterruption {
            enabled,
            mode: interruption_mode.map(|m| m.as_str().to_owned()),
            discard_audio_if_uninterruptible,
            min_duration: min_duration.as_secs_f64(),
            min_words,
            resume_false_interruption,
            false_interruption_timeout: false_interruption_timeout.map(|d| d.as_secs_f64()),
            backchannel_boundary: backchannel_boundary.map(|(s, e)| (s.as_secs_f64(), e.as_secs_f64())),
        },
        preemptive_generation: ResolvedPreemptiveGeneration {
            enabled: preemptive_enabled,
            preemptive_tts,
            max_speech_duration: max_speech_duration.as_secs_f64(),
            max_retries,
        },
        user_turn_limit: ResolvedUserTurnLimit { max_words, max_duration: max_duration.map(|d| d.as_secs_f64()) },
        turn_detection_mode: turn_detection_literal(detection).map(str::to_owned),
        endpointing_overrides: EndpointingOverrides {
            mode: o_mode.map(|m| m.as_str().to_owned()),
            min_delay: o_min.map(|d| d.as_secs_f64()),
            max_delay: o_max.map(|d| d.as_secs_f64()),
            alpha: o_alpha,
        },
        max_tool_steps,
        min_consecutive_speech_delay: min_consecutive_speech_delay.as_secs_f64(),
        use_tts_aligned_transcript,
        tts_text_transforms: tts_text_transforms.map(TextTransformSet::names),
        ivr_detection,
        user_away_timeout: user_away.map(|d| d.as_secs_f64()),
        transcription_timeout: transcription_timeout.map(|d| d.as_secs_f64()),
        aec_warmup_duration: aec_warmup.map(|d| d.as_secs_f64()),
        session_close_transcript_timeout: session_close_transcript_timeout.as_secs_f64(),
    }
}

/// A `Maybe<T>` whose resolved leaf is `Option<T>` (`NOT_GIVEN` resolves to `None`).
fn nullable_of<T>(v: Maybe<T>) -> Option<Nullable<T>> {
    match v {
        Maybe::NotGiven => None,
        Maybe::Given(t) => Some(Nullable::Value(t)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingRequest {
    All(bool),
    Granular(RecordingOverrides),
}

/// `record=True | False | {...}`, read once; `bool` first, so a bool gets the right error.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for RecordingRequest {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        if let Ok(b) = obj.extract::<bool>() {
            return Ok(RecordingRequest::All(b));
        }
        let dict = obj
            .cast::<pyo3::types::PyDict>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("record must be a bool or a RecordingOptions dict"))?;
        Ok(RecordingRequest::Granular(dict.extract::<RecordingOverrides>()?))
    }
}

/// `_resolve_recording_options`. A dict merges onto **all-on** (`{"traces": False}` is everything
/// but traces), while `record=False` records nothing.
pub fn resolve_recording_options(record: RecordingRequest) -> RecordingOptions {
    match record {
        RecordingRequest::All(true) => RECORDING_ALL_ON,
        RecordingRequest::All(false) => RECORDING_ALL_OFF,
        RecordingRequest::Granular(o) => {
            let base = RECORDING_ALL_ON;
            RecordingOptions {
                audio: o.audio.unwrap_or(base.audio),
                traces: o.traces.unwrap_or(base.traces),
                logs: o.logs.unwrap_or(base.logs),
                transcript: o.transcript.unwrap_or(base.transcript),
                redaction: o.redaction.unwrap_or(base.redaction),
            }
        }
    }
}

/// Does a *resolved* `RecordingOptions` record anything? `redaction` is not a capture key.
pub fn records_anything(options: &RecordingOptions) -> bool {
    options.audio || options.traces || options.logs || options.transcript
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SttGate {
    pub aligned_transcript: bool,
    pub streaming: bool,
}

/// Where the session runs: hosted, dev mode, or self-hosted production (adaptive is opt-in).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deployment {
    Hosted,
    Dev,
    Production,
}

/// The inputs `_resolve_interruption_detection` reads, as facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterruptionGateInputs {
    /// The pipeline STT, when the LLM is not a realtime model; `None` is no STT.
    pub stt: Option<SttGate>,
    /// The LLM is realtime, and whether the server detects turns (no local VAD or STT).
    pub realtime: Option<RealtimeGate>,
    /// `self.vad is not None`.
    pub has_vad: bool,
    /// The resolved `turn_detection`: `Manual` and `RealtimeLlm` have nothing to withhold with.
    pub turn_detection: TurnDetection,
    /// `self.allow_interruptions`.
    pub allow_interruptions: bool,
    /// `interruption_detection` as the agent and the session said it, explicitly (`None` is
    /// `NOT_GIVEN`): either saying `Vad` refuses, neither saying anything is the default.
    pub asked: [Option<InterruptionMode>; 2],
    pub deployment: Deployment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealtimeGate {
    pub server_turn_detection: bool,
}

/// The gate's answer. `Build` asks the host to construct the detector; `VadOnly` carries the
/// reference's sentence when it has one.
#[derive(Debug, Clone, PartialEq)]
pub enum InterruptionGate {
    Build,
    /// `reason` is the name the host reads back (`InterruptionGateVerdict::reason`):
    /// `cannot_gatekeep` (no aligned streaming STT, no VAD, `manual` / `realtime_llm`, or a
    /// server-detected realtime model), `interruptions_disabled`, `asked_vad` (the agent or the
    /// session said `"vad"`), or `production_default` (nobody asked, self-hosted production).
    VadOnly { reason: &'static str, diagnostic: Option<DiagnosticEvent> },
}

impl InterruptionGate {
    pub fn verdict(&self) -> InterruptionGateVerdict {
        match self {
            InterruptionGate::Build => {
                InterruptionGateVerdict { build: true, reason: None, level: None, message: None }
            }
            InterruptionGate::VadOnly { reason, diagnostic } => InterruptionGateVerdict {
                build: false,
                reason: Some((*reason).to_owned()),
                level: diagnostic.as_ref().map(|d| d.level.as_str().to_owned()),
                message: diagnostic.as_ref().map(|d| d.message.to_owned()),
            },
        }
    }
}

/// `_resolve_interruption_detection`, condition for condition: nothing can gatekeep (warned
/// only if `adaptive` was asked), `allow_interruptions = false`, either layer said `"vad"`, or
/// neither said anything in production. Then `Build`.
pub fn resolve_interruption_detector(inputs: &InterruptionGateInputs) -> InterruptionGate {
    let InterruptionGateInputs {
        stt,
        realtime,
        has_vad,
        turn_detection,
        allow_interruptions,
        asked,
        deployment,
    } = *inputs;
    let can_gatekeep = match realtime {
        Some(rt) => !rt.server_turn_detection,
        None => stt.is_some_and(|s| s.aligned_transcript && s.streaming),
    };
    let asked_adaptive = asked.contains(&Some(InterruptionMode::Adaptive));
    let asked_vad = asked.contains(&Some(InterruptionMode::Vad));
    let nobody_asked = asked.iter().all(Option::is_none);
    if !can_gatekeep
        || !has_vad
        || matches!(turn_detection, TurnDetection::Manual | TurnDetection::RealtimeLlm)
    {
        return InterruptionGate::VadOnly {
            reason: "cannot_gatekeep",
            diagnostic: asked_adaptive
                .then(|| DiagnosticEvent::warning(INTERRUPTION_DETECTION_INCOMPATIBLE)),
        };
    }
    if !allow_interruptions {
        return InterruptionGate::VadOnly { reason: "interruptions_disabled", diagnostic: None };
    }
    if asked_vad {
        return InterruptionGate::VadOnly { reason: "asked_vad", diagnostic: None };
    }
    if nobody_asked && deployment == Deployment::Production {
        return InterruptionGate::VadOnly {
            reason: "production_default",
            diagnostic: Some(DiagnosticEvent::info(ADAPTIVE_OFF_IN_PRODUCTION)),
        };
    }
    InterruptionGate::Build
}

/// `AgentSession(**kwargs)`'s options: `kwargs` crosses once as a raw dict (deprecated kwargs are
/// migrated here), and the resolved configuration comes back as the dict `session.options` views.
///
/// Before the driver holds them: the raw kwargs, the session layer they became (plus every
/// `update_options` so far), and the diagnostics not yet emitted.
#[lk_agents_abi::class(name = "SessionOptions")]
pub struct HostOptions {
    state: std::sync::Mutex<HostOptionsState>,
}

struct HostOptionsState {
    kwargs: SessionKwargs,
    session: SessionPatch,
    /// Diagnostics (deprecated kwargs) not yet emitted; the session takes them at construction.
    diagnostics: Vec<DiagnosticEvent>,
}

impl HostOptions {
    fn lock(&self) -> std::sync::MutexGuard<'_, HostOptionsState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn session_patch(&self) -> SessionPatch {
        self.lock().session
    }

    /// The diagnostics earned so far, once.
    pub fn take_diagnostics(&self) -> Vec<DiagnosticEvent> {
        std::mem::take(&mut self.lock().diagnostics)
    }
}

/// `AgentSession(**kwargs)`'s options: `kwargs` crosses once as a raw dict (deprecated kwargs are
/// migrated here), and the resolved configuration comes back as the dict `session.options` views.
#[lk_agents_abi::host(object(name = "SessionOptions"))]
impl HostOptions {
    /// Read the kwargs once; migrate the deprecated ones; keep the session layer.
    #[host(new)]
    pub fn new(kwargs: SessionKwargs) -> Self {
        let (session, diagnostics) = session_layer(&kwargs);
        HostOptions { state: std::sync::Mutex::new(HostOptionsState { kwargs, session, diagnostics }) }
    }

    /// `update_options(..)` before the session runs, on the same `merge` the driver uses.
    #[host(method)]
    pub fn update(&self, update: UpdateKwargs) {
        let (patch, diagnostics) = update_patch(&update);
        let mut state = self.lock();
        state.session.merge(&patch);
        state.diagnostics.extend(diagnostics);
    }

    /// Outbound SIP before `start()`: no acoustic echo on a phone line, so the warmup is switched
    /// off (`None`).
    #[host(method)]
    pub fn set_aec_warmup(&self, #[host(required)] duration: Option<f64>) {
        self.lock().session.merge(&aec_warmup_patch(duration));
    }

    /// The resolved configuration over the host-supplied legs, as the dict `session.options` reads
    /// before `start()`.
    #[host(method)]
    pub fn resolved(&self, legs: SessionLegs) -> ResolvedSessionOptions {
        let stack = ConfigStack { session: self.session_patch(), agent: None };
        resolved_session_options(&stack.resolve(&LegSet::from(legs)), &stack.session)
    }

    /// Which STT-context kwarg won: `"stt_context_options"`, `"keyterms_options"` or `"none"`.
    #[host(getter)]
    pub fn stt_context_source(&self) -> &'static str {
        stt_context_source(&self.lock().kwargs)
    }

    /// `"turn_detection" in turn_handling`, after the deprecated `turn_detection=` migration.
    #[host(getter)]
    pub fn turn_detection_explicit(&self) -> bool {
        turn_detection_explicit(&self.lock().kwargs)
    }
}

/// `aec_warmup_duration` said again: the patch that says only that leaf.
pub fn aec_warmup_patch(duration: Option<f64>) -> SessionPatch {
    let mut p = SessionPatch::default();
    p.timers.aec_warmup = Some(Nullable::from_option(duration.map(secs)));
    p
}

/// `voice/agent_session.py::_resolve_recording_options`. A dict merges onto all-on, so
/// `record={"traces": False}` means everything except traces.
#[lk_agents_abi::host(export(name = "resolve_recording_options"))]
pub fn resolve_recording_options_for_host(record: RecordingRequest) -> RecordingOptions {
    resolve_recording_options(record)
}

/// `AgentActivity._rt_turn_detection_enabled`: [`server_turn_detection_enabled`] from the host's
/// reads. `turn_detection` is the explicit mode string (agent over session), `None` when unsaid or
/// a detector object; `streaming_detector` says it was a streaming detector.
///
/// [`server_turn_detection_enabled`]: crate::agent_activity::realtime::server_turn_detection_enabled
#[lk_agents_abi::host(export(name = "rt_server_turn_detection", kw_only))]
pub fn rt_server_turn_detection_for_host(
    caps_turn_detection: bool,
    can_disable_turn_detection: bool,
    vad: bool,
    turn_detection: Option<&str>,
    streaming_detector: bool,
    interruption_given: bool,
) -> bool {
    let side = crate::agent_activity::realtime::SessionSide {
        stt: false,
        vad,
        detection: turn_detection.and_then(|s| s.parse().ok()).unwrap_or(TurnDetection::Auto),
        streaming_detector,
        interruption_given,
        turn_detection_mode: TurnDetectionMode::Manual,
    };
    let caps = lk_agents_abi::llm::realtime::RealtimeCapabilities {
        turn_detection: caps_turn_detection,
        can_disable_turn_detection,
        ..Default::default()
    };
    crate::agent_activity::realtime::server_turn_detection_enabled(&caps, side)
}

/// `AgentActivity._resolve_interruption_detection` (`agent_activity.py:4706`): whether to build the
/// detector, as `{"build", "reason", "level", "message"}`. Constructing it stays the host's.
#[lk_agents_abi::host(export(name = "resolve_interruption_detector", kw_only))]
#[allow(clippy::too_many_arguments)]
pub fn resolve_interruption_detector_for_host(
    stt_aligned_transcript: Option<bool>,
    stt_streaming: Option<bool>,
    realtime: bool,
    rt_server_turn_detection: bool,
    has_vad: bool,
    turn_detection: Option<&str>,
    allow_interruptions: bool,
    agent_mode: Option<InterruptionMode>,
    session_mode: Option<InterruptionMode>,
    hosted: bool,
    dev_mode: bool,
) -> InterruptionGateVerdict {
    let inputs = InterruptionGateInputs {
        stt: match (stt_aligned_transcript, stt_streaming) {
            (Some(aligned_transcript), Some(streaming)) => Some(SttGate { aligned_transcript, streaming }),
            _ => None,
        },
        realtime: realtime.then_some(RealtimeGate { server_turn_detection: rt_server_turn_detection }),
        has_vad,
        // A detector object's repr, or nothing said, parses to `Auto`, as `"unset"` does.
        turn_detection: turn_detection.and_then(|s| s.parse().ok()).unwrap_or(TurnDetection::Auto),
        allow_interruptions,
        asked: [agent_mode, session_mode],
        deployment: if hosted {
            Deployment::Hosted
        } else if dev_mode {
            Deployment::Dev
        } else {
            Deployment::Production
        },
    };
    resolve_interruption_detector(&inputs).verdict()
}

