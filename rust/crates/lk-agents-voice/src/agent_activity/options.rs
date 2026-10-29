//! The session's and the agent's options (the `AgentSession` keywords, `turn_handling` among them
//! in [`crate::turn`]): the layers, their defaults and resolution, and the host's view.

use crate::audio_recognition::TurnDetectionMode;
use crate::events::DiagnosticEvent;
use crate::turn::{
    EndpointingOptions, InterruptionOptions, PreemptiveGenerationOptions, TurnHandlingOptions,
    UserTurnLimitOptions, count, endpointing_options, interruption_options, nullable,
    turn_handling_layer,
};
use lk_agents_abi::{Maybe, Vocabulary, secs};
pub use lk_agents_abi::voice::session_options::{
    DetectorKind, EndpointingKwargs, EndpointingMode, EndpointingOverrides, Expressive, HostValue,
    InterruptionGateVerdict, InterruptionMode, LegSet, RecordingOptions, ResolvedEndpointing,
    ResolvedInterruption, ResolvedPreemptiveGeneration, ResolvedSessionOptions,
    ResolvedUserTurnLimit, SessionKwargs, SessionLegs, TextTransformSet, TurnDetection,
    TurnDetectionKwarg, TurnHandlingKwargs, UpdateKwargs,
};
use std::time::Duration;

/// `_RECORDING_ALL_ON`. Note `redaction: false`: "all on" is about the four capture keys.
pub const RECORDING_ALL_ON: RecordingOptions =
    RecordingOptions { audio: true, traces: true, logs: true, transcript: true, redaction: false };

/// `_RECORDING_ALL_OFF`.
pub const RECORDING_ALL_OFF: RecordingOptions = RecordingOptions {
    audio: false,
    traces: false,
    logs: false,
    transcript: false,
    redaction: false,
};

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

/// One leaf of an option group: `None` / `NotGiven` is unsaid.
pub trait Leaf: Copy {
    type Value;
    fn is_said(&self) -> bool;
    /// The value of a resolved leaf, which `ConfigStack::resolve` guarantees is said.
    fn resolved(self) -> Self::Value;
}

impl<T: Copy> Leaf for Option<T> {
    type Value = T;
    fn is_said(&self) -> bool {
        self.is_some()
    }
    fn resolved(self) -> T {
        self.expect("an option leaf read before resolve")
    }
}

impl<T: Copy> Leaf for Maybe<Option<T>> {
    type Value = Option<T>;
    fn is_said(&self) -> bool {
        self.is_given()
    }
    fn resolved(self) -> Option<T> {
        self.given().expect("an option leaf read before resolve")
    }
}

/// An option group's `merge` (a said leaf wins, an unsaid one keeps) and its resolved reads, one
/// per leaf. `optional` leaves have no read: their `None` is a resolved answer.
macro_rules! option_group {
    ($ty:ident { $($leaf:ident: $value:ty),* $(,)? }
        $(optional($($opt:ident),+))? $(groups($($group:ident),+))?) => {
        impl $ty {
            /// Lay `over` on top: a said leaf wins, an unsaid one keeps its answer.
            pub fn merge(&mut self, over: &Self) {
                $(
                    if $crate::agent_activity::options::Leaf::is_said(&over.$leaf) {
                        self.$leaf = over.$leaf;
                    }
                )*
                $($(if over.$opt.is_some() { self.$opt = over.$opt; })+)?
                $($(self.$group.merge(&over.$group);)+)?
            }
            $(
                pub fn $leaf(&self) -> $value {
                    $crate::agent_activity::options::Leaf::resolved(self.$leaf)
                }
            )*
        }
    };
}
pub(crate) use option_group;

/// The session's options, one layer or all of them resolved. `Default` says nothing.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AgentSessionOptions {
    pub turn: TurnHandlingOptions,
    pub speech: SpeechOptions,
    pub timers: TimerOptions,
}

option_group!(AgentSessionOptions {} groups(turn, speech, timers));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpeechOptions {
    pub max_tool_steps: Option<u32>,
    pub min_consecutive_speech_delay: Option<Duration>,
    /// `use_tts_aligned_transcript`: `None` is the reference's `NOT_GIVEN`, "if the TTS can",
    /// resolved too.
    pub use_tts_aligned_transcript: Option<bool>,
    /// `tts_text_transforms`: `None` is "run none".
    pub tts_text_transforms: Maybe<Option<TextTransformSet>>,
}

option_group!(SpeechOptions {
    max_tool_steps: u32,
    min_consecutive_speech_delay: Duration,
    tts_text_transforms: Option<TextTransformSet>,
} optional(use_tts_aligned_transcript));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TimerOptions {
    pub user_away: Maybe<Option<Duration>>,
    /// `Some(ZERO)` is a region ("fire at end of speech"), not "off".
    pub transcription_timeout: Maybe<Option<Duration>>,
    /// `None` is no warmup at all (an outbound SIP call).
    pub aec_warmup: Maybe<Option<Duration>>,
    pub session_close_transcript_timeout: Option<Duration>,
}

option_group!(TimerOptions {
    user_away: Option<Duration>,
    transcription_timeout: Option<Duration>,
    aec_warmup: Option<Duration>,
    session_close_transcript_timeout: Duration,
});

/// The reference's defaults: every leaf said. The endpointing base tightens to `0.3 / 2.5` with a
/// streaming detector, and the owner is what `Auto` means for these legs.
pub const fn defaults(legs: &LegSet) -> AgentSessionOptions {
    let (min_delay, max_delay) = match legs.detector {
        Some(DetectorKind::Streaming) => (Duration::from_millis(300), Duration::from_millis(2500)),
        None => (Duration::from_millis(500), Duration::from_millis(3000)),
    };
    let detection = TurnDetection::Auto;
    AgentSessionOptions {
        turn: TurnHandlingOptions {
            detection: Some(detection),
            owner: Some(resolve_turn_detection_mode(detection, legs)),
            endpointing: EndpointingOptions {
                mode: Some(EndpointingMode::Fixed),
                min_delay: Some(min_delay),
                max_delay: Some(max_delay),
                alpha: Some(0.9),
            },
            interruption: InterruptionOptions {
                enabled: Some(true),
                mode: None,
                detection_enabled: Some(legs.adaptive_detector),
                by_audio_activity: Some(by_audio_activity(detection)),
                discard_audio_if_uninterruptible: Some(true),
                min_duration: Some(Duration::from_millis(500)),
                min_words: Some(0),
                resume_false_interruption: Some(true),
                false_interruption_timeout: Maybe::Given(Some(Duration::from_millis(2000))),
                backchannel_boundary: Maybe::Given(Some((
                    Duration::from_millis(1000),
                    Duration::from_millis(1000),
                ))),
            },
            preemptive: PreemptiveGenerationOptions {
                enabled: Some(true),
                preemptive_tts: Some(false),
                max_speech_duration: Some(Duration::from_millis(10_000)),
                max_retries: Some(3),
            },
            user_turn_limit: UserTurnLimitOptions {
                max_words: Maybe::Given(None),
                max_duration: Maybe::Given(None),
            },
            ivr_detection: Some(false),
        },
        speech: SpeechOptions {
            max_tool_steps: Some(3),
            min_consecutive_speech_delay: Some(Duration::ZERO),
            use_tts_aligned_transcript: None,
            tts_text_transforms: Maybe::Given(Some(TextTransformSet::ALL)),
        },
        timers: TimerOptions {
            user_away: Maybe::Given(Some(Duration::from_millis(15_000))),
            transcription_timeout: Maybe::Given(None),
            aec_warmup: Maybe::Given(Some(Duration::from_millis(3000))),
            session_close_transcript_timeout: Some(Duration::from_millis(2000)),
        },
    }
}
/// `Auto` falls `realtime_llm` → `vad` → `manual`. A mode the legs cannot honour falls back as
/// `Auto` does; `Manual` always holds. `RealtimeLlm` only when no local leg can decide;
/// `DriverBuilder::start` refines it. Only an asked-for `stt` is `Stt`: the reference's unset mode
/// (`_turn_detection_mode = None`, the default turn detector included) is its VAD-based path, which
/// with an STT and no VAD ends turns on finals alone and leaves the STT's speech events unread.
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
    } else if legs.vad || legs.stt {
        TurnDetectionMode::Vad
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
    /// `__init__`, each `update_options` merged in.
    pub session: AgentSessionOptions,
    /// The active `Agent`'s overrides; replaced whole by `update_agent`.
    pub agent: Option<AgentSessionOptions>,
}

impl ConfigStack {
    /// Pure. The only place a default is read, and every leaf of its answer is said.
    pub fn resolve(&self, legs: &LegSet) -> AgentSessionOptions {
        let mut cfg = defaults(legs);
        cfg.merge(&self.session);
        if let Some(agent) = &self.agent {
            cfg.merge(agent);
        }
        // The derived leaves follow the layered `detection` and the legs, whatever a layer said.
        let detection = cfg.turn.detection();
        cfg.turn.owner = Some(resolve_turn_detection_mode(detection, legs));
        cfg.turn.interruption.by_audio_activity = Some(by_audio_activity(detection));
        cfg.turn.interruption.detection_enabled = Some(legs.adaptive_detector);
        cfg
    }
}

/// The `turn_handling` this session runs on, as written.
pub fn session_turn_handling(kw: &SessionKwargs) -> TurnHandlingKwargs {
    kw.turn_handling.unwrap_or_default()
}

/// **The session layer**: what it leaves unsaid falls to `defaults(legs)`.
/// Past this line nothing is `f64`.
pub fn session_layer(kw: &SessionKwargs) -> AgentSessionOptions {
    let SessionKwargs {
        turn_handling: _,
        stt_context_options: _,
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
    } = kw.clone();

    let mut o = turn_handling_layer(&session_turn_handling(kw));
    o.turn.ivr_detection = Some(ivr_detection);
    o.speech.max_tool_steps = Some(count(max_tool_steps));
    o.speech.min_consecutive_speech_delay = Some(secs(min_consecutive_speech_delay));
    o.speech.use_tts_aligned_transcript = use_tts_aligned_transcript.given();
    o.speech.tts_text_transforms = tts_text_transforms;
    o.timers.user_away = Maybe::Given(user_away_timeout.map(secs));
    o.timers.transcription_timeout = Maybe::Given(transcription_timeout.map(secs));
    o.timers.aec_warmup = nullable(aec_warmup_duration, secs);
    o.timers.session_close_transcript_timeout = Some(secs(session_close_transcript_timeout));
    o
}

/// What `update_options(..)` says.
pub fn update_layer(kw: &UpdateKwargs) -> AgentSessionOptions {
    let UpdateKwargs {
        endpointing_opts,
        turn_detection,
        expressive: _,
        keyterms: _,
        interruption,
        min_consecutive_speech_delay,
        tts_text_transforms,
    } = kw.clone();
    let mut o = AgentSessionOptions::default();
    if let Some(e) = endpointing_opts {
        o.turn.endpointing = endpointing_options(&e);
    }
    if let Some(i) = interruption {
        o.turn.interruption = interruption_options(&i);
    }
    o.speech.min_consecutive_speech_delay = min_consecutive_speech_delay.map(secs);
    o.speech.tts_text_transforms = tts_text_transforms;
    o.turn.detection = match turn_detection {
        Maybe::NotGiven => None,
        Maybe::Given(TurnDetectionKwarg::Null) => Some(TurnDetection::Auto),
        Maybe::Given(TurnDetectionKwarg::Mode(m)) => Some(m),
        // A detector object: refused by `update_turn_options` before this.
        Maybe::Given(TurnDetectionKwarg::Detector { streaming: _ }) => None,
    };
    o
}

/// The resolved options as the host's `ResolvedSessionOptions`. `session` is read for
/// `endpointing_overrides` only. No `..`, so a new leaf must be named.
pub fn resolved_session_options(
    cfg: &AgentSessionOptions,
    session: &AgentSessionOptions,
) -> ResolvedSessionOptions {
    let AgentSessionOptions {
        turn:
            TurnHandlingOptions {
                detection,
                owner: _,
                endpointing: EndpointingOptions { mode, min_delay, max_delay, alpha },
                interruption:
                    InterruptionOptions {
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
                    PreemptiveGenerationOptions {
                        enabled: preemptive_enabled,
                        preemptive_tts,
                        max_speech_duration,
                        max_retries,
                    },
                user_turn_limit: UserTurnLimitOptions { max_words, max_duration },
                ivr_detection,
            },
        speech:
            SpeechOptions {
                max_tool_steps,
                min_consecutive_speech_delay,
                use_tts_aligned_transcript,
                tts_text_transforms,
            },
        timers:
            TimerOptions {
                user_away,
                transcription_timeout,
                aec_warmup,
                session_close_transcript_timeout,
            },
    } = *cfg;
    let secs = |d: Duration| d.as_secs_f64();
    let detection = detection.resolved();
    let overrides = session.turn.endpointing;
    ResolvedSessionOptions {
        endpointing: ResolvedEndpointing {
            mode: mode.resolved().as_str().to_owned(),
            min_delay: secs(min_delay.resolved()),
            max_delay: secs(max_delay.resolved()),
            alpha: alpha.resolved(),
        },
        interruption: ResolvedInterruption {
            enabled: enabled.resolved(),
            mode: interruption_mode.map(|m| m.as_str().to_owned()),
            discard_audio_if_uninterruptible: discard_audio_if_uninterruptible.resolved(),
            min_duration: secs(min_duration.resolved()),
            min_words: min_words.resolved(),
            resume_false_interruption: resume_false_interruption.resolved(),
            false_interruption_timeout: false_interruption_timeout.resolved().map(secs),
            backchannel_boundary: backchannel_boundary.resolved().map(|(s, e)| (secs(s), secs(e))),
        },
        preemptive_generation: ResolvedPreemptiveGeneration {
            enabled: preemptive_enabled.resolved(),
            preemptive_tts: preemptive_tts.resolved(),
            max_speech_duration: secs(max_speech_duration.resolved()),
            max_retries: max_retries.resolved(),
        },
        user_turn_limit: ResolvedUserTurnLimit {
            max_words: max_words.resolved(),
            max_duration: max_duration.resolved().map(secs),
        },
        // The literal, or `None` for `Auto`: "from the legs".
        turn_detection_mode: (detection != TurnDetection::Auto)
            .then(|| detection.as_str().to_owned()),
        endpointing_overrides: EndpointingOverrides {
            mode: overrides.mode.map(|m| m.as_str().to_owned()),
            min_delay: overrides.min_delay.map(secs),
            max_delay: overrides.max_delay.map(secs),
            alpha: overrides.alpha,
        },
        max_tool_steps: max_tool_steps.resolved(),
        min_consecutive_speech_delay: secs(min_consecutive_speech_delay.resolved()),
        use_tts_aligned_transcript,
        tts_text_transforms: tts_text_transforms.resolved().map(TextTransformSet::names),
        ivr_detection: ivr_detection.resolved(),
        user_away_timeout: user_away.resolved().map(secs),
        transcription_timeout: transcription_timeout.resolved().map(secs),
        aec_warmup_duration: aec_warmup.resolved().map(secs),
        session_close_transcript_timeout: secs(session_close_transcript_timeout.resolved()),
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
    VadOnly {
        reason: &'static str,
        diagnostic: Option<DiagnosticEvent>,
    },
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

/// `aec_warmup_duration` said again: the layer that says only that leaf.
pub fn aec_warmup_layer(duration: Option<f64>) -> AgentSessionOptions {
    let mut p = AgentSessionOptions::default();
    p.timers.aec_warmup = Maybe::Given(duration.map(secs));
    p
}

/// The interruption detector's gates, as the host asks them.
pub struct InterruptionGateRules;

impl lk_agents_abi::voice::session_options::InterruptionGateFunctions for InterruptionGateRules {
    fn rt_server_turn_detection(
        caps_turn_detection: bool,
        can_disable_turn_detection: bool,
        vad: bool,
        turn_detection: Option<String>,
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

    fn resolve_interruption_detector(
        stt_aligned_transcript: Option<bool>,
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
    ) -> InterruptionGateVerdict {
        let inputs = InterruptionGateInputs {
            stt: match (stt_aligned_transcript, stt_streaming) {
                (Some(aligned_transcript), Some(streaming)) => {
                    Some(SttGate { aligned_transcript, streaming })
                }
                _ => None,
            },
            realtime: realtime
                .then_some(RealtimeGate { server_turn_detection: rt_server_turn_detection }),
            has_vad,
            // A detector object's repr, or nothing said, parses to `Auto`, as `"unset"` does.
            turn_detection: turn_detection
                .and_then(|s| s.parse().ok())
                .unwrap_or(TurnDetection::Auto),
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
}

#[cfg(test)]
#[path = "options_tests.rs"]
pub(crate) mod tests;
