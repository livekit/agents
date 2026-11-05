//! `voice/turn.py`: `TurnHandlingOptions`. One struct per group is both what a layer says (`None`
//! and `NotGiven` are unsaid) and the resolved answer, where every leaf is said.

use crate::agent_activity::options::{AgentSessionOptions, option_group};
use crate::audio_recognition::TurnDetectionMode;
use lk_agents_abi::voice::session_options::{
    EndpointingKwargs, EndpointingMode, InterruptionKwargs, InterruptionMode, PreemptiveKwargs,
    TurnDetection, TurnDetectionKwarg, TurnHandlingKwargs, UserTurnLimitKwargs,
};
use lk_agents_abi::{Maybe, secs};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TurnHandlingOptions {
    pub detection: Option<TurnDetection>,
    /// What the session runs (four-way); what the turn machine reads. Derived from `detection`
    /// and the legs by `resolve`, never said by a layer.
    pub owner: Option<TurnDetectionMode>,
    pub endpointing: EndpointingOptions,
    pub interruption: InterruptionOptions,
    pub preemptive: PreemptiveGenerationOptions,
    pub user_turn_limit: UserTurnLimitOptions,
    /// `ivr_detection`: a turn's outcome is also fed to the IVR machine.
    pub ivr_detection: Option<bool>,
}

option_group!(TurnHandlingOptions {
    detection: TurnDetection,
    owner: TurnDetectionMode,
    ivr_detection: bool,
} groups(endpointing, interruption, preemptive, user_turn_limit));

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EndpointingOptions {
    pub mode: Option<EndpointingMode>,
    pub min_delay: Option<Duration>,
    pub max_delay: Option<Duration>,
    /// The dynamic mode's smoothing factor. Dimensionless, so the one `f64` that stays.
    pub alpha: Option<f64>,
}

option_group!(EndpointingOptions {
    mode: EndpointingMode,
    min_delay: Duration,
    max_delay: Duration,
    alpha: f64,
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InterruptionOptions {
    pub enabled: Option<bool>,
    /// `None` is "the session picks the best available strategy", resolved too.
    pub mode: Option<InterruptionMode>,
    /// Resolved from `legs.adaptive_detector`; not settable.
    pub detection_enabled: Option<bool>,
    /// Resolved from `detection`: off under `Manual` and `RealtimeLlm`; not settable.
    pub by_audio_activity: Option<bool>,
    pub discard_audio_if_uninterruptible: Option<bool>,
    pub min_duration: Option<Duration>,
    pub min_words: Option<u32>,
    pub resume_false_interruption: Option<bool>,
    pub false_interruption_timeout: Maybe<Option<Duration>>,
    /// `(start, end)`: the scalar form is widened to the pair at the binding edge.
    pub backchannel_boundary: Maybe<Option<(Duration, Duration)>>,
}

option_group!(InterruptionOptions {
    enabled: bool,
    detection_enabled: bool,
    by_audio_activity: bool,
    discard_audio_if_uninterruptible: bool,
    min_duration: Duration,
    min_words: u32,
    resume_false_interruption: bool,
    false_interruption_timeout: Option<Duration>,
    backchannel_boundary: Option<(Duration, Duration)>,
} optional(mode));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PreemptiveGenerationOptions {
    pub enabled: Option<bool>,
    pub preemptive_tts: Option<bool>,
    pub max_speech_duration: Option<Duration>,
    pub max_retries: Option<u32>,
}

option_group!(PreemptiveGenerationOptions {
    enabled: bool,
    preemptive_tts: bool,
    max_speech_duration: Duration,
    max_retries: u32,
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UserTurnLimitOptions {
    pub max_words: Maybe<Option<u32>>,
    pub max_duration: Maybe<Option<Duration>>,
}

option_group!(UserTurnLimitOptions {
    max_words: Option<u32>,
    max_duration: Option<Duration>,
});

/// `turn_handling=` as the layer it says.
pub fn turn_handling_layer(kw: &TurnHandlingKwargs) -> AgentSessionOptions {
    let TurnHandlingKwargs {
        endpointing,
        interruption,
        turn_detection,
        preemptive_generation,
        user_turn_limit,
    } = *kw;
    let mut o = AgentSessionOptions::default();
    if let Some(e) = endpointing {
        o.turn.endpointing = endpointing_options(&e);
    }
    if let Some(i) = interruption {
        o.turn.interruption = interruption_options(&i);
    }
    if let Some(g) = preemptive_generation {
        o.turn.preemptive = preemptive_options(&g);
    }
    if let Some(u) = user_turn_limit {
        o.turn.user_turn_limit = user_turn_limit_options(&u);
    }
    o.turn.detection = turn_detection.given().and_then(TurnDetectionKwarg::detection);
    o
}

/// The leaves an `Agent` may override: from its `turn_handling`, endpointing, `interruption`
/// enabled and mode, preemptive generation and a `turn_detection` literal (the turn limit is
/// session-only); its `min_consecutive_speech_delay` (seconds) and `use_tts_aligned_transcript`.
/// An absent leaf defers to the session's.
pub fn agent_layer(
    turn_handling: Option<&TurnHandlingKwargs>,
    min_consecutive_speech_delay: Option<f64>,
    use_tts_aligned_transcript: Option<bool>,
) -> AgentSessionOptions {
    let mut o = turn_handling
        .map(|kw| turn_handling_layer(&TurnHandlingKwargs { user_turn_limit: None, ..*kw }))
        .unwrap_or_default();
    // `Agent(turn_detection=None)` is given: automatic selection, not the session's mode.
    if turn_handling
        .is_some_and(|kw| matches!(kw.turn_detection, Maybe::Given(TurnDetectionKwarg::Null)))
    {
        o.turn.detection = Some(TurnDetection::Auto);
    }
    let InterruptionOptions { enabled, mode, .. } = o.turn.interruption;
    o.turn.interruption = InterruptionOptions { enabled, mode, ..InterruptionOptions::default() };
    o.speech.min_consecutive_speech_delay =
        min_consecutive_speech_delay.and_then(|s| Duration::try_from_secs_f64(s).ok());
    o.speech.use_tts_aligned_transcript = use_tts_aligned_transcript;
    o
}

pub(crate) fn endpointing_options(kw: &EndpointingKwargs) -> EndpointingOptions {
    EndpointingOptions {
        mode: kw.mode,
        min_delay: kw.min_delay.map(secs),
        max_delay: kw.max_delay.map(secs),
        alpha: kw.alpha,
    }
}

pub(crate) fn interruption_options(kw: &InterruptionKwargs) -> InterruptionOptions {
    InterruptionOptions {
        enabled: kw.enabled,
        mode: kw.mode.given().flatten(),
        detection_enabled: None,
        by_audio_activity: None,
        discard_audio_if_uninterruptible: kw.discard_audio_if_uninterruptible,
        min_duration: kw.min_duration.map(secs),
        min_words: kw.min_words.map(count),
        resume_false_interruption: kw.resume_false_interruption,
        false_interruption_timeout: nullable(kw.false_interruption_timeout, secs),
        backchannel_boundary: nullable(kw.backchannel_boundary, |b| {
            (secs(b.start()), secs(b.end()))
        }),
    }
}

fn preemptive_options(kw: &PreemptiveKwargs) -> PreemptiveGenerationOptions {
    PreemptiveGenerationOptions {
        enabled: kw.enabled,
        preemptive_tts: kw.preemptive_tts,
        max_speech_duration: kw.max_speech_duration.map(secs),
        max_retries: kw.max_retries.map(count),
    }
}

fn user_turn_limit_options(kw: &UserTurnLimitKwargs) -> UserTurnLimitOptions {
    UserTurnLimitOptions {
        max_words: nullable(kw.max_words, count),
        max_duration: nullable(kw.max_duration, secs),
    }
}

/// A count, clamped: wrapping a negative would disable a guard.
pub(crate) fn count(v: i64) -> u32 {
    u32::try_from(v.max(0)).unwrap_or(u32::MAX)
}

/// A nullable kwarg's value mapped: unsaid stays unsaid, `None` is "said off".
pub(crate) fn nullable<T, U>(v: Maybe<Option<T>>, f: impl FnOnce(T) -> U) -> Maybe<Option<U>> {
    match v {
        Maybe::NotGiven => Maybe::NotGiven,
        Maybe::Given(o) => Maybe::Given(o.map(f)),
    }
}
