//! `voice/turn.py`: `TurnHandlingOptions`, resolved ([`TurnPolicy`]) and as one layer's patch
//! ([`TurnPatch`]), and `_migrate_turn_handling` from the deprecated keywords.

use crate::agent_activity::options::SessionPatch;
use crate::audio_recognition::TurnDetectionMode;
use lk_agents_abi::Maybe;
use lk_agents_abi::voice::session_options::{
    DeprecatedTurnKwargs, EndpointingKwargs, EndpointingMode, InterruptionKwargs, InterruptionMode,
    PreemptiveKwargs, TurnDetection, TurnDetectionKwarg, TurnHandlingKwargs, UserTurnLimitKwargs,
};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnHandlingOptions {
    pub detection: TurnDetection,
    /// What the session runs (four-way); what the turn machine reads.
    pub owner: TurnDetectionMode,
    pub endpointing: EndpointingOptions,
    pub interruption: InterruptionOptions,
    pub preemptive: PreemptiveGenerationOptions,
    pub user_turn_limit: UserTurnLimitOptions,
    /// `ivr_detection`: a turn's outcome is also fed to the IVR machine.
    pub ivr_detection: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndpointingOptions {
    pub mode: EndpointingMode,
    pub min_delay: Duration,
    pub max_delay: Duration,
    /// The dynamic mode's smoothing factor. Dimensionless, so the one `f64` that stays.
    pub alpha: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterruptionOptions {
    pub enabled: bool,
    /// `None` is "the session picks the best available strategy" -- a state no concrete mode names.
    pub mode: Option<InterruptionMode>,
    /// Resolved from `legs.adaptive_detector`; not settable.
    pub detection_enabled: bool,
    /// Resolved from `detection`: off under `Manual` and `RealtimeLlm`; not settable.
    pub by_audio_activity: bool,
    pub discard_audio_if_uninterruptible: bool,
    pub min_duration: Duration,
    pub min_words: u32,
    pub resume_false_interruption: bool,
    pub false_interruption_timeout: Option<Duration>,
    /// `(start, end)`: the scalar form is widened to the pair at the binding edge.
    pub backchannel_boundary: Option<(Duration, Duration)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreemptiveGenerationOptions {
    pub enabled: bool,
    pub preemptive_tts: bool,
    pub max_speech_duration: Duration,
    pub max_retries: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserTurnLimitOptions {
    pub max_words: Option<u32>,
    pub max_duration: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnPatch {
    pub detection: Option<TurnDetection>,
    pub endpointing: EndpointingPatch,
    pub interruption: InterruptionPatch,
    pub preemptive: PreemptivePatch,
    pub user_turn_limit: UserTurnLimitPatch,
    pub ivr_detection: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointingPatch {
    pub mode: Option<EndpointingMode>,
    pub min_delay: Option<Duration>,
    pub max_delay: Option<Duration>,
    pub alpha: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterruptionPatch {
    pub enabled: Option<bool>,
    pub mode: Option<InterruptionMode>,
    pub discard_audio_if_uninterruptible: Option<bool>,
    pub min_duration: Option<Duration>,
    pub min_words: Option<u32>,
    pub resume_false_interruption: Option<bool>,
    pub false_interruption_timeout: Option<Nullable<Duration>>,
    pub backchannel_boundary: Option<Nullable<(Duration, Duration)>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreemptivePatch {
    pub enabled: Option<bool>,
    pub preemptive_tts: Option<bool>,
    pub max_speech_duration: Option<Duration>,
    pub max_retries: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserTurnLimitPatch {
    pub max_words: Option<Nullable<u32>>,
    pub max_duration: Option<Nullable<Duration>>,
}

/// A patch leaf whose value may be "off": `None` -- unsaid; `Some(Null)` -- off; `Some(Value(v))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Nullable<T> {
    Null,
    Value(T),
}

impl<T> Nullable<T> {
    pub fn into_option(self) -> Option<T> {
        match self {
            Nullable::Null => None,
            Nullable::Value(v) => Some(v),
        }
    }

    pub fn from_option(o: Option<T>) -> Self {
        match o {
            None => Nullable::Null,
            Some(v) => Nullable::Value(v),
        }
    }
}

pub fn turn_handling_patch(kw: &TurnHandlingKwargs) -> SessionPatch {
    let TurnHandlingKwargs {
        endpointing,
        interruption,
        turn_detection,
        preemptive_generation,
        user_turn_limit,
    } = *kw;
    let mut p = SessionPatch::default();
    if let Some(e) = endpointing {
        p.turn.endpointing = endpointing_patch(&e);
    }
    if let Some(i) = interruption {
        p.turn.interruption = interruption_patch(&i);
    }
    if let Some(g) = preemptive_generation {
        p.turn.preemptive = preemptive_patch(&g);
    }
    if let Some(u) = user_turn_limit {
        p.turn.user_turn_limit = user_turn_limit_patch(&u);
    }
    p.turn.detection = turn_detection.and_then(TurnDetectionKwarg::detection);
    p
}

/// The leaves an `Agent(turn_handling=..)` may override: endpointing, `interruption` enabled and
/// mode, preemptive generation, and a `turn_detection` literal. The turn limit is session-only.
pub fn agent_patch(kw: &TurnHandlingKwargs) -> SessionPatch {
    let mut p = turn_handling_patch(&TurnHandlingKwargs { user_turn_limit: None, ..*kw });
    let InterruptionPatch { enabled, mode, .. } = p.turn.interruption;
    p.turn.interruption = InterruptionPatch { enabled, mode, ..InterruptionPatch::default() };
    p
}

pub(crate) fn endpointing_patch(kw: &EndpointingKwargs) -> EndpointingPatch {
    EndpointingPatch {
        mode: kw.mode,
        min_delay: kw.min_delay.map(secs),
        max_delay: kw.max_delay.map(secs),
        alpha: kw.alpha,
    }
}

fn interruption_patch(kw: &InterruptionKwargs) -> InterruptionPatch {
    InterruptionPatch {
        enabled: kw.enabled,
        mode: kw.mode,
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

fn preemptive_patch(kw: &PreemptiveKwargs) -> PreemptivePatch {
    PreemptivePatch {
        enabled: kw.enabled,
        preemptive_tts: kw.preemptive_tts,
        max_speech_duration: kw.max_speech_duration.map(secs),
        max_retries: kw.max_retries.map(count),
    }
}

fn user_turn_limit_patch(kw: &UserTurnLimitKwargs) -> UserTurnLimitPatch {
    UserTurnLimitPatch {
        max_words: nullable(kw.max_words, count),
        max_duration: nullable(kw.max_duration, secs),
    }
}

/// `_migrate_turn_handling`: eleven deprecated kwargs to one **sparse** [`TurnHandlingKwargs`].
/// Filling defaults would make them explicit and override an `Agent`'s own endpointing.
pub fn migrate_turn_handling(kw: &DeprecatedTurnKwargs) -> TurnHandlingKwargs {
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
    } = *kw;

    let mut out = TurnHandlingKwargs::default();

    let endpointing = EndpointingKwargs {
        mode: None,
        min_delay: min_endpointing_delay.given(),
        max_delay: max_endpointing_delay.given(),
        alpha: None,
    };
    if !endpointing.is_empty() {
        out.endpointing = Some(endpointing);
    }

    let interruption = InterruptionKwargs {
        // `if allow_interruptions is False`: `True` writes nothing.
        enabled: match allow_interruptions {
            Maybe::Given(false) => Some(false),
            Maybe::Given(true) | Maybe::NotGiven => None,
        },
        mode: None,
        discard_audio_if_uninterruptible: discard_audio_if_uninterruptible.given(),
        min_duration: min_interruption_duration.given(),
        min_words: min_interruption_words.given(),
        resume_false_interruption: resume_false_interruption.given(),
        // The newer name wins, silently, when both are given: a plain overwrite in the reference.
        false_interruption_timeout: match agent_false_interruption_timeout {
            Maybe::Given(v) => Maybe::Given(v),
            Maybe::NotGiven => false_interruption_timeout,
        },
        backchannel_boundary: Maybe::NotGiven,
    };
    if !interruption.is_empty() {
        out.interruption = Some(interruption);
    }

    out.turn_detection = turn_detection.given();

    if let Maybe::Given(enabled) = preemptive_generation {
        out.preemptive_generation = Some(PreemptiveKwargs {
            enabled: Some(enabled),
            preemptive_tts: None,
            max_speech_duration: None,
            max_retries: None,
        });
    }

    out
}

/// The deprecated `turn_detection=` as `migrate_turn_handling` takes it: what it means, and the
/// caller's own object (a mode string too), written back as is. `None` is given: a `Null` kwarg and
/// no object.
#[derive(Debug, Clone)]
pub struct TurnDetectionArg(pub TurnDetectionKwarg, pub Option<lk_agents_abi::HostRef>);

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for TurnDetectionArg {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods as _;
        let kwarg = obj.extract::<TurnDetectionKwarg>()?;
        let object = if obj.is_none() { None } else { Some(lk_agents_abi::python::intern_host(obj.to_owned().unbind())) };
        Ok(TurnDetectionArg(kwarg, object))
    }
}

/// `voice/turn.py::_migrate_turn_handling`, for `Agent.__init__`, which stores the sparse dict.
/// `turn_detection` is the caller's own object, written back as is.
#[lk_agents_abi::host(export(name = "migrate_turn_handling", kw_only))]
#[allow(clippy::too_many_arguments)]
pub fn migrate_turn_handling_for_host(
    min_endpointing_delay: Maybe<f64>,
    max_endpointing_delay: Maybe<f64>,
    false_interruption_timeout: Maybe<Option<f64>>,
    turn_detection: Maybe<TurnDetectionArg>,
    discard_audio_if_uninterruptible: Maybe<bool>,
    min_interruption_duration: Maybe<f64>,
    min_interruption_words: Maybe<i64>,
    allow_interruptions: Maybe<bool>,
    resume_false_interruption: Maybe<bool>,
    agent_false_interruption_timeout: Maybe<Option<f64>>,
    preemptive_generation: Maybe<bool>,
) -> lk_agents_abi::voice::session_options::MigratedTurnHandling {
    let (detection, object) = match turn_detection {
        Maybe::NotGiven => (Maybe::NotGiven, None),
        Maybe::Given(TurnDetectionArg(kwarg, object)) => (Maybe::Given(kwarg), Some(object)),
    };
    let out = migrate_turn_handling(&DeprecatedTurnKwargs {
        min_endpointing_delay,
        max_endpointing_delay,
        false_interruption_timeout,
        turn_detection: detection,
        discard_audio_if_uninterruptible,
        min_interruption_duration,
        min_interruption_words,
        allow_interruptions,
        resume_false_interruption,
        agent_false_interruption_timeout,
        preemptive_generation,
    });
    // Keys, order and absences are the declaration's; only the caller's object is filled in.
    lk_agents_abi::voice::session_options::MigratedTurnHandling {
        endpointing: out.endpointing,
        interruption: out.interruption,
        turn_detection: object,
        preemptive_generation: out.preemptive_generation,
    }
}

/// Seconds to a `Duration`, clamping negatives (`from_secs_f64` panics below zero).
pub fn secs(v: f64) -> Duration {
    Duration::from_secs_f64(v.max(0.0))
}

/// A count, clamped: wrapping a negative would disable a guard.
pub(crate) fn count(v: i64) -> u32 {
    u32::try_from(v.max(0)).unwrap_or(u32::MAX)
}

/// A `Maybe<Option<T>>` leaf as a patch leaf: unsaid stays unsaid, `None` is "said off".
pub(crate) fn nullable<T, U>(v: Maybe<Option<T>>, f: impl FnOnce(T) -> U) -> Option<Nullable<U>> {
    match v {
        Maybe::NotGiven => None,
        Maybe::Given(None) => Some(Nullable::Null),
        Maybe::Given(Some(t)) => Some(Nullable::Value(f(t))),
    }
}
