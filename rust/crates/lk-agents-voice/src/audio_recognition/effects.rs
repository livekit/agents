//! Ordered outcomes of recognition transitions. Recognition reports domain facts and requests;
//! the voice driver owns timer execution, deferred callbacks, and runtime delivery.
//!
//! [`AudioRecognition`](super::machine::AudioRecognition) produces these effects;
//! [`VoiceDriver`](crate::agent_session::machine::VoiceDriver) consumes them in transition order.

use std::time::Duration;

use lk_agents_abi::Ts;
use lk_agents_abi::turn::detector::ReqId;

use super::{EndOfTurnInfo, InterruptionSignal, TurnDetectionMode};
use crate::agent_activity::false_interruption::{AudioInterruptTrigger, TurnSettled};
use crate::agent_session::effects::{AudioInterruptGuard, FlushReason, StreamKind, Timer};
use crate::events::SessionEvent;

/// Keep outcomes in transition order: in particular, teardown precedes a replacement deadline
/// and a turn's transcript publication precedes its end-of-turn notification.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioRecognitionEffect {
    SetDeadline(Timer, Ts),
    ClearDeadline(Timer),
    FlushInput(StreamKind),
    PublishSessionEvent(SessionEvent),
    RequestEot(ReqId, Option<Duration>),
    CancelEot(ReqId, bool),
    NotifyInterruption(InterruptionSignal),
    FlushTurnDetector(FlushReason),
    EndOfTurn(EndOfTurnInfo),
    TurnSettled(TurnSettled),
    AudioInterruptGuard(AudioInterruptGuard),
    InterruptByAudioActivity(AudioInterruptTrigger),
    EndpointingDelayUpdated {
        min_delay: Duration,
    },
    TurnDeclared {
        owner: TurnDetectionMode,
    },
    /// For the countdown's `eou_detection` span.
    EouDetected(EouDetected),
}

/// What the end-of-turn detector did for a countdown.
#[derive(Debug, Clone, PartialEq)]
pub enum EouDetected {
    /// The countdown waits on the detector's prediction.
    Awaiting,
    /// The prediction arrived, or was already there (`from_cache`); `None` when it failed or
    /// timed out. `delay` is the endpointing delay it leaves the countdown.
    Resolved {
        prediction: Option<EouPredicted>,
        delay: Duration,
        from_cache: bool,
        language: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EouPredicted {
    pub probability: f64,
    /// Below it the turn is unlikely to be over; `None` when the language has none.
    pub threshold: Option<f64>,
    /// A streaming detector's latest audio to prediction, in seconds.
    pub detection_delay: Option<f64>,
}
