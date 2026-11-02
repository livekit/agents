//! Ordered outcomes of recognition transitions. Recognition reports domain facts and requests;
//! the voice driver owns timer execution, deferred callbacks, and runtime delivery.

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
    EndpointingDelayUpdated { min_delay: Duration },
    TurnDeclared { owner: TurnDetectionMode },
}
