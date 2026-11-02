//! Recognition observations retained for diagnostics. These records never dispatch work.

use super::{EndOfTurnInfo, TurnDetectionMode};
use crate::agent_activity::false_interruption::{AudioInterruptTrigger, TurnSettled};
use crate::agent_session::effects::AudioInterruptGuard;

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum RecognitionObservation {
    /// `decide_commit`, tell the host, answer the machine, and maybe reply.
    EndOfTurn(EndOfTurnInfo),
    TurnSettled(TurnSettled),
    AudioInterruptGuard(AudioInterruptGuard),
    /// Still subject to `decide_interrupt`'s guards.
    InterruptByAudioActivity(AudioInterruptTrigger),
    /// Journal-only.
    EndpointingDelayUpdated {
        min_delay: std::time::Duration,
    },
    /// Even when a VAD end already closed the user's speech. Journal-only.
    TurnDeclared {
        owner: TurnDetectionMode,
    },
}
