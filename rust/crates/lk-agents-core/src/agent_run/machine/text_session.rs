//! Session composition points. A parked invocation owns no host work: all non-delegated calls
//! must have settled before its history can replace the invocation in saved routing state.

use lk_agents_abi::{ChatItem, Ts};

use super::{AgentRun, AgentRunEffect, CallId, Phase};

pub enum DelegationCheckpoint {
    Waiting,
    ConflictingOutcome,
    Ready { history: Vec<ChatItem> },
}

impl AgentRun {
    /// Starts a fresh invocation from a history whose pending calls have already been answered.
    /// Unlike `Start`, this appends no synthetic user message and answers no human-input tools.
    pub fn continue_from_history(&mut self, now: Ts) -> Vec<AgentRunEffect> {
        if !matches!(self.phase, Phase::Idle) {
            return Vec::new();
        }
        self.history = std::mem::take(&mut self.input.history);
        self.first_new = self.history.len();
        let mut effects = Vec::new();
        self.next_step(now, &mut effects);
        effects
    }

    /// Returns a checkpoint only when the named local child calls are the sole unfinished
    /// work. The caller takes ownership of these calls before dropping this invocation.
    pub fn delegation_checkpoint(&self, calls: &[CallId], now: Ts) -> DelegationCheckpoint {
        let Phase::Acting(acting) = &self.phase else { return DelegationCheckpoint::Waiting };
        if calls.is_empty() {
            return DelegationCheckpoint::Waiting;
        }
        if !acting.accepted.is_empty() || acting.calls.has_terminal_outcome() {
            return DelegationCheckpoint::ConflictingOutcome;
        }
        if !acting.finals.is_empty() {
            return DelegationCheckpoint::Waiting;
        }
        let mut history = self.history.clone();
        let Some(items) =
            acting.calls.delegation_checkpoint(calls, self.input.mode, self.stamp(now))
        else {
            return DelegationCheckpoint::Waiting;
        };
        history.extend(items);
        DelegationCheckpoint::Ready { history }
    }
}
