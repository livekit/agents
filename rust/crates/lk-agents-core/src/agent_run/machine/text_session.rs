use lk_agents_abi::{ChatItem, Ts};

use super::{AgentRun, AgentRunEffect, CallId, Phase};

// A parked invocation owns no host work: every non-delegated call must settle before its history
// can replace the invocation in saved routing state.

pub enum DelegationCheckpoint {
    Waiting,
    ConflictingOutcome,
    Ready { history: Vec<ChatItem> },
}

impl AgentRun {
    /// Starts a fresh invocation from a history whose pending calls have already been answered.
    /// This appends no synthetic user message and infers no human answer from history. Origins
    /// already admitted by the conversation owner still authorize a validated approval repeat.
    /// A detached body's native requirements use the same terminal-question presentation path.
    pub fn continue_from_history(
        &mut self,
        requirements: Vec<lk_agents_abi::agent::run::RunInputRequirement>,
        now: Ts,
    ) -> Vec<AgentRunEffect> {
        if !matches!(self.phase, Phase::Idle) {
            return Vec::new();
        }
        self.history = std::mem::take(&mut self.input.history);
        self.approval_answers = std::mem::take(&mut self.input_answers);
        self.first_new = self.history.len();
        let mut effects = Vec::new();
        if let Some(first) = requirements.first() {
            self.pending_call = Some(first.call.clone());
            let ask = lk_agents_abi::agent::run::RunQuestion {
                question: first.question.clone(),
                verbatim: first.verbatim,
            };
            self.input_requirements = requirements;
            self.ask_from_tool(ask, now, &mut effects);
        } else {
            self.next_step(now, &mut effects);
        }
        effects
    }

    /// Import acknowledged dialogue as context without crediting it to this invocation's
    /// output or granting approval/input authority. Existing own messages keep their boundary.
    pub fn observe_context(&mut self, messages: &[lk_agents_abi::ChatMessage]) {
        for message in messages {
            if !self.history.iter().any(|item| item.id() == message.id) {
                self.context_items.insert(message.id.clone());
                self.history.push(message.clone().into());
            }
        }
        if let Phase::Checkpoint(pending) = &mut self.phase {
            pending.observe_context(messages);
        }
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
