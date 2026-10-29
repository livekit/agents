//! An invocation a coordinator (VoiceSession) composes: it owns the record, plans each request
//! and decides each settled step's continuation.

use std::collections::BTreeSet;

use lk_agents_abi::{ChatMessage, Ts};

use super::{AgentRun, Effect, Next, Phase, StepId};

/// How an invocation opens, and who owns its record and its requests.
#[derive(Debug, Clone)]
pub enum Opening {
    /// Direct runs and TextSession: the incoming message, answered against the history, and every
    /// request planned here.
    Message(ChatMessage),
    /// VoiceSession: the coordinator owns the record and plans each request
    /// ([`Effect::PlanRequest`]), and decides each step's continuation ([`Effect::StepSettled`]).
    /// `approval_answers`: the read-backs the person's latest message answered.
    Coordinated { approval_answers: BTreeSet<String> },
}

impl Opening {
    pub fn message(&self) -> Option<&ChatMessage> {
        match self {
            Opening::Message(message) => Some(message),
            Opening::Coordinated { .. } => None,
        }
    }

    pub(super) fn coordinated(&self) -> bool {
        matches!(self, Opening::Coordinated { .. })
    }
}

/// The coordinator's verdict on a settled step ([`Effect::StepSettled`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepVerdict {
    /// The run goes on as it decided; a next model step offers its tools but may not call them
    /// when `tools_forbidden`.
    Continue { tools_forbidden: bool },
    /// No further model step: a run that would continue completes with nothing.
    End,
    /// The person cut the reply: the run ends interrupted, unless it already failed or accepted
    /// its output.
    Interrupted,
}

impl AgentRun {
    /// The person's latest answers: what a step that has not called yet grants with. The person
    /// moved on from the answers a step already calling read: they grant nothing more.
    pub(super) fn answered(&mut self, approval_answers: BTreeSet<String>) {
        if let Some(acting) = self.phase.acting() {
            let answers =
                if acting.calls.is_empty() { approval_answers.clone() } else { BTreeSet::new() };
            acting.calls.answered(answers);
        }
        self.approval_answers = approval_answers;
    }

    /// The coordinator's verdict on a settled step: it may end or forbid what the run would do
    /// next, never undo a failure or an accepted output.
    pub(super) fn verdict(
        &mut self,
        step: StepId,
        verdict: StepVerdict,
        now: Ts,
        fx: &mut Vec<Effect>,
    ) {
        if !matches!(&self.phase, Phase::Settling { step: s, .. } if *s == step) {
            return;
        }
        let Phase::Settling { next, .. } = std::mem::replace(&mut self.phase, Phase::Idle) else {
            unreachable!()
        };
        match (verdict, next) {
            (_, next @ (Next::Fail(_) | Next::Output(_))) => self.proceed(next, now, fx),
            (StepVerdict::Interrupted, _) => self.complete_interrupted(fx),
            (StepVerdict::End, Next::Continue) => self.end_without_output(fx),
            (StepVerdict::Continue { tools_forbidden }, Next::Continue) => {
                self.tools_forbidden = tools_forbidden;
                self.next_step(now, fx)
            }
            (_, next) => self.proceed(next, now, fx),
        }
    }
}
